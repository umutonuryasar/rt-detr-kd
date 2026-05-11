"""RT-DETR Transformer Decoder.

Implements an end-to-end Transformer decoder that attends to encoder memory.
Cross-attention maps are stored as attributes after each forward pass to
support Knowledge Distillation without requiring forward hooks.

Architecture per layer:
  1. Self-attention on queries.
  2. Cross-attention between queries and encoder memory  -> stored attn map.
  3. Feed-forward network.

After all decoder layers:
  - Classification head: Linear(hidden_dim, num_classes)
  - Bounding-box head:  MLP -> sigmoid -> [cx, cy, w, h] in [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ---------------------------------------------------------------------------
# Positional encoding for queries (learned)
# ---------------------------------------------------------------------------

class LearnedQueryEmbed(nn.Module):
    """Learned positional embeddings for object queries."""

    def __init__(self, num_queries: int, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_queries, hidden_dim))
        nn.init.uniform_(self.weight)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.weight.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# MLP head for bounding-box regression
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class RTDETRDecoderLayer(nn.Module):
    """Single RT-DETR decoder layer.

    Stores the cross-attention weight tensor after each forward pass as
    ``self.cross_attn_weights`` with shape [B, nhead, num_queries, N_mem].
    """

    def __init__(
        self,
        hidden_dim: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop2 = nn.Dropout(dropout)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Will be populated during forward
        self.cross_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single decoder layer forward.

        Args:
            queries: [B, num_queries, D]
            memory:  [B, N_mem, D]  — encoder output
            query_pos: Optional positional encoding added to queries for
                       self- and cross-attention keys/queries.
            memory_key_padding_mask: Optional mask for encoder tokens.

        Returns:
            Updated queries [B, num_queries, D].
        """
        # Add query positional encoding
        q = queries if query_pos is None else queries + query_pos
        k = q  # self-attention: key = query

        # 1. Self-attention (pre-norm)
        residual = queries
        q_norm = self.norm1(q)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = residual + self.drop1(sa_out)

        # 2. Cross-attention (pre-norm)
        residual = queries
        q_ca = self.norm2(queries)
        if query_pos is not None:
            q_ca = q_ca + query_pos
        ca_out, attn_w = self.cross_attn(
            q_ca,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # keep per-head weights
        )
        # attn_w shape: [B, nhead, num_queries, N_mem]
        self.cross_attn_weights = attn_w.detach()
        queries = residual + self.drop2(ca_out)

        # 3. Feed-forward (pre-norm)
        residual = queries
        queries = residual + self.ff(self.norm3(queries))

        return queries


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------

class RTDETRDecoder(nn.Module):
    """RT-DETR Transformer decoder.

    After each forward pass the following attributes are available:
      - ``self.attn_maps``: list of length num_decoder_layers, each element
        is a tensor [B, nhead, num_queries, N_mem] — the cross-attention
        weights from that layer.

    Args:
        num_classes: Number of object categories.
        hidden_dim: Transformer hidden dimension.
        num_queries: Number of object queries.
        num_decoder_layers: Number of stacked decoder layers.
        nhead: Number of attention heads.
        dim_feedforward: Feed-forward hidden size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers

        # Query content embeddings (learned)
        self.query_embed = LearnedQueryEmbed(num_queries, hidden_dim)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                RTDETRDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Prediction heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # Storage for KD
        self.attn_maps: list[torch.Tensor] = []

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)
        # Bias initialization for classification: prior probability ~0.01
        prior_prob = 0.01
        bias_val = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_head.bias, bias_val)

    def forward(self, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode object queries from encoder memory.

        Args:
            memory: Encoder output [B, N_mem, hidden_dim].

        Returns:
            Dictionary with:
              'pred_logits': [B, num_queries, num_classes]
              'pred_boxes':  [B, num_queries, 4]  (cx, cy, w, h in [0,1])
        """
        B = memory.size(0)
        queries = self.query_embed(B)  # [B, num_queries, D]

        # Reset attention map storage
        self.attn_maps = []

        for layer in self.layers:
            queries = layer(queries, memory)
            if layer.cross_attn_weights is not None:
                self.attn_maps.append(layer.cross_attn_weights)

        queries = self.norm(queries)

        pred_logits = self.class_head(queries)  # [B, Q, num_classes]
        pred_boxes = self.bbox_head(queries).sigmoid()  # [B, Q, 4]

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }

    def get_attn_maps_tensor(self) -> Optional[torch.Tensor]:
        """Stack all layer attention maps.

        Returns:
            Tensor of shape [L, B, nhead, num_queries, N_mem] or None if empty.
        """
        if not self.attn_maps:
            return None
        return torch.stack(self.attn_maps, dim=0)  # [L, B, H, Q, N]
