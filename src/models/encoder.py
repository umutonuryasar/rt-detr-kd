"""Hybrid Encoder for RT-DETR.

Combines CNN feature fusion (RepCSP-style) with a lightweight Transformer
encoder to produce a flat sequence of encoded tokens from multi-scale
backbone features.

Architecture:
  1. Project each backbone scale to hidden_dim via 1x1 conv + BN + activation.
  2. Top-down feature fusion (FPN-style) using RepCSP blocks.
  3. Flatten all spatial positions and concatenate to form token sequence.
  4. Apply standard Transformer encoder layers.
  5. Return encoded sequence [B, N_total, hidden_dim].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class ConvBnAct(nn.Module):
    """Conv2d -> BatchNorm2d -> Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RepCSPBottleneck(nn.Module):
    """Single bottleneck block used inside RepCSP."""

    def __init__(self, channels: int, expansion: float = 0.5):
        super().__init__()
        hidden = int(channels * expansion)
        self.cv1 = ConvBnAct(channels, hidden, 1)
        self.cv2 = ConvBnAct(hidden, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class RepCSP(nn.Module):
    """Simplified CSP block for cross-scale feature fusion.

    Splits the channel dimension, processes one branch through N bottleneck
    blocks, then concatenates and projects back to out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3):
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = ConvBnAct(in_channels, hidden, 1)
        self.cv2 = ConvBnAct(in_channels, hidden, 1)
        self.bottlenecks = nn.Sequential(
            *[RepCSPBottleneck(hidden) for _ in range(num_blocks)]
        )
        self.cv3 = ConvBnAct(2 * hidden, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.bottlenecks(self.cv1(x))
        branch2 = self.cv2(x)
        return self.cv3(torch.cat([branch1, branch2], dim=1))


# ---------------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer (pre-norm)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        # Pre-norm self-attention
        residual = src
        src = self.norm1(src)
        src2, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = residual + self.dropout(src2)
        # Pre-norm feed-forward
        residual = src
        src = residual + self.ff(self.norm2(src))
        return src


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding for 2-D feature maps
# ---------------------------------------------------------------------------

def build_2d_sincos_pos_embed(h: int, w: int, d_model: int, device=None) -> torch.Tensor:
    """Build 2-D sinusoidal positional encodings.

    Returns tensor of shape [1, h*w, d_model].
    """
    assert d_model % 4 == 0, "d_model must be divisible by 4 for 2-D pos embed"
    grid_y = torch.arange(h, dtype=torch.float32, device=device)
    grid_x = torch.arange(w, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")  # [h, w]

    dim_half = d_model // 4
    omega = torch.arange(dim_half, dtype=torch.float32, device=device) / dim_half
    omega = 1.0 / (10000 ** omega)  # [dim_half]

    # y encodings
    out_y = grid_y.flatten().unsqueeze(1) * omega.unsqueeze(0)  # [h*w, dim_half]
    # x encodings
    out_x = grid_x.flatten().unsqueeze(1) * omega.unsqueeze(0)  # [h*w, dim_half]

    pos_embed = torch.cat(
        [out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1
    )  # [h*w, d_model]
    return pos_embed.unsqueeze(0)  # [1, h*w, d_model]


# ---------------------------------------------------------------------------
# Hybrid Encoder
# ---------------------------------------------------------------------------

class HybridEncoder(nn.Module):
    """Hybrid encoder combining CSP feature fusion with Transformer encoder.

    Processes multi-scale backbone features and outputs a single flat sequence
    of encoded tokens suitable for the RT-DETR decoder.

    Args:
        in_channels: List of channel counts for each backbone scale [C3, C4, C5].
        hidden_dim: Common feature dimension throughout the encoder.
        num_encoder_layers: Number of Transformer encoder layers.
        nhead: Number of attention heads in the Transformer.
        dim_feedforward: FF hidden size in Transformer encoder.
        dropout: Dropout rate.
        num_csp_blocks: Number of RepCSP bottleneck blocks.
    """

    def __init__(
        self,
        in_channels: list[int] = [128, 256, 512],
        hidden_dim: int = 256,
        num_encoder_layers: int = 1,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_csp_blocks: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = len(in_channels)

        # ---------- input projections ----------
        self.input_proj = nn.ModuleList(
            [ConvBnAct(c, hidden_dim, 1) for c in in_channels]
        )

        # ---------- top-down fusion (FPN-style) ----------
        # Fuse C5 into C4, then C4 into C3
        self.fusion_c4 = RepCSP(hidden_dim * 2, hidden_dim, num_csp_blocks)
        self.fusion_c3 = RepCSP(hidden_dim * 2, hidden_dim, num_csp_blocks)

        # ---------- Transformer encoder ----------
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode multi-scale backbone features.

        Follows the RT-DETR AIFI + CCFF design:
          1. Project all scales to hidden_dim.
          2. Apply Transformer encoder ONLY on C5 (coarsest, fewest tokens).
          3. Top-down CSP fusion: C5_enc -> C4, C4_fused -> C3.
          4. Return concatenation of all 3 encoded scales.

        Args:
            features: Dict from backbone with keys '0', '1', '2' mapping to
                      C3, C4, C5 feature tensors respectively.

        Returns:
            Encoded token sequence of shape [B, N_total, hidden_dim] where
            N_total = H3*W3 + H4*W4 + H5*W5.
        """
        c3 = self.input_proj[0](features["0"])  # [B, D, H/8,  W/8]
        c4 = self.input_proj[1](features["1"])  # [B, D, H/16, W/16]
        c5 = self.input_proj[2](features["2"])  # [B, D, H/32, W/32]

        # --- AIFI: Transformer encoder on C5 only (400 tokens at 640x640) ---
        B, D, H5, W5 = c5.shape
        pos5 = build_2d_sincos_pos_embed(H5, W5, D, device=c5.device)  # [1, H5*W5, D]
        tokens5 = c5.flatten(2).permute(0, 2, 1) + pos5               # [B, H5*W5, D]
        for layer in self.encoder_layers:
            tokens5 = layer(tokens5)
        tokens5 = self.encoder_norm(tokens5)
        c5_enc = tokens5.permute(0, 2, 1).reshape(B, D, H5, W5)       # back to spatial

        # --- CCFF: top-down CSP fusion ---
        c5_up = F.interpolate(c5_enc, size=c4.shape[-2:], mode="nearest")
        c4_fused = self.fusion_c4(torch.cat([c4, c5_up], dim=1))

        c4_up = F.interpolate(c4_fused, size=c3.shape[-2:], mode="nearest")
        c3_fused = self.fusion_c3(torch.cat([c3, c4_up], dim=1))

        # Flatten C4 + C5 only as decoder memory (C3 is used for fusion only).
        # C4: H/16 * W/16 = 1600 tokens at 640x640
        # C5: H/32 * W/32 = 400 tokens at 640x640  → total 2000, manageable on 4 GB.
        tokens = self._flatten_and_embed([c4_fused, c5_enc])
        return tokens  # [B, N_total, D]

    def _flatten_and_embed(self, feature_maps: list[torch.Tensor]) -> torch.Tensor:
        """Flatten feature maps and append sinusoidal 2-D positional encodings."""
        parts = []
        for feat in feature_maps:
            B, C, H, W = feat.shape
            tokens = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            pos = build_2d_sincos_pos_embed(H, W, C, device=feat.device)  # [1, H*W, C]
            parts.append(tokens + pos)
        return torch.cat(parts, dim=1)  # [B, N_total, D]
