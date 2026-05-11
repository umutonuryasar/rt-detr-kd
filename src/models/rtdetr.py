"""Full RT-DETR model assembling backbone, encoder, and decoder.

After a forward pass the following intermediate features are stored as
attributes on the model instance, making them available for KD without hooks:
  - ``self.encoder_output``: [B, N, hidden_dim] — encoder token sequence.
  - ``self.attn_maps``:       list of per-layer cross-attention tensors from
                               the decoder (see RTDETRDecoder.attn_maps).
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbone import ResNetBackbone, BACKBONE_OUT_CHANNELS
from .encoder import HybridEncoder
from .decoder import RTDETRDecoder


class RTDETR(nn.Module):
    """RT-DETR object detection model.

    Args:
        backbone_name: 'resnet18' (student) or 'resnet50' (teacher).
        num_classes: Number of object categories (default: 80 for COCO).
        hidden_dim: Transformer hidden dimension.
        num_queries: Number of object queries for the decoder.
        num_decoder_layers: Number of stacked decoder layers.
        nhead: Attention heads in encoder and decoder.
        dim_feedforward: Feed-forward width in Transformer layers.
        dropout: Dropout probability.
        num_encoder_layers: Number of Transformer encoder layers.
        pretrained_backbone: Load ImageNet-pretrained backbone weights.
        freeze_bn: Replace backbone BatchNorm with FrozenBatchNorm.
        freeze_stages: Number of early backbone stages to freeze.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_encoder_layers: int = 1,
        pretrained_backbone: bool = True,
        freeze_bn: bool = False,
        freeze_stages: int = 1,
    ):
        super().__init__()

        self.backbone = ResNetBackbone(
            name=backbone_name,
            pretrained=pretrained_backbone,
            freeze_bn=freeze_bn,
            freeze_stages=freeze_stages,
        )
        backbone_out_ch = BACKBONE_OUT_CHANNELS[backbone_name]

        self.encoder = HybridEncoder(
            in_channels=backbone_out_ch,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.decoder = RTDETRDecoder(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Will be populated after every forward pass
        self.encoder_output: Optional[torch.Tensor] = None
        self.attn_maps: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            images: Input batch [B, 3, H, W].

        Returns:
            Dict with 'pred_logits' [B, Q, num_classes] and
            'pred_boxes' [B, Q, 4].

        Side effects:
            Sets ``self.encoder_output`` and ``self.attn_maps`` for KD access.
        """
        # Backbone: multi-scale features
        features = self.backbone(images)

        # Encoder: flat token sequence
        enc_out = self.encoder(features)  # [B, N, D]
        self.encoder_output = enc_out  # store for KD

        # Decoder: predictions
        outputs = self.decoder(enc_out)
        self.attn_maps = self.decoder.attn_maps  # list of [B, H, Q, N]

        return outputs

    def get_attn_maps_tensor(self) -> Optional[torch.Tensor]:
        """Return stacked cross-attention maps [L, B, H, Q, N] or None."""
        return self.decoder.get_attn_maps_tensor()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_rtdetr(cfg: dict) -> "RTDETR":
    """Build an RTDETR model from a config dictionary.

    Args:
        cfg: Dict with at minimum 'backbone' and optionally any other
             RTDETR __init__ keyword argument.

    Returns:
        Initialized RTDETR model.
    """
    model_cfg = cfg.get("model", cfg)
    return RTDETR(
        backbone_name=model_cfg.get("backbone", "resnet18"),
        num_classes=model_cfg.get("num_classes", 80),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_queries=model_cfg.get("num_queries", 300),
        num_decoder_layers=model_cfg.get("num_decoder_layers", 6),
        nhead=model_cfg.get("nhead", 8),
        dim_feedforward=model_cfg.get("dim_feedforward", 1024),
        dropout=model_cfg.get("dropout", 0.0),
        num_encoder_layers=model_cfg.get("num_encoder_layers", 1),
    )
