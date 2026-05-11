"""Combined Knowledge Distillation loss wrapper.

L_total = L_det + kd_lambda * L_KD

where L_det is the standard RT-DETR detection loss and L_KD is either the
logit-KD or feature-KD loss depending on configuration.

Returns a unified dict with all individual loss components so every term can
be logged independently to TensorBoard.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..losses.detection_loss import RTDETRLoss
from .logit_kd import LogitKDLoss
from .feature_kd import FeatureKDLoss


class KDLoss(nn.Module):
    """Unified KD loss combining detection loss and distillation loss.

    Args:
        kd_type:   'logit' or 'feature'.
        kd_lambda: Weight multiplier for the KD term (λ).
        temperature: Softmax temperature for logit KD (ignored for feature KD).
        alpha:     Attention weight for feature KD (ignored for logit KD).
        num_classes: Number of object categories.
        student_dim: Student encoder feature dimension (feature KD only).
        teacher_dim: Teacher encoder feature dimension (feature KD only).
    """

    def __init__(
        self,
        kd_type: str = "logit",
        kd_lambda: float = 1.0,
        temperature: float = 4.0,
        alpha: float = 0.5,
        num_classes: int = 80,
        student_dim: int = 256,
        teacher_dim: int = 256,
    ):
        super().__init__()
        assert kd_type in ("logit", "feature"), (
            f"kd_type must be 'logit' or 'feature', got '{kd_type}'"
        )

        self.kd_type = kd_type
        self.kd_lambda = kd_lambda

        self.detection_loss = RTDETRLoss(num_classes=num_classes)

        if kd_type == "logit":
            self.kd_loss_fn: nn.Module = LogitKDLoss(temperature=temperature)
        else:
            self.kd_loss_fn = FeatureKDLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                alpha=alpha,
            )

    def forward(
        self,
        model_outputs: dict,
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Compute total loss.

        Args:
            model_outputs: Output dict from RTDETRWithKD.forward():
                'student':        {'pred_logits', 'pred_boxes'}
                'teacher':        {'pred_logits', 'pred_boxes'}
                'student_enc_out': [B, N_s, D_s]
                'teacher_enc_out': [B, N_t, D_t]
                'student_attn':    [L, B, H, Q_s, N_s] or None
                'teacher_attn':    [L, B, H, Q_t, N_t] or None
            targets: List of B dicts with 'labels' and 'boxes'.

        Returns:
            Dict with the following scalar tensors:
              'loss_ce':    focal classification loss
              'loss_bbox':  L1 bbox loss
              'loss_giou':  GIoU bbox loss
              'loss_det':   weighted sum of detection losses
              'loss_kd':    knowledge distillation loss (pre-lambda scaling)
              'loss_total': loss_det + kd_lambda * loss_kd
              (for feature KD also:)
              'loss_feat':  encoder MSE
              'loss_attn':  attention cosine loss
        """
        student_out = model_outputs["student"]
        teacher_out = model_outputs["teacher"]

        # ---- Detection loss (student predictions vs ground truth) ----
        det_losses = self.detection_loss(student_out, targets)

        # ---- Distillation loss ----
        if self.kd_type == "logit":
            loss_kd = self.kd_loss_fn(
                student_out["pred_logits"],
                teacher_out["pred_logits"],
            )
            kd_dict: dict[str, torch.Tensor] = {"loss_kd": loss_kd}

        else:  # feature KD
            student_enc = model_outputs["student_enc_out"]
            teacher_enc = model_outputs["teacher_enc_out"]
            student_attn = model_outputs.get("student_attn")
            teacher_attn = model_outputs.get("teacher_attn")

            feat_losses = self.kd_loss_fn(
                student_enc, teacher_enc, student_attn, teacher_attn
            )
            kd_dict = {
                "loss_feat": feat_losses["loss_feat"],
                "loss_attn": feat_losses["loss_attn"],
                "loss_kd": feat_losses["loss_kd"],
            }

        # ---- Combined total loss ----
        loss_total = det_losses["loss_det"] + self.kd_lambda * kd_dict["loss_kd"]

        return {
            **det_losses,
            **kd_dict,
            "loss_total": loss_total,
        }

    def extra_repr(self) -> str:
        return f"kd_type={self.kd_type}, kd_lambda={self.kd_lambda}"
