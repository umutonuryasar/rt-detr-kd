"""Unified Knowledge Distillation loss wrapper.

L_total = L_det + kd_lambda * L_KD

Supports seven distillation strategies:
  logit          — KL divergence on classification logits (Hinton et al., 2015)
  feature        — Encoder MSE + decoder cross-attention cosine similarity
  combined       — Logit + Feature simultaneously, with tunable weights
  cwd            — Channel-Wise Distillation (Yang et al., ICCV 2021)
  mgd            — Masked Generative Distillation (Yang et al., ECCV 2022)
  query          — Decoder object query distillation (novel, RT-DETR-specific)
  stage_adaptive — Cosine curriculum shift from feature to logit (novel)

All individual loss components are returned in the output dict so every term
can be logged independently to TensorBoard / W&B.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..losses.detection_loss import RTDETRLoss
from .logit_kd import LogitKDLoss
from .feature_kd import FeatureKDLoss
from .cwd import CWDLoss
from .mgd import MGDLoss
from .query_kd import QueryKDLoss
from .stage_adaptive_kd import StageAdaptiveKDLoss

SUPPORTED_KD_TYPES = (
    "logit", "feature", "combined", "cwd", "mgd", "query", "stage_adaptive"
)


class KDLoss(nn.Module):
    """Unified KD loss combining detection loss and distillation loss.

    Args:
        kd_type:        One of SUPPORTED_KD_TYPES.
        kd_lambda:      Weight multiplier for the KD term (λ).
        temperature:    Softmax temperature for logit/combined/stage_adaptive KD.
        alpha:          Attention weight for feature/combined/query/stage_adaptive KD.
        feat_weight:    Encoder MSE weight within feature KD (0.0 = attention-only).
        logit_weight:   Logit component weight within combined KD.
        feature_weight: Feature component weight within combined KD.
        num_classes:    Number of object categories.
        student_dim:    Student encoder/decoder feature dimension.
        teacher_dim:    Teacher encoder/decoder feature dimension.
        tau:            Spatial softmax temperature for CWD.
        mask_ratio:     Token mask ratio for MGD.
        total_epochs:   Total training epochs for stage_adaptive schedule.
    """

    def __init__(
        self,
        kd_type: str = "logit",
        kd_lambda: float = 1.0,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feat_weight: float = 1.0,
        logit_weight: float = 0.5,
        feature_weight: float = 0.5,
        num_classes: int = 80,
        student_dim: int = 256,
        teacher_dim: int = 256,
        tau: float = 1.0,
        mask_ratio: float = 0.75,
        total_epochs: int = 36,
        schedule: str = "cosine",
    ):
        super().__init__()
        if kd_type not in SUPPORTED_KD_TYPES:
            raise ValueError(
                f"kd_type must be one of {SUPPORTED_KD_TYPES}, got '{kd_type}'"
            )

        self.kd_type = kd_type
        self.kd_lambda = kd_lambda
        self.logit_weight = logit_weight
        self.feature_weight = feature_weight

        self.detection_loss = RTDETRLoss(num_classes=num_classes)

        if kd_type == "logit":
            self.kd_loss_fn = LogitKDLoss(temperature=temperature)

        elif kd_type == "feature":
            self.kd_loss_fn = FeatureKDLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                alpha=alpha,
                feat_weight=feat_weight,
            )

        elif kd_type == "combined":
            self.logit_loss_fn = LogitKDLoss(temperature=temperature)
            self.feature_loss_fn = FeatureKDLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                alpha=alpha,
                feat_weight=feat_weight,
            )

        elif kd_type == "cwd":
            self.kd_loss_fn = CWDLoss(
                student_channels=student_dim,
                teacher_channels=teacher_dim,
                tau=tau,
            )

        elif kd_type == "mgd":
            self.kd_loss_fn = MGDLoss(
                student_channels=student_dim,
                teacher_channels=teacher_dim,
                mask_ratio=mask_ratio,
            )

        elif kd_type == "query":
            self.kd_loss_fn = QueryKDLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                alpha=alpha,
            )

        elif kd_type == "stage_adaptive":
            feature_loss = FeatureKDLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                alpha=alpha,
                feat_weight=feat_weight,
            )
            logit_loss = LogitKDLoss(temperature=temperature)
            self.kd_loss_fn = StageAdaptiveKDLoss(
                feature_loss=feature_loss,
                logit_loss=logit_loss,
                total_epochs=total_epochs,
                schedule=schedule,
            )

    def forward(
        self,
        model_outputs: dict,
        targets: list[dict[str, torch.Tensor]],
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Compute total loss.

        Args:
            model_outputs: Output dict from RTDETRWithKD.forward():
                'student':         {'pred_logits', 'pred_boxes'}
                'teacher':         {'pred_logits', 'pred_boxes'}
                'student_enc_out': [B, N_s, D_s]
                'teacher_enc_out': [B, N_t, D_t]
                'student_attn':    [L, B, H, Q_s, N_s] or None
                'teacher_attn':    [L, B, H, Q_t, N_t] or None
                'student_queries': [B, Q_s, D_s] or None
                'teacher_queries': [B, Q_t, D_t] or None
            targets: List of B dicts with 'labels' and 'boxes'.
            epoch:   Current training epoch (used by stage_adaptive only).

        Returns:
            Dict with scalar tensors. Always present:
              'loss_ce', 'loss_bbox', 'loss_giou', 'loss_det'
              'loss_kd', 'loss_total'
            Additional keys depending on kd_type:
              feature/combined/stage_adaptive: 'loss_feat', 'loss_attn'
              combined: also 'loss_logit'
              stage_adaptive: also 'loss_logit', 'w_feat', 'w_logit'
              query: 'loss_query', 'loss_query_attn'
        """
        student_out = model_outputs["student"]
        teacher_out = model_outputs["teacher"]

        # ---- Detection loss (student predictions vs ground truth) ----
        det_losses = self.detection_loss(student_out, targets)

        # ---- Distillation loss ----
        kd_dict: dict[str, torch.Tensor]

        if self.kd_type == "logit":
            loss_kd = self.kd_loss_fn(
                student_out["pred_logits"],
                teacher_out["pred_logits"],
            )
            kd_dict = {"loss_kd": loss_kd}

        elif self.kd_type == "feature":
            feat_losses = self.kd_loss_fn(
                model_outputs["student_enc_out"],
                model_outputs["teacher_enc_out"],
                model_outputs.get("student_attn"),
                model_outputs.get("teacher_attn"),
            )
            kd_dict = {
                "loss_feat": feat_losses["loss_feat"],
                "loss_attn": feat_losses["loss_attn"],
                "loss_kd":   feat_losses["loss_kd"],
            }

        elif self.kd_type == "combined":
            loss_logit = self.logit_loss_fn(
                student_out["pred_logits"],
                teacher_out["pred_logits"],
            )
            feat_losses = self.feature_loss_fn(
                model_outputs["student_enc_out"],
                model_outputs["teacher_enc_out"],
                model_outputs.get("student_attn"),
                model_outputs.get("teacher_attn"),
            )
            loss_kd = self.logit_weight * loss_logit + self.feature_weight * feat_losses["loss_kd"]
            kd_dict = {
                "loss_logit": loss_logit,
                "loss_feat":  feat_losses["loss_feat"],
                "loss_attn":  feat_losses["loss_attn"],
                "loss_kd":    loss_kd,
            }

        elif self.kd_type in ("cwd", "mgd"):
            loss_kd = self.kd_loss_fn(
                model_outputs["student_enc_out"],
                model_outputs["teacher_enc_out"],
            )
            kd_dict = {"loss_kd": loss_kd}

        elif self.kd_type == "query":
            query_losses = self.kd_loss_fn(
                model_outputs["student_queries"],
                model_outputs["teacher_queries"],
                model_outputs.get("student_attn"),
                model_outputs.get("teacher_attn"),
            )
            kd_dict = {
                "loss_query":      query_losses["loss_query"],
                "loss_query_attn": query_losses["loss_query_attn"],
                "loss_kd":         query_losses["loss_kd"],
            }

        elif self.kd_type == "stage_adaptive":
            sa_losses = self.kd_loss_fn(
                epoch=epoch,
                student_enc=model_outputs["student_enc_out"],
                teacher_enc=model_outputs["teacher_enc_out"],
                student_logits=student_out["pred_logits"],
                teacher_logits=teacher_out["pred_logits"],
                student_attn=model_outputs.get("student_attn"),
                teacher_attn=model_outputs.get("teacher_attn"),
            )
            kd_dict = {
                "loss_feat":  sa_losses["loss_feat"],
                "loss_attn":  sa_losses["loss_attn"],
                "loss_logit": sa_losses["loss_logit"],
                "loss_kd":    sa_losses["loss_kd"],
                "w_feat":     sa_losses["w_feat"],
                "w_logit":    sa_losses["w_logit"],
            }

        # ---- Total loss ----
        loss_total = det_losses["loss_det"] + self.kd_lambda * kd_dict["loss_kd"]

        return {
            **det_losses,
            **kd_dict,
            "loss_total": loss_total,
        }

    def extra_repr(self) -> str:
        return f"kd_type={self.kd_type}, kd_lambda={self.kd_lambda}"
