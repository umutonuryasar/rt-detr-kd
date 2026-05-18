"""Stage-Adaptive Knowledge Distillation loss.

Novel contribution: automatically transitions the KD objective from
feature-heavy (structural alignment) to logit-heavy (semantic refinement)
across training using a cosine annealing schedule.

Motivation: Early training benefits from feature alignment to guide the student
toward meaningful internal representations. Later training benefits from
logit-level semantic refinement when the student structure is already aligned.

Weight schedule:
    w_feat(e)  = cos( pi * e / (2 * E) )   [1 → 0 over training]
    w_logit(e) = 1 - w_feat(e)             [0 → 1 over training]

    L_KD(e) = w_feat(e) * L_feat + w_logit(e) * L_logit

where e is the current epoch and E is total epochs.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from .feature_kd import FeatureKDLoss
from .logit_kd import LogitKDLoss


class StageAdaptiveKDLoss(nn.Module):
    """Curriculum KD that shifts from feature to logit distillation.

    Args:
        feature_loss:  Instantiated FeatureKDLoss.
        logit_loss:    Instantiated LogitKDLoss.
        total_epochs:  Total number of training epochs (E in the schedule).
    """

    def __init__(
        self,
        feature_loss: FeatureKDLoss,
        logit_loss: LogitKDLoss,
        total_epochs: int = 36,
    ):
        super().__init__()
        if total_epochs <= 0:
            raise ValueError(f"total_epochs must be > 0, got {total_epochs}")
        self.feature_loss = feature_loss
        self.logit_loss = logit_loss
        self.total_epochs = total_epochs

    def _weights(self, epoch: int) -> tuple[float, float]:
        """Return (w_feat, w_logit) for the given epoch."""
        e = max(0, min(epoch, self.total_epochs))
        w_feat = math.cos(math.pi * e / (2 * self.total_epochs))
        w_logit = 1.0 - w_feat
        return w_feat, w_logit

    def forward(
        self,
        epoch: int,
        student_enc: torch.Tensor,
        teacher_enc: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_attn: Optional[torch.Tensor] = None,
        teacher_attn: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute stage-adaptive KD loss.

        Args:
            epoch:          Current training epoch (0-indexed).
            student_enc:    [B, N_s, D_s] — student encoder output.
            teacher_enc:    [B, N_t, D_t] — teacher encoder output (detached).
            student_logits: [B, Q_s, num_classes] — student class logits.
            teacher_logits: [B, Q_t, num_classes] — teacher class logits (detached).
            student_attn:   [L, B, H, Q_s, N_s] or None.
            teacher_attn:   [L, B, H, Q_t, N_t] or None.

        Returns:
            Dict with scalar losses:
              'loss_feat':   encoder MSE component.
              'loss_attn':   attention cosine component.
              'loss_logit':  logit KL divergence component.
              'loss_kd':     w_feat * L_feat_total + w_logit * L_logit.
              'w_feat':      current feature weight (for logging).
              'w_logit':     current logit weight (for logging).
        """
        w_feat, w_logit = self._weights(epoch)

        feat_losses = self.feature_loss(
            student_enc, teacher_enc, student_attn, teacher_attn
        )
        loss_logit = self.logit_loss(student_logits, teacher_logits)

        loss_kd = w_feat * feat_losses["loss_kd"] + w_logit * loss_logit

        return {
            "loss_feat": feat_losses["loss_feat"],
            "loss_attn": feat_losses["loss_attn"],
            "loss_logit": loss_logit,
            "loss_kd": loss_kd,
            "w_feat": torch.tensor(w_feat, device=student_enc.device),
            "w_logit": torch.tensor(w_logit, device=student_enc.device),
        }

    def extra_repr(self) -> str:
        return f"total_epochs={self.total_epochs}"
