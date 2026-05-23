"""Stage-Adaptive Knowledge Distillation loss.

Novel contribution: automatically transitions the KD objective from
feature-heavy (structural alignment) to logit-heavy (semantic refinement)
across training. The default is a cosine annealing schedule; the schedule
shape itself is an ablatable design choice — see ``schedule`` argument.

Motivation: Early training benefits from feature alignment to guide the student
toward meaningful internal representations. Later training benefits from
logit-level semantic refinement when the student structure is already aligned.

Schedules (with ``e`` the current epoch and ``E`` total epochs):

    cosine          w_feat = cos(π e / (2E))                    [1 → 0]
    linear          w_feat = 1 - e/E                            [1 → 0]
    step            w_feat = 1 if e < E/2 else 0                [hard switch at midpoint]
    sigmoid         w_feat = σ(k (E/2 - e) / E),  k = 10        [smooth transition around E/2]
    inverse_cosine  w_feat = sin(π e / (2E))                    [0 → 1, control / sanity check]

The ``inverse_cosine`` variant is the curriculum-direction control: if it
performs comparably to ``cosine`` the curriculum direction is not what is
driving the effect, weakening the contribution.

    L_KD(e) = w_feat(e) · L_feat + w_logit(e) · L_logit
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from .feature_kd import FeatureKDLoss
from .logit_kd import LogitKDLoss


SUPPORTED_SCHEDULES = ("cosine", "linear", "step", "sigmoid", "inverse_cosine")


class StageAdaptiveKDLoss(nn.Module):
    """Curriculum KD that shifts from feature to logit distillation.

    Args:
        feature_loss:  Instantiated FeatureKDLoss.
        logit_loss:    Instantiated LogitKDLoss.
        total_epochs:  Total number of training epochs (E in the schedule).
        schedule:      One of SUPPORTED_SCHEDULES.
        sigmoid_k:     Steepness for the ``sigmoid`` schedule (default 10).
                       Higher = more step-like.
    """

    def __init__(
        self,
        feature_loss: FeatureKDLoss,
        logit_loss: LogitKDLoss,
        total_epochs: int = 36,
        schedule: str = "cosine",
        sigmoid_k: float = 10.0,
    ):
        super().__init__()
        if total_epochs <= 0:
            raise ValueError(f"total_epochs must be > 0, got {total_epochs}")
        if schedule not in SUPPORTED_SCHEDULES:
            raise ValueError(
                f"schedule must be one of {SUPPORTED_SCHEDULES}, got '{schedule}'"
            )
        self.feature_loss = feature_loss
        self.logit_loss = logit_loss
        self.total_epochs = total_epochs
        self.schedule = schedule
        self.sigmoid_k = sigmoid_k

    def _weights(self, epoch: int) -> tuple[float, float]:
        """Return (w_feat, w_logit) for the given epoch under self.schedule."""
        e = max(0, min(epoch, self.total_epochs))
        E = self.total_epochs

        if self.schedule == "cosine":
            w_feat = math.cos(math.pi * e / (2 * E))
        elif self.schedule == "linear":
            w_feat = 1.0 - e / E
        elif self.schedule == "step":
            w_feat = 1.0 if e < E / 2 else 0.0
        elif self.schedule == "sigmoid":
            # Centered at E/2 with steepness sigmoid_k.
            x = self.sigmoid_k * (E / 2 - e) / E
            w_feat = 1.0 / (1.0 + math.exp(-x))
        elif self.schedule == "inverse_cosine":
            # 0 → 1 (logit-heavy early) — curriculum-direction control.
            w_feat = math.sin(math.pi * e / (2 * E))
        else:
            raise AssertionError(self.schedule)

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
        return f"total_epochs={self.total_epochs}, schedule={self.schedule}"
