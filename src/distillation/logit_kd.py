"""Logit-level Knowledge Distillation loss.

RT-DETR (like most DETR-family detectors) trains its classification head with
a *per-class sigmoid* focal loss — logits are independent binary scores, not
a categorical distribution. Two KD modes are therefore provided:

  mode="binary" (default, recommended):
      Per-class binary KL divergence between temperature-scaled sigmoid
      probabilities. This matches the distribution family the logits were
      actually trained under.

        p_t = sigmoid(t / T),  p_s = sigmoid(s / T)
        L = T^2 * mean_c[ p_t*log(p_t/p_s) + (1-p_t)*log((1-p_t)/(1-p_s)) ]

  mode="softmax" (legacy / ablation):
      Hinton et al. (2015) categorical KL over class-dimension softmax.
      Kept for ablation against the binary formulation; note it imposes a
      categorical structure that sigmoid-focal-trained logits do not have.

        L = T^2 * KL( softmax(t/T) || softmax(s/T) )

Only the classification head logits are distilled; bounding-box regression is
excluded because teacher/student query correspondence is not guaranteed here.

When the teacher and student have a different number of queries the first
min(Q_s, Q_t) queries are used. NOTE: index-wise truncation is a weak
correspondence for query-level signals in general; it is tolerable here only
because the binary KL is computed per (query, class) marginal rather than
matching specific objects. For object-level alignment see QueryKDLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_LOGIT_KD_MODES = ("binary", "softmax")


class LogitKDLoss(nn.Module):
    """KL-divergence distillation on classification logits.

    Args:
        temperature: Temperature T. Higher values produce softer
                     probability distributions. Typical values: {2, 4, 8}.
        mode:        "binary" (per-class sigmoid KL — matches focal-sigmoid
                     training, default) or "softmax" (Hinton categorical KL,
                     kept for ablation).
    """

    def __init__(self, temperature: float = 4.0, mode: str = "binary"):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if mode not in SUPPORTED_LOGIT_KD_MODES:
            raise ValueError(
                f"mode must be one of {SUPPORTED_LOGIT_KD_MODES}, got '{mode}'"
            )
        self.T = temperature
        self.mode = mode
        # KLDiv expects log-probabilities for the input and probabilities for
        # the target; reduction='batchmean' divides by batch size (standard).
        self.kl_div = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logit KD loss.

        Args:
            student_logits: [B, Q_s, num_classes]  — raw student class logits.
            teacher_logits: [B, Q_t, num_classes]  — raw teacher class logits
                            (must be detached before calling if not already).

        Returns:
            Scalar loss value.
        """
        # Align query counts
        Q_s = student_logits.size(1)
        Q_t = teacher_logits.size(1)
        Q = min(Q_s, Q_t)
        s = student_logits[:, :Q, :]  # [B, Q, C]
        t = teacher_logits[:, :Q, :]  # [B, Q, C]

        # Flatten batch and query dimensions -> [B*Q, C]
        s = s.reshape(-1, s.size(-1))
        t = t.reshape(-1, t.size(-1)).detach()

        if self.mode == "binary":
            return self._binary_kl(s, t)
        return self._softmax_kl(s, t)

    def _binary_kl(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Per-class binary KL between temperature-scaled sigmoid probs.

        Numerically stable formulation via BCE-with-logits:
          KL(p_t || p_s) = H(p_t, p_s) - H(p_t)
        where H(p_t, p_s) = BCEWithLogits(s/T, p_t) and H(p_t) is the entropy
        of the (constant) teacher distribution. The entropy term carries no
        student gradient but is included so the loss is a true KL (>= 0,
        equals 0 iff distributions match) — useful for logging/monitoring.
        """
        s_scaled = s / self.T
        p_t = torch.sigmoid(t / self.T)

        # Cross-entropy of teacher targets under student sigmoid.
        ce = F.binary_cross_entropy_with_logits(s_scaled, p_t, reduction="mean")
        # Teacher entropy (constant w.r.t. student parameters).
        eps = 1e-6
        p_t_c = p_t.clamp(eps, 1.0 - eps)
        h_t = -(p_t_c * p_t_c.log() + (1 - p_t_c) * (1 - p_t_c).log()).mean()

        return (ce - h_t) * (self.T ** 2)

    def _softmax_kl(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Legacy Hinton et al. (2015) categorical KL (kept for ablation)."""
        s_log_prob = F.log_softmax(s / self.T, dim=-1)   # log P_s
        t_prob = F.softmax(t / self.T, dim=-1)            # P_t (target)
        # KL(P_t || P_s) * T^2 (restores gradient magnitude per Hinton et al.)
        return self.kl_div(s_log_prob, t_prob) * (self.T ** 2)

    def extra_repr(self) -> str:
        return f"temperature={self.T}, mode={self.mode}"
