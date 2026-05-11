"""Logit-level Knowledge Distillation loss.

Applies temperature-scaled KL divergence between student and teacher
classification logits. Only the classification head logits are distilled;
bounding-box regression is excluded because the teacher and student may
predict different sets of boxes (query alignment is not guaranteed).

Loss formula (Hinton et al., 2015):
    L_logit = T² * KL( softmax(t_logits / T) || softmax(s_logits / T) )

where KL is computed in the forward direction (teacher as target distribution).

When the teacher and student have a different number of queries the first
min(Q_s, Q_t) queries are used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitKDLoss(nn.Module):
    """KL-divergence distillation on classification logits.

    Args:
        temperature: Softmax temperature T. Higher values produce softer
                     probability distributions, encouraging the student to
                     match inter-class relationships. Typical values: {2, 4, 8}.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.T = temperature
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
        t = t.reshape(-1, t.size(-1))

        # Temperature-scaled distributions
        s_log_prob = F.log_softmax(s / self.T, dim=-1)   # log P_s
        t_prob = F.softmax(t / self.T, dim=-1)            # P_t (target)

        # KL(P_t || P_s) * T² (restores gradient magnitude per Hinton et al.)
        loss = self.kl_div(s_log_prob, t_prob) * (self.T ** 2)
        return loss

    def extra_repr(self) -> str:
        return f"temperature={self.T}"
