"""Channel-Wise Distillation loss.

Yang et al., "Focal and Global Knowledge Distillation for Detectors", ICCV 2021.

Distills the channel-wise spatial distributions of encoder token sequences.
Each feature dimension D is treated as a "channel" with a distribution over
N spatial positions. KL divergence is applied between softmax-normalized
teacher and student channel distributions.

  L_CWD = sum_c KL( softmax_spatial(t_c / tau) || softmax_spatial(s_c / tau) )

A 1x1 Conv1d projection aligns student channels to teacher channels before
computing the loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CWDLoss(nn.Module):
    """Channel-Wise Distillation on encoder token sequences.

    Args:
        student_channels: Feature dimension of student encoder output (D_s).
        teacher_channels: Feature dimension of teacher encoder output (D_t).
        tau:              Temperature for spatial softmax normalization.
    """

    def __init__(
        self,
        student_channels: int = 256,
        teacher_channels: int = 256,
        tau: float = 1.0,
    ):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.tau = tau
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        self.align = nn.Conv1d(student_channels, teacher_channels, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.align.weight)

    def forward(
        self,
        student_enc: torch.Tensor,
        teacher_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CWD loss.

        Args:
            student_enc: [B, N_s, D_s] — student encoder output.
            teacher_enc: [B, N_t, D_t] — teacher encoder output (detached).

        Returns:
            Scalar CWD loss.
        """
        # Transpose to [B, D, N] for Conv1d
        s = student_enc.permute(0, 2, 1)  # [B, D_s, N_s]
        t = teacher_enc.permute(0, 2, 1)  # [B, D_t, N_t]

        s = self.align(s)  # [B, D_t, N_s]

        # Align sequence lengths via interpolation
        if s.size(-1) != t.size(-1):
            s = F.interpolate(s, size=t.size(-1), mode="linear", align_corners=False)

        # Channel-wise spatial softmax: normalize each channel over N positions
        # s/t: [B, D_t, N] → probability distributions over N for each channel
        s_norm = F.log_softmax(s / self.tau, dim=-1)   # log-probs for KLDiv input
        t_norm = F.softmax(t.detach() / self.tau, dim=-1)   # probs for KLDiv target

        # KL divergence summed over channels, averaged over batch
        # Reshape to [B*D_t, N] for batchmean reduction
        B, D, N = s_norm.shape
        loss = F.kl_div(
            s_norm.reshape(B * D, N),
            t_norm.reshape(B * D, N),
            reduction="batchmean",
        )
        return loss

    def extra_repr(self) -> str:
        return (
            f"student_channels={self.student_channels}, "
            f"teacher_channels={self.teacher_channels}, tau={self.tau}"
        )
