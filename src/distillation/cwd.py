"""Channel-Wise Distillation loss.

Shu et al., "Channel-wise Knowledge Distillation for Dense Prediction",
ICCV 2021. (Not to be confused with FGD — Yang et al., "Focal and Global
Knowledge Distillation for Detectors", CVPR 2022, which is a different
method.)

Distills the channel-wise spatial distributions of encoder token sequences.
Each feature dimension D is treated as a "channel" with a distribution over
N spatial positions. KL divergence is applied between softmax-normalized
teacher and student channel distributions.

  L_CWD = sum_c KL( softmax_spatial(t_c / tau) || softmax_spatial(s_c / tau) )

A 1x1 Conv1d projection aligns student channels to teacher channels before
computing the loss.

Multi-scale alignment: when per-scale shapes are provided the sequences are
split per scale, paired from the coarsest end, 2-D-aligned within each scale,
and re-concatenated — avoiding the cross-scale blending that 1-D
interpolation over the concatenated axis would cause. Legacy 1-D
interpolation remains as a fallback when shapes are unavailable.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .feature_kd import split_by_shapes, align_scales

logger = logging.getLogger(__name__)


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
        self._warned_legacy_interp = False
        self.align = nn.Conv1d(student_channels, teacher_channels, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.align.weight)

    def forward(
        self,
        student_enc: torch.Tensor,
        teacher_enc: torch.Tensor,
        student_shapes: Optional[list[tuple[int, int]]] = None,
        teacher_shapes: Optional[list[tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """Compute CWD loss.

        Args:
            student_enc:    [B, N_s, D_s] — student encoder output.
            teacher_enc:    [B, N_t, D_t] — teacher encoder output (detached).
            student_shapes: Per-scale (H, W) list for the student sequence.
            teacher_shapes: Per-scale (H, W) list for the teacher sequence.

        Returns:
            Scalar CWD loss.
        """
        # Transpose to [B, D, N] for Conv1d
        s = student_enc.permute(0, 2, 1)            # [B, D_s, N_s]
        t = teacher_enc.permute(0, 2, 1).detach()   # [B, D_t, N_t]

        s = self.align(s)  # [B, D_t, N_s]

        if student_shapes is not None and teacher_shapes is not None:
            # Correct path: per-scale 2-D alignment (see feature_kd).
            s_maps = split_by_shapes(s, list(student_shapes))
            t_maps = split_by_shapes(t, list(teacher_shapes))
            s, t = align_scales(s_maps, t_maps)     # [B, D_t, N_matched] x2
        elif s.size(-1) != t.size(-1):
            # Legacy fallback: 1-D interpolation over the concatenated axis.
            if not self._warned_legacy_interp:
                logger.warning(
                    "CWDLoss: per-scale shapes not provided; falling back to "
                    "1-D interpolation over the concatenated multi-scale "
                    "token axis. Pass student_shapes/teacher_shapes for "
                    "correct per-scale alignment."
                )
                self._warned_legacy_interp = True
            s = F.interpolate(s, size=t.size(-1), mode="linear", align_corners=False)

        # Channel-wise spatial softmax: normalize each channel over N positions
        s_norm = F.log_softmax(s / self.tau, dim=-1)   # log-probs for KLDiv input
        t_norm = F.softmax(t / self.tau, dim=-1)       # probs for KLDiv target

        # KL divergence summed over channels and spatial positions, averaged
        # over batch.  Using reduction="batchmean" on the [B*D, N] view would
        # divide by B*D instead of B, making the loss D (=256) times too small.
        # reduction="sum" / B gives the correct per-sample channel-sum.
        B, D, N = s_norm.shape
        loss = F.kl_div(
            s_norm.reshape(B * D, N),
            t_norm.reshape(B * D, N),
            reduction="sum",
        ) / B
        return loss

    def extra_repr(self) -> str:
        return (
            f"student_channels={self.student_channels}, "
            f"teacher_channels={self.teacher_channels}, tau={self.tau}"
        )
