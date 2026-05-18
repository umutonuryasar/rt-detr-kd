"""Masked Generative Distillation loss.

Yang et al., "Masked Generative Distillation", ECCV 2022.

Randomly masks student encoder tokens and trains a lightweight generator to
reconstruct teacher features from the masked student features. This forces the
student to learn holistic feature representations rather than per-token mimicry.

  L_MGD = || G(mask(s_feat)) - t_feat ||_2^2

The generator G is a small two-layer ConvNet. A 1x1 alignment projection
handles student/teacher channel dimension mismatch before masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MGDLoss(nn.Module):
    """Masked Generative Distillation on encoder token sequences.

    Args:
        student_channels: Feature dimension of student encoder output (D_s).
        teacher_channels: Feature dimension of teacher encoder output (D_t).
        mask_ratio:       Fraction of student tokens to mask (default 0.75).
    """

    def __init__(
        self,
        student_channels: int = 256,
        teacher_channels: int = 256,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        self.mask_ratio = mask_ratio
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels

        self.align = nn.Conv1d(student_channels, teacher_channels, kernel_size=1, bias=False)

        # Two-layer conv generator operating on the token sequence
        self.generation = nn.Sequential(
            nn.Conv1d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.align.weight)
        for m in self.generation.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        student_enc: torch.Tensor,
        teacher_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MGD loss.

        Args:
            student_enc: [B, N_s, D_s] — student encoder output.
            teacher_enc: [B, N_t, D_t] — teacher encoder output (detached).

        Returns:
            Scalar MGD loss.
        """
        # Transpose to [B, D, N] for Conv1d operations
        s = student_enc.permute(0, 2, 1)  # [B, D_s, N_s]
        t = teacher_enc.permute(0, 2, 1)  # [B, D_t, N_t]

        s = self.align(s)  # [B, D_t, N_s]

        # Align sequence lengths
        if s.size(-1) != t.size(-1):
            s = F.interpolate(s, size=t.size(-1), mode="linear", align_corners=False)

        # Random token mask: [B, 1, N] where 1 = keep, 0 = mask
        B, D, N = s.shape
        mask = (torch.rand(B, 1, N, device=s.device) > self.mask_ratio).float()
        masked = s * mask  # [B, D_t, N]

        generated = self.generation(masked)  # [B, D_t, N]

        loss = F.mse_loss(generated, t.detach())
        return loss

    def extra_repr(self) -> str:
        return (
            f"student_channels={self.student_channels}, "
            f"teacher_channels={self.teacher_channels}, mask_ratio={self.mask_ratio}"
        )
