"""Feature-level Knowledge Distillation loss.

Two complementary components:

1. **Encoder feature distillation** (L_feat):
   MSE loss between the projected student encoder output and the teacher
   encoder output.  A 1×1 Conv1d projection aligns channel dimensions when
   student_dim != teacher_dim, initialized with Xavier uniform.

   L_feat = MSE( proj(student_enc), teacher_enc.detach() )

2. **Decoder cross-attention distillation** (L_attn):
   1 - cosine similarity between student and teacher cross-attention maps,
   averaged over decoder layers and attention heads.

   L_attn = mean( 1 - cos_sim(student_attn, teacher_attn.detach()) )

Combined:
   L_KD = L_feat + alpha * L_attn   (alpha=0.5 by default)

Spatial mismatch handling:
   The encoder sequences for student and teacher may differ in length
   (N_s != N_t) because they use different backbone strides/channels.
   We interpolate the smaller sequence to match the larger before MSE.
   For attention maps we average over the spatial (N) dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeatureKDLoss(nn.Module):
    """MSE encoder feature distillation + cosine attention distillation.

    Args:
        student_dim: Channel dimension of student encoder output.
        teacher_dim: Channel dimension of teacher encoder output.
        alpha:       Weight for the attention distillation term (L_attn).
        feat_weight: Weight for the encoder MSE term (L_feat). Set to 0.0
                     for attention-only distillation.
    """

    def __init__(
        self,
        student_dim: int = 256,
        teacher_dim: int = 256,
        alpha: float = 0.5,
        feat_weight: float = 1.0,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.alpha = alpha
        self.feat_weight = feat_weight

        # Projection to align channel dimensions.
        # Conv1d operates on [B, C, N] which fits token sequences naturally.
        # When dims already match we use Identity to avoid a learnable mapping
        # that the optimizer would otherwise have to discover is unnecessary.
        if student_dim != teacher_dim:
            self.proj = nn.Conv1d(student_dim, teacher_dim, kernel_size=1, bias=False)
            self._init_weights()
        else:
            self.proj = nn.Identity()

    def _init_weights(self) -> None:
        if isinstance(self.proj, nn.Conv1d):
            nn.init.xavier_uniform_(self.proj.weight)

    def forward(
        self,
        student_enc: torch.Tensor,
        teacher_enc: torch.Tensor,
        student_attn: Optional[torch.Tensor] = None,
        teacher_attn: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute feature KD loss.

        Args:
            student_enc:  [B, N_s, D_s] — student encoder output.
            teacher_enc:  [B, N_t, D_t] — teacher encoder output (detached).
            student_attn: [L, B, H, Q_s, N_s] or None.
            teacher_attn: [L, B, H, Q_t, N_t] or None.

        Returns:
            Dict with scalar losses:
              'loss_feat': encoder feature MSE loss.
              'loss_attn': attention cosine similarity loss (0 if no attn maps).
              'loss_kd':   combined L_feat + alpha * L_attn.
        """
        # ---- Encoder feature distillation ----
        # Transpose to [B, D, N] for Conv1d projection
        s_enc = student_enc.permute(0, 2, 1)   # [B, D_s, N_s]
        t_enc = teacher_enc.permute(0, 2, 1)   # [B, D_t, N_t]

        s_proj = self.proj(s_enc)  # [B, D_t, N_s]

        # Align sequence lengths via linear interpolation if needed
        if s_proj.size(-1) != t_enc.size(-1):
            s_proj = F.interpolate(s_proj, size=t_enc.size(-1), mode="linear", align_corners=False)

        loss_feat = F.mse_loss(s_proj, t_enc.detach())

        # ---- Attention distillation ----
        loss_attn = torch.tensor(0.0, device=student_enc.device)

        if student_attn is not None and teacher_attn is not None:
            # student_attn: [L_s, B, H_s, Q_s, N_s]
            # teacher_attn: [L_t, B, H_t, Q_t, N_t]
            # Use min(L_s, L_t) layers
            L = min(student_attn.size(0), teacher_attn.size(0))
            s_attn = student_attn[:L]  # [L, B, H_s, Q_s, N_s]
            t_attn = teacher_attn[:L]  # [L, B, H_t, Q_t, N_t]

            # Average over heads -> [L, B, Q, N]
            s_attn = s_attn.mean(dim=2)  # [L, B, Q_s, N_s]
            t_attn = t_attn.mean(dim=2)  # [L, B, Q_t, N_t]

            # Align query count
            Q = min(s_attn.size(2), t_attn.size(2))
            s_attn = s_attn[:, :, :Q, :]  # [L, B, Q, N_s]
            t_attn = t_attn[:, :, :Q, :]  # [L, B, Q, N_t]

            # Align spatial token count via average pooling
            if s_attn.size(-1) != t_attn.size(-1):
                # Reshape to [L*B*Q, 1, N] for adaptive_avg_pool1d
                target_N = min(s_attn.size(-1), t_attn.size(-1))
                L_, B_, Q_ = s_attn.shape[:3]
                s_flat = s_attn.reshape(L_ * B_ * Q_, 1, -1)
                t_flat = t_attn.reshape(L_ * B_ * Q_, 1, -1)
                s_flat = F.adaptive_avg_pool1d(s_flat, target_N)
                t_flat = F.adaptive_avg_pool1d(t_flat, target_N)
                s_attn = s_flat.reshape(L_, B_, Q_, target_N)
                t_attn = t_flat.reshape(L_, B_, Q_, target_N)

            # Cosine similarity along the spatial (N) dimension
            # Flatten L, B, Q for batch cosine similarity
            L_, B_, Q_, N_ = s_attn.shape
            s_flat = s_attn.reshape(-1, N_)  # [L*B*Q, N]
            t_flat = t_attn.reshape(-1, N_)  # [L*B*Q, N]

            cos_sim = F.cosine_similarity(s_flat, t_flat.detach(), dim=-1)  # [L*B*Q]
            loss_attn = (1.0 - cos_sim).mean()

        loss_kd = self.feat_weight * loss_feat + self.alpha * loss_attn

        return {
            "loss_feat": loss_feat,
            "loss_attn": loss_attn,
            "loss_kd": loss_kd,
        }

    def extra_repr(self) -> str:
        return (
            f"student_dim={self.student_dim}, teacher_dim={self.teacher_dim}, "
            f"alpha={self.alpha}, feat_weight={self.feat_weight}"
        )
