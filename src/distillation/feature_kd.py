"""Feature-level Knowledge Distillation loss.

Two complementary components:

1. **Encoder feature distillation** (L_feat):
   MSE loss between the projected student encoder output and the teacher
   encoder output.  A 1x1 Conv1d projection aligns channel dimensions when
   student_dim != teacher_dim, initialized with Xavier uniform.

   L_feat = MSE( proj(student_enc), teacher_enc.detach() )

2. **Decoder cross-attention distillation** (L_attn):
   1 - cosine similarity between student and teacher cross-attention maps,
   averaged over decoder layers and attention heads.

   L_attn = mean( 1 - cos_sim(student_attn, teacher_attn.detach()) )

Combined:
   L_KD = feat_weight * L_feat + alpha * L_attn   (alpha=0.5 by default)

Multi-scale alignment (IMPORTANT)
---------------------------------
Encoder token sequences are *concatenations of flattened scales* (e.g.,
student C4+C5, teacher C3+C4+C5). Naive 1-D interpolation over the
concatenated axis blends tokens across scale boundaries and destroys the 2-D
spatial structure. The correct alignment — used whenever per-scale shapes
are provided via ``student_shapes`` / ``teacher_shapes`` — is:

  1. Split each sequence back into its per-scale chunks.
  2. Pair scales from the coarsest end (C5<->C5, C4<->C4, ...); the
     teacher's extra fine scales (e.g., C3) are dropped, not distilled.
  3. If a paired scale's spatial dims differ, interpolate *in 2-D* within
     that scale only (bilinear).
  4. Concatenate the aligned pairs and compute one MSE.

When shapes are unavailable the legacy 1-D interpolation is used as a
fallback with a one-time warning, so older checkpoints/tests keep working.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)


def split_by_shapes(
    tokens_bdn: torch.Tensor, shapes: list[tuple[int, int]]
) -> list[torch.Tensor]:
    """Split a [B, D, N_total] sequence into per-scale [B, D, H, W] maps."""
    B, D, N = tokens_bdn.shape
    expected = sum(h * w for h, w in shapes)
    if expected != N:
        raise ValueError(
            f"scale shapes {shapes} sum to {expected} tokens but sequence has {N}"
        )
    maps = []
    offset = 0
    for h, w in shapes:
        chunk = tokens_bdn[:, :, offset:offset + h * w]
        maps.append(chunk.reshape(B, D, h, w))
        offset += h * w
    return maps


def align_scales(
    s_maps: list[torch.Tensor],
    t_maps: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair scales from the coarsest end and 2-D-align each pair.

    Returns flattened, concatenated (student, teacher) token tensors
    [B, D, N_matched] covering only the matched scales.
    """
    n_pairs = min(len(s_maps), len(t_maps))
    s_parts, t_parts = [], []
    # Coarsest scales are at the END of the list (C3, C4, C5 ordering).
    for k in range(1, n_pairs + 1):
        s_map = s_maps[-k]
        t_map = t_maps[-k]
        if s_map.shape[-2:] != t_map.shape[-2:]:
            s_map = F.interpolate(
                s_map, size=t_map.shape[-2:], mode="bilinear", align_corners=False
            )
        s_parts.append(s_map.flatten(2))   # [B, D, H*W]
        t_parts.append(t_map.flatten(2))
    return torch.cat(s_parts, dim=-1), torch.cat(t_parts, dim=-1)


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
        self._warned_legacy_interp = False

        # Projection to align channel dimensions.
        # Conv1d operates on [B, C, N] which fits token sequences naturally.
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
        student_shapes: Optional[list[tuple[int, int]]] = None,
        teacher_shapes: Optional[list[tuple[int, int]]] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute feature KD loss.

        Args:
            student_enc:    [B, N_s, D_s] — student encoder output.
            teacher_enc:    [B, N_t, D_t] — teacher encoder output (detached).
            student_attn:   [L, B, H, Q_s, N_s] or None.
            teacher_attn:   [L, B, H, Q_t, N_t] or None.
            student_shapes: Per-scale (H, W) list for the student sequence.
            teacher_shapes: Per-scale (H, W) list for the teacher sequence.
                            When both are given, alignment is per-scale in 2-D.

        Returns:
            Dict with scalar losses:
              'loss_feat': encoder feature MSE loss.
              'loss_attn': attention cosine similarity loss (0 if no attn maps).
              'loss_kd':   combined feat_weight * L_feat + alpha * L_attn.
        """
        # ---- Encoder feature distillation ----
        s_enc = student_enc.permute(0, 2, 1)   # [B, D_s, N_s]
        t_enc = teacher_enc.permute(0, 2, 1).detach()   # [B, D_t, N_t]

        s_proj = self.proj(s_enc)  # [B, D_t, N_s]

        if student_shapes is not None and teacher_shapes is not None:
            # Correct path: per-scale 2-D alignment.
            s_maps = split_by_shapes(s_proj, list(student_shapes))
            t_maps = split_by_shapes(t_enc, list(teacher_shapes))
            s_aligned, t_aligned = align_scales(s_maps, t_maps)
            loss_feat = F.mse_loss(s_aligned, t_aligned)
        else:
            # Legacy fallback: 1-D interpolation over the concatenated axis.
            # Mixes tokens across scale boundaries — only acceptable when
            # scale shapes are genuinely unavailable.
            if s_proj.size(-1) != t_enc.size(-1):
                if not self._warned_legacy_interp:
                    logger.warning(
                        "FeatureKDLoss: per-scale shapes not provided; falling "
                        "back to 1-D interpolation over the concatenated "
                        "multi-scale token axis. This blends tokens across "
                        "scales — pass student_shapes/teacher_shapes for "
                        "correct per-scale alignment."
                    )
                    self._warned_legacy_interp = True
                s_proj = F.interpolate(
                    s_proj, size=t_enc.size(-1), mode="linear", align_corners=False
                )
            loss_feat = F.mse_loss(s_proj, t_enc)

        # ---- Attention distillation ----
        loss_attn = torch.tensor(0.0, device=student_enc.device)

        if student_attn is not None and teacher_attn is not None:
            L = min(student_attn.size(0), teacher_attn.size(0))
            s_attn = student_attn[:L]  # [L, B, H_s, Q_s, N_s]
            t_attn = teacher_attn[:L]  # [L, B, H_t, Q_t, N_t]

            # Average over heads -> [L, B, Q, N]
            s_attn = s_attn.mean(dim=2)
            t_attn = t_attn.mean(dim=2)

            # Align query count
            Q = min(s_attn.size(2), t_attn.size(2))
            s_attn = s_attn[:, :, :Q, :]
            t_attn = t_attn[:, :, :Q, :]

            # Align spatial token count via average pooling
            if s_attn.size(-1) != t_attn.size(-1):
                target_N = min(s_attn.size(-1), t_attn.size(-1))
                L_, B_, Q_ = s_attn.shape[:3]
                s_flat = s_attn.reshape(L_ * B_ * Q_, 1, -1)
                t_flat = t_attn.reshape(L_ * B_ * Q_, 1, -1)
                s_flat = F.adaptive_avg_pool1d(s_flat, target_N)
                t_flat = F.adaptive_avg_pool1d(t_flat, target_N)
                s_attn = s_flat.reshape(L_, B_, Q_, target_N)
                t_attn = t_flat.reshape(L_, B_, Q_, target_N)

            L_, B_, Q_, N_ = s_attn.shape
            s_flat = s_attn.reshape(-1, N_)
            t_flat = t_attn.reshape(-1, N_)

            cos_sim = F.cosine_similarity(s_flat, t_flat.detach(), dim=-1)
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
