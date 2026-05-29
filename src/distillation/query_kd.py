"""Query-level Knowledge Distillation loss.

Novel RT-DETR-specific distillation that targets the decoder object queries
directly — a transformer-specific component not addressed by prior KD methods
designed for CNN detectors.

Two components:
1. Query embedding MSE: aligns the learned object query representations
   between teacher and student decoders.

     L_query = MSE(q_s, q_t)

2. Decoder cross-attention alignment (optional): aligns the attention patterns
   each query produces when attending to encoder memory.

     L_query_attn = mean( 1 - cos_sim(A_s^dec, A_t^dec) )

Combined:
     L_KD = L_query + alpha * L_query_attn

The decoder cross-attention maps (A^dec) are the same tensors already stored
by RTDETRDecoder.attn_maps and exposed via RTDETR.get_attn_maps_tensor().
Query embeddings are the post-norm decoder output before the prediction heads,
exposed as RTDETR.decoder_queries [B, num_queries, D].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class QueryKDLoss(nn.Module):
    """Object query distillation for RT-DETR.

    Args:
        student_dim: Hidden dimension of student decoder (for projection).
        teacher_dim: Hidden dimension of teacher decoder.
        alpha:       Weight for the decoder cross-attention alignment term.
    """

    def __init__(
        self,
        student_dim: int = 256,
        teacher_dim: int = 256,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim

        # Project student queries to teacher dimension if needed
        if student_dim != teacher_dim:
            self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
            nn.init.xavier_uniform_(self.proj.weight)
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        student_queries: torch.Tensor,
        teacher_queries: torch.Tensor,
        student_attn: Optional[torch.Tensor] = None,
        teacher_attn: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute query KD loss.

        Args:
            student_queries: [B, Q_s, D_s] — student decoder query embeddings.
            teacher_queries: [B, Q_t, D_t] — teacher decoder query embeddings (detached).
            student_attn:    [L, B, H, Q_s, N_s] or None — decoder cross-attn maps.
            teacher_attn:    [L, B, H, Q_t, N_t] or None.

        Returns:
            Dict with scalar losses:
              'loss_query':      query embedding MSE.
              'loss_query_attn': decoder attention cosine loss (0 if no attn).
              'loss_kd':         combined loss.
        """
        # Align query counts
        Q = min(student_queries.size(1), teacher_queries.size(1))
        s_q = student_queries[:, :Q, :]   # [B, Q, D_s]
        t_q = teacher_queries[:, :Q, :]   # [B, Q, D_t]

        s_q = self.proj(s_q)  # [B, Q, D_t]

        loss_query = F.mse_loss(s_q, t_q.detach())

        # ---- Decoder cross-attention alignment ----
        loss_query_attn = torch.tensor(0.0, device=student_queries.device)

        if student_attn is not None and teacher_attn is not None:
            L = min(student_attn.size(0), teacher_attn.size(0))
            s_attn = student_attn[:L].mean(dim=2)  # [L, B, Q_s, N_s] (avg over heads)
            t_attn = teacher_attn[:L].mean(dim=2)  # [L, B, Q_t, N_t]

            Q_ = min(s_attn.size(2), t_attn.size(2))
            s_attn = s_attn[:, :, :Q_, :]
            t_attn = t_attn[:, :, :Q_, :]

            # Align spatial dimension
            if s_attn.size(-1) != t_attn.size(-1):
                N_target = min(s_attn.size(-1), t_attn.size(-1))
                L_, B_, Q__, _ = s_attn.shape
                s_flat = F.adaptive_avg_pool1d(s_attn.reshape(L_ * B_ * Q__, 1, -1), N_target)
                t_flat = F.adaptive_avg_pool1d(t_attn.reshape(L_ * B_ * Q__, 1, -1), N_target)
                s_attn = s_flat.reshape(L_, B_, Q__, N_target)
                t_attn = t_flat.reshape(L_, B_, Q__, N_target)

            L_, B_, Q__, N_ = s_attn.shape
            s_flat = s_attn.reshape(-1, N_)
            t_flat = t_attn.reshape(-1, N_)
            cos_sim = F.cosine_similarity(s_flat, t_flat.detach(), dim=-1)
            loss_query_attn = (1.0 - cos_sim).mean()

        loss_kd = loss_query + self.alpha * loss_query_attn

        return {
            "loss_query": loss_query,
            "loss_query_attn": loss_query_attn,
            "loss_kd": loss_kd,
        }

    def extra_repr(self) -> str:
        return (
            f"student_dim={self.student_dim}, "
            f"teacher_dim={self.teacher_dim}, alpha={self.alpha}"
        )
