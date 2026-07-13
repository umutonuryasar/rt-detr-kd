"""Query-level Knowledge Distillation loss.

Novel RT-DETR-specific distillation that targets the decoder object queries
directly — a transformer-specific component not addressed by prior KD methods
designed for CNN detectors.

Query correspondence
--------------------
Decoder queries carry NO canonical ordering: which query ends up representing
which object is image- and model-dependent. Index-wise truncation to the first
min(Q_s, Q_t) queries therefore matches semantically unrelated pairs. Two
matching strategies are provided:

  matching="hungarian" (default, recommended):
      Per-image bipartite matching between student and teacher queries.
      The matching cost is computed in *prediction space* when class logits
      and boxes are provided (what object does this query describe?):

        cost = ||sigmoid(s_cls) - sigmoid(t_cls)||_1 + w_box * ||s_box - t_box||_1

      falling back to negative cosine similarity of the (projected) query
      embeddings when predictions are unavailable. Matching runs under
      no_grad (scipy linear_sum_assignment); gradients flow through the
      MSE on the matched pairs only.

  matching="index" (legacy / ablation):
      First-min(Q_s, Q_t) index-wise alignment. Kept strictly as an
      ablation baseline to quantify the value of proper matching.

Loss components:
1. Query embedding MSE over matched pairs:     L_query = MSE(q_s[i], q_t[j])
2. Decoder cross-attention alignment (optional), over the same matched pairs:
       L_query_attn = mean( 1 - cos_sim(A_s[i], A_t[j]) )
Combined:  L_KD = L_query + alpha * L_query_attn

Graceful degradation: if teacher queries are unavailable (e.g., the canonical
lyuwenyu deformable-attention teacher does not expose post-norm query
embeddings), the loss returns zeros and logs a one-time warning instead of
crashing — but note that Query-KD against such a teacher is a no-op and the
run configuration should be reconsidered.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_QUERY_MATCHING = ("hungarian", "index")


class QueryKDLoss(nn.Module):
    """Object query distillation for RT-DETR.

    Args:
        student_dim: Hidden dimension of student decoder (for projection).
        teacher_dim: Hidden dimension of teacher decoder.
        alpha:       Weight for the decoder cross-attention alignment term.
        matching:    "hungarian" (bipartite matching, default) or "index"
                     (legacy first-K truncation, kept for ablation).
        box_cost_weight: Weight of the box L1 term in the prediction-space
                     matching cost (only used with matching="hungarian").
    """

    def __init__(
        self,
        student_dim: int = 256,
        teacher_dim: int = 256,
        alpha: float = 0.5,
        matching: str = "hungarian",
        box_cost_weight: float = 5.0,
    ):
        super().__init__()
        if matching not in SUPPORTED_QUERY_MATCHING:
            raise ValueError(
                f"matching must be one of {SUPPORTED_QUERY_MATCHING}, got '{matching}'"
            )
        self.alpha = alpha
        self.matching = matching
        self.box_cost_weight = box_cost_weight
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self._warned_missing_teacher = False

        # Project student queries to teacher dimension if needed
        if student_dim != teacher_dim:
            self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
            nn.init.xavier_uniform_(self.proj.weight)
        else:
            self.proj = nn.Identity()

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _match(
        self,
        s_q: torch.Tensor,                       # [B, Q_s, D_t] (projected)
        t_q: torch.Tensor,                       # [B, Q_t, D_t]
        student_preds: Optional[dict] = None,    # {'pred_logits','pred_boxes'}
        teacher_preds: Optional[dict] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-image bipartite matching. Returns list of (s_idx, t_idx)."""
        from scipy.optimize import linear_sum_assignment

        B = s_q.size(0)
        use_preds = (
            student_preds is not None
            and teacher_preds is not None
            and "pred_logits" in student_preds
            and "pred_logits" in teacher_preds
        )

        indices = []
        for b in range(B):
            if use_preds:
                # Prediction-space cost: match queries that describe the
                # same object (class distribution + box location).
                s_p = student_preds["pred_logits"][b].sigmoid()   # [Q_s, C]
                t_p = teacher_preds["pred_logits"][b].sigmoid()   # [Q_t, C]
                cost_cls = torch.cdist(s_p, t_p, p=1)             # [Q_s, Q_t]
                s_b = student_preds["pred_boxes"][b]              # [Q_s, 4]
                t_b = teacher_preds["pred_boxes"][b]              # [Q_t, 4]
                cost_box = torch.cdist(s_b, t_b, p=1)             # [Q_s, Q_t]
                cost = cost_cls + self.box_cost_weight * cost_box
            else:
                # Embedding-space fallback: negative cosine similarity.
                s_n = F.normalize(s_q[b], dim=-1)                 # [Q_s, D]
                t_n = F.normalize(t_q[b], dim=-1)                 # [Q_t, D]
                cost = -(s_n @ t_n.T)                             # [Q_s, Q_t]

            row, col = linear_sum_assignment(cost.float().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(row, dtype=torch.long, device=s_q.device),
                    torch.as_tensor(col, dtype=torch.long, device=s_q.device),
                )
            )
        return indices

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        student_queries: Optional[torch.Tensor],
        teacher_queries: Optional[torch.Tensor],
        student_attn: Optional[torch.Tensor] = None,
        teacher_attn: Optional[torch.Tensor] = None,
        student_preds: Optional[dict] = None,
        teacher_preds: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute query KD loss.

        Args:
            student_queries: [B, Q_s, D_s] — student decoder query embeddings.
            teacher_queries: [B, Q_t, D_t] — teacher decoder query embeddings
                             (detached), or None if the teacher does not
                             expose them (loss degrades to zero + warning).
            student_attn:    [L, B, H, Q_s, N_s] or None — decoder cross-attn maps.
            teacher_attn:    [L, B, H, Q_t, N_t] or None.
            student_preds:   Optional {'pred_logits','pred_boxes'} for
                             prediction-space Hungarian matching.
            teacher_preds:   Optional dito (detached).

        Returns:
            Dict with scalar losses:
              'loss_query':      query embedding MSE.
              'loss_query_attn': decoder attention cosine loss (0 if no attn).
              'loss_kd':         combined loss.
        """
        # ---- Graceful degradation when teacher queries are unavailable ----
        if student_queries is None or teacher_queries is None:
            if not self._warned_missing_teacher:
                logger.warning(
                    "QueryKDLoss: teacher (or student) decoder queries are None — "
                    "Query-KD contributes ZERO loss this run. The canonical "
                    "lyuwenyu deformable teacher does not expose post-norm query "
                    "embeddings; use the own-architecture teacher for Query-KD "
                    "or add a decoder hook to the teacher adapter."
                )
                self._warned_missing_teacher = True
            device = (
                student_queries.device if student_queries is not None
                else (teacher_queries.device if teacher_queries is not None
                      else torch.device("cpu"))
            )
            zero = torch.tensor(0.0, device=device)
            return {"loss_query": zero, "loss_query_attn": zero.clone(),
                    "loss_kd": zero.clone()}

        s_q_full = self.proj(student_queries)          # [B, Q_s, D_t]
        t_q_full = teacher_queries.detach()            # [B, Q_t, D_t]

        # ---- Build correspondence ----
        if self.matching == "hungarian":
            match_indices = self._match(
                s_q_full, t_q_full, student_preds, teacher_preds
            )
            s_list, t_list = [], []
            for b, (s_idx, t_idx) in enumerate(match_indices):
                s_list.append(s_q_full[b, s_idx])      # [K_b, D_t]
                t_list.append(t_q_full[b, t_idx])      # [K_b, D_t]
            s_m = torch.cat(s_list, dim=0)
            t_m = torch.cat(t_list, dim=0)
        else:  # "index" — legacy truncation, ablation baseline
            Q = min(s_q_full.size(1), t_q_full.size(1))
            match_indices = None
            s_m = s_q_full[:, :Q, :].reshape(-1, s_q_full.size(-1))
            t_m = t_q_full[:, :Q, :].reshape(-1, t_q_full.size(-1))

        loss_query = F.mse_loss(s_m, t_m)

        # ---- Decoder cross-attention alignment over the same pairs ----
        loss_query_attn = torch.tensor(0.0, device=student_queries.device)

        if student_attn is not None and teacher_attn is not None:
            L = min(student_attn.size(0), teacher_attn.size(0))
            s_attn = student_attn[:L].mean(dim=2)  # [L, B, Q_s, N_s] (avg heads)
            t_attn = teacher_attn[:L].mean(dim=2)  # [L, B, Q_t, N_t]

            # Reorder queries according to the matching (or truncate for index)
            if self.matching == "hungarian" and match_indices is not None:
                s_rows, t_rows = [], []
                for b, (s_idx, t_idx) in enumerate(match_indices):
                    s_rows.append(s_attn[:, b, s_idx, :])   # [L, K_b, N_s]
                    t_rows.append(t_attn[:, b, t_idx, :])   # [L, K_b, N_t]
                s_attn = torch.stack(s_rows, dim=1)          # [L, B, K, N_s]
                t_attn = torch.stack(t_rows, dim=1)          # [L, B, K, N_t]
            else:
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
            f"student_dim={self.student_dim}, teacher_dim={self.teacher_dim}, "
            f"alpha={self.alpha}, matching={self.matching}"
        )
