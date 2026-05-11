"""Hungarian bipartite matcher for RT-DETR.

Computes an optimal assignment between the N predicted queries and the M
ground-truth objects (M <= N) for each image in a batch. The matching cost
combines classification cost, L1 bounding-box cost, and GIoU cost.

Returns a list of (pred_indices, gt_indices) index pairs — one pair per image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, box_convert


class HungarianMatcher(nn.Module):
    """Compute bipartite matching between predictions and ground truth.

    Args:
        cost_class: Weight for the classification cost.
        cost_bbox:  Weight for the L1 bounding-box coordinate cost.
        cost_giou:  Weight for the Generalized IoU cost.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Find optimal assignment between predictions and targets.

        Args:
            outputs: Dict with:
                'pred_logits': [B, num_queries, num_classes]
                'pred_boxes':  [B, num_queries, 4]  (cx,cy,w,h in [0,1])
            targets: List of B dicts, each with:
                'labels': [M]    — integer class indices
                'boxes':  [M, 4] — (cx,cy,w,h) normalized to [0,1]

        Returns:
            List of (pred_idx, gt_idx) pairs, one per image in the batch.
            Each element is a tuple of 1-D LongTensors on CPU.
        """
        B, Q, _ = outputs["pred_logits"].shape
        device = outputs["pred_logits"].device

        # Flatten batch dimension for efficient cost computation
        # Probabilities via sigmoid (multi-label friendly, matches focal loss)
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [B*Q, C]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)             # [B*Q, 4]

        # Concatenate all GT targets in the batch
        tgt_ids = torch.cat([t["labels"] for t in targets])      # [sum_M]
        tgt_bbox = torch.cat([t["boxes"] for t in targets])      # [sum_M, 4]

        if tgt_ids.numel() == 0:
            # No ground-truth objects in entire batch — return empty matches
            return [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))] * B

        # ---- Classification cost: -p_hat[target_class] ----
        # out_prob: [B*Q, C]  tgt_ids: [sum_M]
        cost_class = -out_prob[:, tgt_ids]  # [B*Q, sum_M]

        # ---- L1 bounding-box cost ----
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [B*Q, sum_M]

        # ---- GIoU cost ----
        # Convert (cx,cy,w,h) -> (x1,y1,x2,y2) for GIoU computation
        out_bbox_xyxy = box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy").clamp(0, 1)
        tgt_bbox_xyxy = box_convert(tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy").clamp(0, 1)
        cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)  # [B*Q, sum_M]

        # ---- Combined cost matrix ----
        C = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )
        C = C.view(B, Q, -1).cpu()  # [B, Q, sum_M]

        # Split sum_M back into per-image counts
        sizes = [len(t["labels"]) for t in targets]

        indices = []
        for i, (c, size) in enumerate(zip(C.split(sizes, dim=-1), sizes)):
            if size == 0:
                indices.append(
                    (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
                )
            else:
                row_ind, col_ind = linear_sum_assignment(c[i].numpy())
                indices.append(
                    (
                        torch.as_tensor(row_ind, dtype=torch.long),
                        torch.as_tensor(col_ind, dtype=torch.long),
                    )
                )

        return indices
