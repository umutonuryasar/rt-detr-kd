"""RT-DETR detection loss.

Combines:
  - Focal loss for multi-class classification (varifocal-style).
  - L1 loss on (cx, cy, w, h) bounding-box coordinates.
  - Generalized IoU loss for bounding-box regression.

Uses Hungarian matching (HungarianMatcher) to assign predictions to GT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, box_convert

from .matcher import HungarianMatcher


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """Sigmoid focal loss (Lin et al., 2017).

    Args:
        inputs:  [N, C] raw logits.
        targets: [N, C] binary targets in {0, 1}.
        alpha:   Weighting factor for the rare class.
        gamma:   Focusing parameter.
        reduction: 'none', 'mean', or 'sum'.

    Returns:
        Scalar loss (if reduction != 'none').
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * ((1 - p_t) ** gamma)
    loss = focal_weight * ce_loss

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    else:  # sum
        return loss.sum()


class RTDETRLoss(nn.Module):
    """Compute the RT-DETR detection loss for a batch.

    Args:
        num_classes: Number of object categories.
        matcher: HungarianMatcher instance (or None to use defaults).
        cost_class: Matcher classification cost weight.
        cost_bbox: Matcher L1 bbox cost weight.
        cost_giou: Matcher GIoU cost weight.
        focal_alpha: Focal loss alpha.
        focal_gamma: Focal loss gamma.
        weight_class: Loss weight for classification term.
        weight_bbox: Loss weight for L1 bbox term.
        weight_giou: Loss weight for GIoU term.
    """

    def __init__(
        self,
        num_classes: int = 80,
        matcher: HungarianMatcher | None = None,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou

        self.matcher = matcher or HungarianMatcher(cost_class, cost_bbox, cost_giou)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Compute detection loss.

        Args:
            outputs: Dict with 'pred_logits' [B, Q, C] and 'pred_boxes' [B, Q, 4].
            targets: List of B dicts, each containing:
                'labels': [M] integer class ids.
                'boxes':  [M, 4] normalized (cx, cy, w, h) in [0, 1].

        Returns:
            Dict with scalar losses:
              'loss_ce':   Focal classification loss.
              'loss_bbox': L1 bounding-box regression loss.
              'loss_giou': GIoU bounding-box regression loss.
              'loss_det':  Weighted sum of all three.
        """
        device = outputs["pred_logits"].device
        B, Q, C = outputs["pred_logits"].shape

        # Compute matching
        indices = self.matcher(outputs, targets)

        # Total number of matched objects across the batch (for normalization)
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)

        # ---- Build matched predictions and ground-truth tensors ----
        pred_logits = outputs["pred_logits"]  # [B, Q, C]
        pred_boxes = outputs["pred_boxes"]    # [B, Q, 4]

        # ----- Classification loss (focal) -----
        # Target: one-hot over ALL queries; only matched ones get a positive label
        target_classes = torch.zeros(B, Q, C, device=device)
        for i, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            labels = targets[i]["labels"][gt_idx].to(device)  # [M_i]
            target_classes[i, pred_idx, labels] = 1.0

        loss_ce = sigmoid_focal_loss(
            pred_logits.reshape(-1, C),
            target_classes.reshape(-1, C),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="sum",
        ) / num_boxes

        # ----- Bounding-box losses -----
        # Collect only matched prediction/gt pairs
        matched_pred_boxes = []
        matched_gt_boxes = []
        for i, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            matched_pred_boxes.append(pred_boxes[i][pred_idx])
            matched_gt_boxes.append(targets[i]["boxes"][gt_idx].to(device))

        if matched_pred_boxes:
            matched_pred = torch.cat(matched_pred_boxes, dim=0)  # [M_total, 4]
            matched_gt = torch.cat(matched_gt_boxes, dim=0)      # [M_total, 4]

            # L1 loss
            loss_bbox = F.l1_loss(matched_pred, matched_gt, reduction="sum") / num_boxes

            # GIoU loss
            pred_xyxy = box_convert(matched_pred, in_fmt="cxcywh", out_fmt="xyxy").clamp(0, 1)
            gt_xyxy = box_convert(matched_gt, in_fmt="cxcywh", out_fmt="xyxy").clamp(0, 1)
            giou = generalized_box_iou(pred_xyxy, gt_xyxy)
            # generalized_box_iou returns [M, M]; we want the diagonal
            loss_giou = (1 - giou.diag()).sum() / num_boxes
        else:
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        loss_det = (
            self.weight_class * loss_ce
            + self.weight_bbox * loss_bbox
            + self.weight_giou * loss_giou
        )

        return {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_det": loss_det,
        }
