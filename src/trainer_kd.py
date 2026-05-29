"""KD-aware training loop for RT-DETR Knowledge Distillation.

Features:
  - Mixed-precision training (torch.cuda.amp) for RTX 3050 VRAM budget.
  - Gradient accumulation (effective batch size = batch_size * accumulate_steps).
  - Cosine LR schedule with linear warmup.
  - Differential LR: lower rate for backbone, higher for transformer head.
  - TensorBoard logging of all individual loss components.
  - COCO mAP evaluation via pycocotools.
  - Checkpoint saving every N epochs; best checkpoint tracked by mAP.
"""

import os
import json
import time
import math
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCO = None
    COCOeval = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decoding (DETR-style top-k over Q*C)
# ---------------------------------------------------------------------------

def _topk_decode(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    top_k: int = 100,
):
    """DETR-style top-k decoding.

    Argmax-per-query under-counts multi-label predictions because a single
    object query may legitimately be the top scorer for multiple classes.
    Top-k over the flattened (Q*C) score tensor recovers those predictions.

    Args:
        pred_logits: [B, Q, C] raw class logits.
        pred_boxes:  [B, Q, 4] normalized cxcywh boxes.
        top_k:       Number of predictions to keep per image (DETR uses 100).

    Returns:
        Tuple of (scores [B, K], labels [B, K], boxes [B, K, 4]) where
        K = min(top_k, Q*C).
    """
    B, Q, C = pred_logits.shape
    prob = pred_logits.sigmoid()                                  # [B, Q, C]
    k = min(top_k, Q * C)
    topk_scores, topk_idx = prob.flatten(1).topk(k, dim=1)        # [B, K]
    labels = topk_idx % C                                          # [B, K]
    query_idx = topk_idx // C                                      # [B, K]
    boxes = torch.gather(
        pred_boxes, 1, query_idx.unsqueeze(-1).expand(-1, -1, 4)
    )                                                              # [B, K, 4]
    return topk_scores, labels, boxes


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    total_iters: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_iters: Number of iterations for linear warmup.
        total_iters: Total training iterations.

    Returns:
        LambdaLR scheduler (step per iteration, not per epoch).
    """
    def lr_lambda(current_iter: int) -> float:
        if current_iter < warmup_iters:
            return float(current_iter) / max(1, warmup_iters)
        progress = float(current_iter - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # last_epoch=-1 + verbose=False avoids the "step before optimizer.step" warning
    # that PyTorch emits when LambdaLR calls lr_lambda(0) at construction time.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.AdamW:
    """Build AdamW optimizer with differential learning rates.

    Backbone parameters use a lower LR than the transformer head.

    Args:
        model: The model (RTDETRWithKD or plain RTDETR).
        cfg: Training config dict with keys 'lr_backbone', 'lr_head',
             'weight_decay'.

    Returns:
        Configured AdamW optimizer.
    """
    lr_backbone = cfg.get("lr_backbone", 1e-4)
    lr_head = cfg.get("lr_head", 1e-3)
    weight_decay = cfg.get("weight_decay", 1e-4)

    # Separate backbone from head parameters
    backbone_params = []
    head_params = []

    # For RTDETRWithKD we only optimize student parameters
    base_model = getattr(model, "student", model)

    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Also add KD projection layer parameters (from KDLoss)
    # They are part of the loss module, passed separately if needed
    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]
    # Filter empty groups
    param_groups = [g for g in param_groups if g["params"]]

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class KDTrainer:
    """Knowledge Distillation trainer for RT-DETR.

    Args:
        model:         RTDETRWithKD model (or plain RTDETR for baseline).
        loss_fn:       KDLoss instance.
        optimizer:     Optimizer (built externally or via build_optimizer).
        scheduler:     LR scheduler (step per iteration).
        train_loader:  DataLoader for training set.
        val_loader:    DataLoader for validation set.
        cfg:           Full config dict including 'train' and 'checkpoint' sub-dicts.
        device:        torch.device for training.
        scaler:        GradScaler for AMP (created automatically if None and use_amp=True).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
        scaler: Optional[GradScaler] = None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        train_cfg = cfg.get("train", {})
        self.use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
        self.accumulate_steps = train_cfg.get("accumulate_steps", 2)
        self.clip_max_norm = train_cfg.get("clip_max_norm", 0.1)

        if self.use_amp:
            self.scaler = scaler or GradScaler('cuda')
        else:
            self.scaler = None

        # Output / logging
        checkpoint_cfg = cfg.get("checkpoint", {})
        self.output_dir = Path(checkpoint_cfg.get("output_dir", "runs/kd"))
        self.save_every = checkpoint_cfg.get("save_every", 5)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))
        self.global_step = 0
        self.best_map = 0.0

        # COCO val annotation file (needed for pycocotools evaluation)
        data_cfg = cfg.get("data", {})
        self.val_ann_file = data_cfg.get("val_ann", None)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(self, epochs: int) -> None:
        """Full training loop.

        Args:
            epochs: Total number of training epochs.
        """
        logger.info(f"Starting training for {epochs} epochs. Output dir: {self.output_dir}")
        logger.info(f"AMP: {self.use_amp}, accumulate_steps: {self.accumulate_steps}")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch}/{epochs} [{epoch_time:.1f}s] "
                f"loss_total={train_metrics['loss_total']:.4f} "
                f"loss_det={train_metrics['loss_det']:.4f} "
                f"loss_kd={train_metrics.get('loss_kd', 0.0):.4f}"
            )

            # TensorBoard epoch-level scalars
            for key, val in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", val, epoch)

            # Periodic checkpoint
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch, tag=f"epoch_{epoch:04d}")

            # Evaluation
            map_score = self.evaluate(epoch)
            self.writer.add_scalar("val/mAP", map_score, epoch)
            logger.info(f"  mAP@[.5:.95] = {map_score:.4f}")

            if map_score > self.best_map:
                self.best_map = map_score
                self.save_checkpoint(epoch, tag="best")
                logger.info(f"  New best mAP: {self.best_map:.4f}")

        logger.info(f"Training complete. Best mAP: {self.best_map:.4f}")
        self.writer.close()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch.

        Returns:
            Dict of average loss values over the epoch.
        """
        self.model.train()
        self.loss_fn.train()

        running_losses: dict[str, float] = {}
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            # Mixed-precision forward pass
            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                losses = self.loss_fn(outputs, targets, epoch=epoch)
                loss = losses["loss_total"] / self.accumulate_steps

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation: only step every accumulate_steps iterations
            if (batch_idx + 1) % self.accumulate_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_max_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_max_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            # Accumulate scalar losses for logging
            for key, val in losses.items():
                running_losses[key] = running_losses.get(key, 0.0) + val.item()

            # Per-step TensorBoard + console logging
            self.global_step += 1
            if self.global_step % 50 == 0:
                for key, val in losses.items():
                    self.writer.add_scalar(f"step/{key}", val.item(), self.global_step)
                current_lr = self.optimizer.param_groups[-1]["lr"]
                self.writer.add_scalar("step/lr", current_lr, self.global_step)

            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[-1]["lr"]
                logger.info(
                    f"  Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"loss={losses['loss_total'].item():.4f}  "
                    f"det={losses['loss_det'].item():.4f}  "
                    f"kd={losses.get('loss_kd', torch.tensor(0.0)).item():.4f}  "
                    f"lr={current_lr:.2e}"
                )

            num_batches += 1

        # Average over batches
        avg_losses = {k: v / num_batches for k, v in running_losses.items()}
        return avg_losses

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, epoch: int = 0) -> float:
        """Run COCO evaluation on the validation set.

        Returns:
            mAP@[0.5:0.95] (primary COCO metric). Returns 0.0 if pycocotools
            is not available or val_ann_file is not set.
        """
        if COCO is None or self.val_ann_file is None:
            logger.warning("pycocotools or val_ann_file not available; skipping eval.")
            return 0.0

        if not os.path.exists(self.val_ann_file):
            logger.warning(f"Annotation file not found: {self.val_ann_file}")
            return 0.0

        self.model.eval()
        # For KD model, evaluate using student predictions
        eval_model = getattr(self.model, "student", self.model)
        eval_model.eval()

        coco_gt = COCO(self.val_ann_file)

        # Build reverse mapping: contiguous 0..79 -> COCO category IDs
        from .data.coco_dataset import _COCO_CATEGORIES_80
        idx_to_coco_id = {i: cat_id for i, cat_id in enumerate(_COCO_CATEGORIES_80)}

        results = []

        for images, targets in self.val_loader:
            images = images.to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                outputs = eval_model(images)

            pred_logits = outputs["pred_logits"]  # [B, Q, C]
            pred_boxes = outputs["pred_boxes"]    # [B, Q, 4]

            # Top-k decoding over (Q*C) — standard DETR / RT-DETR eval protocol.
            # A single query may surface under multiple classes, which recovers
            # the multi-label mAP that argmax decoding under-counts.
            scores, labels, decoded_boxes = _topk_decode(pred_logits, pred_boxes,
                                                         top_k=100)

            for i, (img_scores, img_labels, img_boxes, target) in enumerate(
                zip(scores, labels, decoded_boxes, targets)
            ):
                img_id = target["image_id"]
                if isinstance(img_id, torch.Tensor):
                    img_id = img_id.item()
                orig_h, orig_w = target["orig_size"]
                if isinstance(orig_h, torch.Tensor):
                    orig_h, orig_w = orig_h.item(), orig_w.item()

                # Convert normalized cxcywh -> pixel xywh
                cx = img_boxes[:, 0] * orig_w
                cy = img_boxes[:, 1] * orig_h
                bw = img_boxes[:, 2] * orig_w
                bh = img_boxes[:, 3] * orig_h
                x0 = cx - bw / 2
                y0 = cy - bh / 2

                for j in range(img_scores.size(0)):
                    score = img_scores[j].item()
                    if score < 0.01:
                        continue
                    cat_id = idx_to_coco_id.get(img_labels[j].item(), -1)
                    if cat_id < 0:
                        continue
                    results.append({
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [
                            round(x0[j].item(), 2),
                            round(y0[j].item(), 2),
                            round(bw[j].item(), 2),
                            round(bh[j].item(), 2),
                        ],
                        "score": round(score, 4),
                    })

        if not results:
            logger.warning("No predictions above threshold — mAP = 0.0")
            return 0.0

        # Save results to temp file for pycocotools
        results_file = self.output_dir / f"results_epoch{epoch:04d}.json"
        with open(results_file, "w") as f:
            json.dump(results, f)

        coco_dt = coco_gt.loadRes(str(results_file))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        map_score = float(coco_eval.stats[0])  # mAP@[0.5:0.95]

        # Clean up temp file
        results_file.unlink(missing_ok=True)

        return map_score

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, tag: str = "") -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number (stored in checkpoint).
            tag:   Suffix for the checkpoint filename.
        """
        # Save student (or full model for baseline)
        model_to_save = getattr(self.model, "student", self.model)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_map": self.best_map,
            "cfg": self.cfg,
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        filename = f"checkpoint_{tag}.pth" if tag else f"checkpoint_epoch{epoch:04d}.pth"
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> int:
        """Load a checkpoint and restore model (and optionally optimizer) state.

        Args:
            path: Path to .pth checkpoint file.
            load_optimizer: Whether to restore optimizer state.

        Returns:
            Epoch number from the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        model_to_load = getattr(self.model, "student", self.model)
        model_to_load.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.best_map = checkpoint.get("best_map", 0.0)
        epoch = checkpoint.get("epoch", 0)
        logger.info(f"Loaded checkpoint from epoch {epoch}: {path}")
        return epoch
