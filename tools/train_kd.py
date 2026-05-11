#!/usr/bin/env python3
"""Main entry point for RT-DETR Knowledge Distillation training.

Usage
-----
# Feature-KD with lambda=1.0 (best config)
python tools/train_kd.py \\
    --student-cfg configs/rtdetr_r18vd_coco.yml \\
    --teacher-cfg configs/rtdetr_r50vd_coco.yml \\
    --kd-type feature \\
    --kd-lambda 1.0 \\
    --temperature 4 \\
    --epochs 36 \\
    --batch-size 4 \\
    --output-dir runs/feature_kd_l1.0 \\
    --coco-train /data/coco/train2017 \\
    --coco-val /data/coco/val2017 \\
    --train-ann /data/coco/annotations/instances_train2017.json \\
    --val-ann /data/coco/annotations/instances_val2017.json \\
    --teacher-weights /path/to/teacher.pth

# Baseline (no KD)
python tools/train_kd.py \\
    --student-cfg configs/rtdetr_r18vd_coco.yml \\
    --kd-type none \\
    --epochs 36 \\
    --output-dir runs/baseline
"""

import sys
import os
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.models.rtdetr import build_rtdetr
from src.models.rtdetr_kd import RTDETRWithKD
from src.distillation.kd_loss import KDLoss
from src.losses.detection_loss import RTDETRLoss
from src.data.coco_dataset import COCODetection, collate_fn
from src.data.transforms import build_transforms, MosaicWrapper
from src.trainer_kd import KDTrainer, build_optimizer, build_lr_scheduler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_kd")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RT-DETR Knowledge Distillation Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config files
    p.add_argument("--student-cfg", default="configs/rtdetr_r18vd_coco.yml",
                   help="Student model YAML config.")
    p.add_argument("--teacher-cfg", default="configs/rtdetr_r50vd_coco.yml",
                   help="Teacher model YAML config.")
    p.add_argument("--kd-cfg", default=None,
                   help="Optional KD-specific YAML config (overrides CLI flags).")

    # KD settings
    p.add_argument("--kd-type", default="feature", choices=["logit", "feature", "none"],
                   help="Type of knowledge distillation. 'none' = baseline training.")
    p.add_argument("--kd-lambda", type=float, default=1.0,
                   help="Weight for the KD loss term (λ).")
    p.add_argument("--temperature", type=float, default=4.0,
                   help="Temperature for logit KD (T). Ignored for feature KD.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Attention weight in feature KD. Ignored for logit KD.")

    # Training
    p.add_argument("--epochs", type=int, default=36)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulate-steps", type=int, default=2,
                   help="Gradient accumulation steps (effective BS = batch_size * steps).")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--clip-max-norm", type=float, default=0.1)
    p.add_argument("--use-amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="use_amp", action="store_false")
    p.add_argument("--mosaic", action="store_true", default=False,
                   help="Enable Mosaic augmentation (p=0.5).")

    # Data paths
    p.add_argument("--coco-train", default="/data/coco/train2017")
    p.add_argument("--coco-val", default="/data/coco/val2017")
    p.add_argument("--train-ann",
                   default="/data/coco/annotations/instances_train2017.json")
    p.add_argument("--val-ann",
                   default="/data/coco/annotations/instances_val2017.json")
    p.add_argument("--num-workers", type=int, default=4)

    # Weights
    p.add_argument("--teacher-weights", default=None,
                   help="Path to teacher pretrained weights (.pth).")
    p.add_argument("--student-weights", default=None,
                   help="Path to student pretrained/resume weights (.pth).")

    # Output
    p.add_argument("--output-dir", default="runs/kd_experiment")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def build_cfg_from_args(args: argparse.Namespace) -> dict:
    """Build a unified config dict from parsed arguments and YAML files."""
    # Load base config files
    student_cfg = load_yaml(args.student_cfg)
    teacher_cfg = load_yaml(args.teacher_cfg)

    # Override with CLI flags
    student_cfg.setdefault("train", {})
    student_cfg["train"].update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulate_steps": args.accumulate_steps,
        "img_size": args.img_size,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "weight_decay": args.weight_decay,
        "warmup_iters": args.warmup_iters,
        "clip_max_norm": args.clip_max_norm,
        "use_amp": args.use_amp,
    })
    student_cfg["data"] = {
        "train_ann": args.train_ann,
        "val_ann": args.val_ann,
        "train_img": args.coco_train,
        "val_img": args.coco_val,
        "num_workers": args.num_workers,
    }
    student_cfg["checkpoint"] = {
        "output_dir": args.output_dir,
        "save_every": args.save_every,
    }
    student_cfg["kd"] = {
        "type": args.kd_type,
        "lambda": args.kd_lambda,
        "temperature": args.temperature,
        "alpha": args.alpha,
    }

    # Optionally load KD-specific YAML override
    if args.kd_cfg and os.path.exists(args.kd_cfg):
        kd_yaml = load_yaml(args.kd_cfg)
        if "kd" in kd_yaml:
            student_cfg["kd"].update(kd_yaml["kd"])

    return student_cfg, teacher_cfg


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    logger.info(f"Device: {args.device}")
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    cfg, teacher_cfg_dict = build_cfg_from_args(args)

    # ---- Build models ----
    logger.info("Building student model...")
    student = build_rtdetr(cfg)
    logger.info(f"  Student params: {student.num_parameters:,}")

    if args.kd_type != "none":
        logger.info("Building teacher model...")
        teacher = build_rtdetr(teacher_cfg_dict)
        logger.info(f"  Teacher params: {teacher.num_parameters:,}")

        if args.teacher_weights:
            logger.info(f"Loading teacher weights from: {args.teacher_weights}")
            ckpt = torch.load(args.teacher_weights, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            teacher.load_state_dict(state, strict=False)

        model = RTDETRWithKD(student=student, teacher=teacher)
    else:
        # Baseline: no KD, train student directly
        model = student

    if args.student_weights:
        logger.info(f"Loading student weights from: {args.student_weights}")
        ckpt = torch.load(args.student_weights, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        student.load_state_dict(state, strict=False)

    # ---- Build loss ----
    num_classes = cfg["model"].get("num_classes", 80)
    hidden_dim = cfg["model"].get("hidden_dim", 256)

    if args.kd_type != "none":
        loss_fn = KDLoss(
            kd_type=args.kd_type,
            kd_lambda=args.kd_lambda,
            temperature=args.temperature,
            alpha=args.alpha,
            num_classes=num_classes,
            student_dim=hidden_dim,
            teacher_dim=hidden_dim,
        )
    else:
        # Wrap RTDETRLoss to match KDLoss forward signature
        class BaselineLossWrapper(torch.nn.Module):
            def __init__(self, nc):
                super().__init__()
                self.det_loss = RTDETRLoss(num_classes=nc)
            def forward(self, model_outputs, targets):
                # For baseline, model_outputs is the raw dict from RTDETR
                losses = self.det_loss(model_outputs, targets)
                losses["loss_total"] = losses["loss_det"]
                losses["loss_kd"] = torch.tensor(0.0, device=losses["loss_det"].device)
                return losses
        loss_fn = BaselineLossWrapper(num_classes)

    # ---- Build datasets ----
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    img_size = train_cfg.get("img_size", 640)

    logger.info("Building datasets...")
    train_transforms = build_transforms(train=True, img_size=img_size)
    val_transforms = build_transforms(train=False, img_size=img_size)

    train_dataset = COCODetection(
        img_folder=data_cfg["train_img"],
        ann_file=data_cfg["train_ann"],
        transforms=train_transforms,
    )

    if args.mosaic:
        logger.info("Mosaic augmentation ENABLED (p=0.5)")
        train_dataset = MosaicWrapper(train_dataset, img_size=img_size, p=0.5)

    val_dataset = COCODetection(
        img_folder=data_cfg["val_img"],
        ann_file=data_cfg["val_ann"],
        transforms=val_transforms,
        remove_no_annotations=False,
    )

    logger.info(f"Train set: {len(train_dataset)} images")
    logger.info(f"Val set:   {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    # ---- Optimizer and scheduler ----
    optimizer = build_optimizer(model, train_cfg)

    # Also include KDLoss learnable parameters (e.g., FeatureKD projection)
    kd_params = [p for p in loss_fn.parameters() if p.requires_grad]
    if kd_params:
        optimizer.add_param_group({"params": kd_params, "lr": train_cfg["lr_head"]})

    total_iters = len(train_loader) * args.epochs // train_cfg["accumulate_steps"]
    scheduler = build_lr_scheduler(optimizer, train_cfg["warmup_iters"], total_iters)

    # ---- Train ----
    trainer = KDTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )

    # Resume from checkpoint if student weights provided
    start_epoch = 0
    if args.student_weights and os.path.exists(args.student_weights):
        try:
            start_epoch = trainer.load_checkpoint(args.student_weights)
        except Exception:
            pass  # weights may be backbone-only; model already loaded above

    logger.info(f"Starting from epoch {start_epoch + 1}")
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
