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
    p.add_argument(
        "--kd-type",
        default="feature",
        choices=[
            "logit", "feature", "combined",
            "cwd", "mgd", "query", "stage_adaptive",
            "none",
        ],
        help="Type of knowledge distillation. 'none' = baseline training.",
    )
    p.add_argument("--kd-lambda", type=float, default=1.0,
                   help="Weight for the KD loss term (λ).")
    p.add_argument("--temperature", type=float, default=4.0,
                   help="Temperature for logit/combined/stage_adaptive KD.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Attention weight in feature/combined/query/stage_adaptive KD.")
    p.add_argument("--feat-weight", type=float, default=1.0,
                   help="Encoder MSE weight inside feature/combined/stage_adaptive KD.")
    p.add_argument("--logit-weight", type=float, default=0.5,
                   help="Logit-component weight inside combined KD.")
    p.add_argument("--feature-weight", type=float, default=0.5,
                   help="Feature-component weight inside combined KD.")
    p.add_argument("--tau", type=float, default=1.0,
                   help="Spatial softmax temperature for CWD.")
    p.add_argument("--mask-ratio", type=float, default=0.75,
                   help="Token mask ratio for MGD.")
    p.add_argument("--schedule", default="cosine",
                   choices=["cosine", "linear", "step", "sigmoid", "inverse_cosine"],
                   help="Weight schedule for stage_adaptive KD.")

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
    p.add_argument("--teacher-source", default="own",
                   choices=["own", "lyuwenyu"],
                   help="Teacher implementation: 'own' (simplified, defined in "
                        "src.models.rtdetr) or 'lyuwenyu' (canonical RT-DETR "
                        "from the original authors, via third_party submodule).")
    p.add_argument("--lyuwenyu-cfg", default=None,
                   help="If --teacher-source=lyuwenyu: path to one of their "
                        "configs/rtdetr/*.yml files.")

    # Teacher sanity gates (guard against silently-broken KD signal)
    p.add_argument("--teacher-max-missing-ratio", type=float, default=0.05,
                   help="Abort if more than this fraction of teacher state dict "
                        "keys are missing after load_state_dict.")
    p.add_argument("--teacher-min-map", type=float, default=0.0,
                   help="Run a 200-image eval pass on the teacher before training. "
                        "Abort if mAP < this threshold. 0.0 disables the gate. "
                        "Recommended: 0.40 (own teacher) / 0.45 (real RT-DETR).")
    p.add_argument("--teacher-gate-num-images", type=int, default=200,
                   help="Number of val images for the teacher mAP gate.")
    p.add_argument("--skip-teacher-gate", action="store_true", default=False,
                   help="Skip both teacher sanity gates (NOT recommended).")

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
        "feat_weight": args.feat_weight,
        "logit_weight": args.logit_weight,
        "feature_weight": args.feature_weight,
        "tau": args.tau,
        "mask_ratio": args.mask_ratio,
        "schedule": args.schedule,
    }

    # Optionally load KD-specific YAML override.
    # Two YAML schemas are supported:
    #   (a) nested:  kd: { type: ..., lambda: ..., ... }
    #   (b) flat:    kd_type: ...,  kd_lambda: ..., tau: ..., mask_ratio: ...
    if args.kd_cfg and os.path.exists(args.kd_cfg):
        kd_yaml = load_yaml(args.kd_cfg)
        # Schema (a): nested
        if "kd" in kd_yaml and isinstance(kd_yaml["kd"], dict):
            student_cfg["kd"].update(kd_yaml["kd"])
        # Schema (b): flat top-level keys with kd_ prefix or known names
        flat_map = {
            "kd_type": "type",
            "kd_lambda": "lambda",
            "temperature": "temperature",
            "alpha": "alpha",
            "feat_weight": "feat_weight",
            "logit_weight": "logit_weight",
            "feature_weight": "feature_weight",
            "tau": "tau",
            "mask_ratio": "mask_ratio",
            "schedule": "schedule",
        }
        for yaml_key, cfg_key in flat_map.items():
            if yaml_key in kd_yaml:
                student_cfg["kd"][cfg_key] = kd_yaml[yaml_key]

    return student_cfg, teacher_cfg


def _check_teacher_state_dict(
    teacher,
    missing: list,
    unexpected: list,
    max_missing_ratio: float,
) -> None:
    """Abort training if too many teacher weights failed to load.

    A silently-mismatched teacher state dict is the highest-impact bug in a KD
    pipeline — every downstream experiment becomes meaningless. We require that
    at least ``1 - max_missing_ratio`` of the teacher's parameters loaded from
    the checkpoint.
    """
    total_keys = len(teacher.state_dict())
    n_missing = len(missing)
    n_unexpected = len(unexpected)
    missing_ratio = n_missing / max(total_keys, 1)

    logger.info(
        f"Teacher state-dict load: {total_keys - n_missing}/{total_keys} keys "
        f"loaded ({n_missing} missing, {n_unexpected} unexpected)."
    )
    if missing:
        logger.info(f"  First few missing keys: {missing[:5]}")
    if unexpected:
        logger.info(f"  First few unexpected keys: {unexpected[:5]}")

    if missing_ratio > max_missing_ratio:
        raise RuntimeError(
            f"Teacher state-dict mismatch too large: {missing_ratio:.1%} of keys "
            f"missing (threshold {max_missing_ratio:.1%}). Refusing to train — "
            f"a silently broken teacher produces meaningless KD signal. "
            f"Fix the checkpoint or pass --skip-teacher-gate."
        )


def _teacher_map_gate(
    teacher,
    val_loader,
    val_ann_file: str,
    device,
    num_images: int,
    min_map: float,
    use_amp: bool = True,
) -> float:
    """Run a quick mAP eval on the teacher and abort if below threshold.

    Returns the measured mAP. Bounded to ``num_images`` to keep startup time
    reasonable (~30 s on A100, ~2 min on RTX 3050).
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        logger.warning("pycocotools not installed; skipping teacher mAP gate.")
        return 0.0

    if not os.path.exists(val_ann_file):
        logger.warning(f"val_ann_file not found ({val_ann_file}); skipping teacher mAP gate.")
        return 0.0

    from src.data.coco_dataset import _COCO_CATEGORIES_80
    from torch.amp import autocast

    idx_to_coco_id = {i: cid for i, cid in enumerate(_COCO_CATEGORIES_80)}
    teacher.eval()
    results = []
    images_seen = 0

    with torch.no_grad():
        for images, targets in val_loader:
            if images_seen >= num_images:
                break
            images = images.to(device)
            with autocast('cuda', enabled=use_amp and device.type == "cuda"):
                outputs = teacher(images)
            scores, labels = outputs["pred_logits"].sigmoid().max(dim=-1)
            for i in range(images.size(0)):
                if images_seen >= num_images:
                    break
                images_seen += 1
                img_id = targets[i]["image_id"]
                if isinstance(img_id, torch.Tensor):
                    img_id = img_id.item()
                orig_h, orig_w = targets[i]["orig_size"]
                if isinstance(orig_h, torch.Tensor):
                    orig_h, orig_w = orig_h.item(), orig_w.item()
                cx = outputs["pred_boxes"][i, :, 0] * orig_w
                cy = outputs["pred_boxes"][i, :, 1] * orig_h
                bw = outputs["pred_boxes"][i, :, 2] * orig_w
                bh = outputs["pred_boxes"][i, :, 3] * orig_h
                x0 = (cx - bw / 2).clamp(min=0)
                y0 = (cy - bh / 2).clamp(min=0)
                for j in range(scores.size(1)):
                    s = scores[i, j].item()
                    if s < 0.05:
                        continue
                    cid = idx_to_coco_id.get(labels[i, j].item(), -1)
                    if cid < 0:
                        continue
                    results.append({
                        "image_id": img_id, "category_id": cid,
                        "bbox": [round(x0[j].item(), 2), round(y0[j].item(), 2),
                                 round(bw[j].item(), 2), round(bh[j].item(), 2)],
                        "score": round(s, 4),
                    })

    if not results:
        logger.error("Teacher produced no predictions on val sample — definitely broken.")
        raise RuntimeError("Teacher mAP gate: zero predictions above 0.05 score.")

    import tempfile, json as _json
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        _json.dump(results, f)
        results_path = f.name

    coco_gt = COCO(val_ann_file)
    coco_dt = coco_gt.loadRes(results_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = list({r["image_id"] for r in results})
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    teacher_map = float(coco_eval.stats[0])
    os.unlink(results_path)

    logger.info(f"Teacher mAP on {images_seen} val images: {teacher_map:.4f}")
    if teacher_map < min_map:
        raise RuntimeError(
            f"Teacher mAP {teacher_map:.4f} < required {min_map:.4f}. "
            f"Refusing to train — KD with a weak teacher is worse than no KD. "
            f"Verify weights/config or pass --skip-teacher-gate."
        )
    return teacher_map


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

    # After YAML merge the effective KD config may differ from CLI args.
    # Authoritative source from here on is cfg["kd"].
    kd_cfg = cfg["kd"]
    effective_kd_type = kd_cfg["type"]
    logger.info(
        f"Effective KD config: type={effective_kd_type}, "
        f"lambda={kd_cfg['lambda']}, temperature={kd_cfg['temperature']}, "
        f"alpha={kd_cfg['alpha']}, tau={kd_cfg['tau']}, "
        f"mask_ratio={kd_cfg['mask_ratio']}, schedule={kd_cfg['schedule']}"
    )

    # ---- Build models ----
    logger.info("Building student model...")
    student = build_rtdetr(cfg)
    logger.info(f"  Student params: {student.num_parameters:,}")

    if effective_kd_type != "none":
        if args.teacher_source == "lyuwenyu":
            logger.info("Building teacher from lyuwenyu/RT-DETR (canonical)...")
            if args.lyuwenyu_cfg is None:
                raise ValueError(
                    "--teacher-source=lyuwenyu requires --lyuwenyu-cfg pointing "
                    "to one of third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/*.yml."
                )
            from src.models.rtdetr_teacher import build_lyuwenyu_teacher
            teacher = build_lyuwenyu_teacher(
                config=args.lyuwenyu_cfg,
                checkpoint=args.teacher_weights,
            )
            logger.info(f"  Teacher params: {teacher.num_parameters:,}")
        else:
            logger.info("Building teacher model (own simplified RT-DETR)...")
            teacher = build_rtdetr(teacher_cfg_dict)
            logger.info(f"  Teacher params: {teacher.num_parameters:,}")
            if args.teacher_weights:
                logger.info(f"Loading teacher weights from: {args.teacher_weights}")
                ckpt = torch.load(args.teacher_weights, map_location="cpu")
                state = ckpt.get("model_state_dict", ckpt)
                missing, unexpected = teacher.load_state_dict(state, strict=False)
                _check_teacher_state_dict(teacher, missing, unexpected,
                                           max_missing_ratio=args.teacher_max_missing_ratio)
            else:
                logger.warning(
                    "No --teacher-weights given; teacher will be randomly initialized. "
                    "KD signal is meaningless without a competent teacher. "
                    "Pass --teacher-weights or use --skip-teacher-gate to override."
                )

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
    teacher_hidden_dim = teacher_cfg_dict.get("model", teacher_cfg_dict).get(
        "hidden_dim", hidden_dim
    )

    if effective_kd_type != "none":
        loss_fn = KDLoss(
            kd_type=effective_kd_type,
            kd_lambda=kd_cfg["lambda"],
            temperature=kd_cfg["temperature"],
            alpha=kd_cfg["alpha"],
            feat_weight=kd_cfg["feat_weight"],
            logit_weight=kd_cfg["logit_weight"],
            feature_weight=kd_cfg["feature_weight"],
            tau=kd_cfg["tau"],
            mask_ratio=kd_cfg["mask_ratio"],
            num_classes=num_classes,
            student_dim=hidden_dim,
            teacher_dim=teacher_hidden_dim,
            total_epochs=args.epochs,
            schedule=kd_cfg["schedule"],
        )
    else:
        # Wrap RTDETRLoss to match KDLoss forward signature
        class BaselineLossWrapper(torch.nn.Module):
            def __init__(self, nc):
                super().__init__()
                self.det_loss = RTDETRLoss(num_classes=nc)
            def forward(self, model_outputs, targets, epoch: int = 0):
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

    # ---- Teacher mAP sanity gate ----
    # Refuses to train if the teacher is broken (random init, bad weights, etc).
    # Skipped for baseline (no KD) and when --skip-teacher-gate is set.
    if (
        effective_kd_type != "none"
        and not args.skip_teacher_gate
        and args.teacher_min_map > 0.0
    ):
        logger.info(
            f"Running teacher mAP sanity gate "
            f"({args.teacher_gate_num_images} images, threshold {args.teacher_min_map})..."
        )
        teacher.to(device)
        _teacher_map_gate(
            teacher=teacher,
            val_loader=val_loader,
            val_ann_file=data_cfg["val_ann"],
            device=device,
            num_images=args.teacher_gate_num_images,
            min_map=args.teacher_min_map,
            use_amp=train_cfg.get("use_amp", True),
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
