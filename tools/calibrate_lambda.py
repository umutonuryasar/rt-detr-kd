#!/usr/bin/env python
"""Calibrate per-method kd_lambda so every KD term starts at the same scale.

Why
---
Raw KD loss magnitudes differ by ~1000x across methods (logit ~0.2, query ~2,
feature ~46). Running all nine ablation runs at kd_lambda=1.0 would therefore
compare methods at wildly different *effective* KD strengths — that is a
lambda comparison, not a method comparison.

Rule applied here (identical for every method):

    lambda_method = median(L_det) / median(L_KD)

measured over N batches at initialization, with the real teacher and a freshly
built student. This makes the KD term start at roughly the same magnitude as
the detection loss for every method, so the ablation varies the method and not
the weighting.

The script only PRINTS the values. Paste them into scripts/run_ablation.sh —
keeping lambdas visible in the run script rather than hidden in YAML (a YAML
key silently overriding a CLI flag has bitten this project before).

Usage
-----
    python tools/calibrate_lambda.py \\
        --teacher-weights /content/drive/.../teacher_r50_lr1e4/checkpoint_best.pth \\
        --coco-train /content/coco/train2017 \\
        --train-ann  /content/coco/annotations/instances_train2017_30k_train.json \\
        --batches 20 --batch-size 16 --img-size 512
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from statistics import median

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.rtdetr import build_rtdetr
from src.models.rtdetr_kd import RTDETRWithKD
from src.distillation.kd_loss import KDLoss
from src.data.coco_dataset import COCODetection, collate_fn
from src.data.transforms import build_transforms

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("calibrate")

# (label, kd_type, extra kwargs for KDLoss)
# One entry per ablation run that carries a KD term. Query hungarian/index and
# stage cosine/inverse_cosine are measured separately because their magnitudes
# genuinely differ at initialization.
METHODS = [
    ("logit_binary", "logit", {"logit_mode": "binary"}),
    ("logit_softmax", "logit", {"logit_mode": "softmax"}),
    ("feature", "feature", {}),
    ("cwd", "cwd", {}),
    ("query_hungarian", "query", {"query_matching": "hungarian"}),
    ("query_index", "query", {"query_matching": "index"}),
    ("stage_cosine", "stage_adaptive", {"schedule": "cosine"}),
    ("stage_invcos", "stage_adaptive", {"schedule": "inverse_cosine"}),
]

# KD types that consume decoder cross-attention maps.
ATTN_KD_TYPES = {"feature", "combined", "query", "stage_adaptive"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--student-cfg", default="configs/rtdetr_r18vd_coco.yml")
    p.add_argument("--teacher-cfg", default="configs/rtdetr_r50vd_coco.yml")
    p.add_argument("--teacher-weights", required=True)
    p.add_argument("--coco-train", required=True)
    p.add_argument("--train-ann", required=True)
    p.add_argument(
        "--batches",
        type=int,
        default=20,
        help="Batches to measure per method (median over these).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Must match the ablation's batch size.",
    )
    p.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Must match the ablation's image size.",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_loader(args) -> DataLoader:
    """Same data pipeline the ablation will use (train split, train transforms)."""
    tf = build_transforms(img_size=args.img_size, train=True)
    ds = COCODetection(
        img_folder=args.coco_train, ann_file=args.train_ann, transforms=tf
    )
    logger.info(f"Train set: {len(ds)} images")
    g = torch.Generator()
    g.manual_seed(args.seed)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device == "cuda",
        drop_last=True,
    )


def measure(label, kd_type, extra, args, loader, teacher_cfg, student_cfg):
    """Return (median L_det, median L_KD) over args.batches for one method."""
    device = torch.device(args.device)

    # Rebuild student + loss fresh for every method so the randomly initialised
    # KD projection layers are drawn from the same seed each time — otherwise
    # the measured magnitudes would depend on method ordering.
    torch.manual_seed(args.seed)

    capture = kd_type in ATTN_KD_TYPES
    s_cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in student_cfg.items()}
    t_cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in teacher_cfg.items()}
    s_cfg.setdefault("model", {})["capture_attn"] = capture
    t_cfg.setdefault("model", {})["capture_attn"] = capture

    student = build_rtdetr(s_cfg).to(device)
    teacher = build_rtdetr(t_cfg).to(device)

    ckpt = torch.load(args.teacher_weights, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"Teacher weights do not match the architecture "
            f"(missing={len(missing)}, unexpected={len(unexpected)}). "
            f"Check --teacher-cfg against the checkpoint."
        )

    model = RTDETRWithKD(student, teacher).to(device).train()

    hidden = s_cfg.get("model", {}).get("hidden_dim", 256)
    t_hidden = t_cfg.get("model", {}).get("hidden_dim", 256)
    loss_fn = KDLoss(
        kd_type=kd_type,
        kd_lambda=1.0,
        num_classes=80,
        student_dim=hidden,
        teacher_dim=t_hidden,
        total_epochs=36,
        **extra,
    ).to(device)

    dets, kds = [], []
    it = iter(loader)
    for _ in range(args.batches):
        images, targets = next(it)
        images = images.to(device)
        targets = [
            {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in t.items()
            }
            for t in targets
        ]
        with torch.no_grad():
            out = model(images)
            losses = loss_fn(out, targets, epoch=1)  # epoch 1 = start of training
        dets.append(losses["loss_det"].item())
        kds.append(losses["loss_kd"].item())

    del model, student, teacher, loss_fn
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return median(dets), median(kds)


def main() -> None:
    args = parse_args()
    logger.info(
        f"Device: {args.device} | batches={args.batches} "
        f"batch_size={args.batch_size} img_size={args.img_size}"
    )

    student_cfg = yaml.safe_load(open(args.student_cfg))
    teacher_cfg = yaml.safe_load(open(args.teacher_cfg))
    loader = build_loader(args)

    rows = []
    for label, kd_type, extra in METHODS:
        t0 = time.time()
        m_det, m_kd = measure(
            label, kd_type, extra, args, loader, teacher_cfg, student_cfg
        )
        if m_kd <= 0:
            lam = float("nan")
            note = "KD term is zero — signal unavailable for this teacher"
        else:
            lam = m_det / m_kd
            note = ""
        rows.append((label, m_det, m_kd, lam, note))
        logger.info(
            f"{label:16s} L_det={m_det:8.4f}  L_KD={m_kd:10.4f}  "
            f"lambda={lam:9.4g}  [{time.time() - t0:.0f}s] {note}"
        )

    print("\n" + "=" * 78)
    print("CALIBRATION RESULT — lambda_method = median(L_det) / median(L_KD)")
    print("=" * 78)
    print(f"{'method':<17}{'L_det':>10}{'L_KD':>13}{'lambda':>13}")
    print("-" * 78)
    for label, d, k, lam, _ in rows:
        print(f"{label:<17}{d:>10.4f}{k:>13.4f}{lam:>13.4g}")
    print("-" * 78)
    print("Paste into scripts/run_ablation.sh (one --kd-lambda per run):\n")
    for label, _, _, lam, _ in rows:
        if lam == lam:  # not NaN
            print(f"  {label:<17} --kd-lambda {lam:.4g}")
    print()
    print("Caveats to carry into the writeup:")
    print("  * Magnitudes are measured at initialization; the ratio drifts as")
    print("    training progresses. This equalises the STARTING KD strength,")
    print("    which is the fairest single-lambda-per-method rule available.")
    print("  * stage_adaptive shifts its own feature/logit mix over epochs, so")
    print("    its lambda reflects the epoch-1 mix only.")
    print("  * Same rule, same batches, same seed for every method — that")
    print("    uniformity is what makes the comparison defensible.")


if __name__ == "__main__":
    main()
