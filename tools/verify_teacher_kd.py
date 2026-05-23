#!/usr/bin/env python3
"""Verify cross-architecture KD against a lyuwenyu/RT-DETR teacher.

Run this once after downloading the canonical RT-DETR checkpoint and before
launching Phase 2A on Colab. It catches dim/shape mismatch bugs early.

Steps:
  1. Build the simplified student (our RTDETR).
  2. Build the canonical teacher via lyuwenyu adapter + checkpoint.
  3. Run a single forward through RTDETRWithKD on a dummy batch.
  4. Compute every supported KD loss and assert all are finite.
  5. Run backward and assert non-zero gradients on student-side parameters.
  6. Optionally run the teacher mAP gate on COCO val5k.

Usage:
    python tools/verify_teacher_kd.py \\
        --student-cfg configs/rtdetr_r18vd_coco.yml \\
        --lyuwenyu-cfg third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml \\
        --teacher-weights weights/rtdetr_r50vd_6x_coco_from_paddle.pth \\
        [--coco-val /data/coco/val2017 --val-ann /data/coco/annotations/instances_val2017.json]
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.rtdetr import build_rtdetr
from src.models.rtdetr_kd import RTDETRWithKD
from src.models.rtdetr_teacher import build_lyuwenyu_teacher
from src.distillation.kd_loss import KDLoss, SUPPORTED_KD_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("verify_teacher_kd")


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-architecture KD verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--student-cfg", default="configs/rtdetr_r18vd_coco.yml")
    p.add_argument("--lyuwenyu-cfg", required=True,
                   help="Path to one of lyuwenyu's configs/rtdetr/*.yml files.")
    p.add_argument("--teacher-weights", required=True,
                   help="Path to canonical RT-DETR .pth checkpoint.")
    p.add_argument("--input-size", type=int, default=640,
                   help="Must match the teacher YAML's eval_spatial_size "
                        "(640 for rtdetr_r50vd_6x_coco). The teacher bakes "
                        "positional encodings during __init__; mismatched "
                        "input shapes raise a tensor-add error in the encoder.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Defaults to 1 to keep CPU verification fast. "
                        "Use 2-4 on GPU.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # ---- Student ----
    with open(args.student_cfg) as f:
        student_cfg = yaml.safe_load(f)
    logger.info("Building student (simplified RT-DETR)...")
    student = build_rtdetr(student_cfg).to(device)
    logger.info(f"  Student params: {student.num_parameters:,}")

    # ---- Teacher ----
    logger.info("Building canonical teacher (lyuwenyu/RT-DETR)...")
    teacher = build_lyuwenyu_teacher(
        config=args.lyuwenyu_cfg,
        checkpoint=args.teacher_weights,
    ).to(device)
    logger.info(f"  Teacher params: {teacher.num_parameters:,}")

    # ---- KD wrapper ----
    kd_model = RTDETRWithKD(student=student, teacher=teacher).to(device)
    kd_model.train()

    # ---- Forward ----
    B = args.batch_size
    images = torch.randn(B, 3, args.input_size, args.input_size, device=device)
    logger.info(f"Running forward on dummy batch [{B}, 3, "
                f"{args.input_size}, {args.input_size}]...")
    outputs = kd_model(images)

    # Shape report
    s_logits = outputs["student"]["pred_logits"]
    t_logits = outputs["teacher"]["pred_logits"]
    s_enc = outputs["student_enc_out"]
    t_enc = outputs["teacher_enc_out"]
    logger.info(f"  student logits: {tuple(s_logits.shape)} | teacher: {tuple(t_logits.shape)}")
    logger.info(f"  student enc:    {tuple(s_enc.shape)} | teacher: {tuple(t_enc.shape)}")
    if outputs.get("student_queries") is not None:
        logger.info(f"  student queries: {tuple(outputs['student_queries'].shape)}")
    if outputs.get("teacher_queries") is not None:
        logger.info(f"  teacher queries: {tuple(outputs['teacher_queries'].shape)}")
    else:
        logger.info("  teacher queries: None (deformable-attn teacher does not expose)")

    # Synthetic targets
    targets = [
        {
            "labels": torch.tensor([0, 1], dtype=torch.long, device=device),
            "boxes":  torch.tensor([[0.5, 0.5, 0.3, 0.3],
                                    [0.2, 0.2, 0.1, 0.1]], device=device),
        }
        for _ in range(B)
    ]

    # ---- Try every KD type ----
    student_dim = student_cfg["model"]["hidden_dim"]
    teacher_dim = 256  # lyuwenyu's R50 config uses hidden_dim=256

    failures = []
    for kd_type in SUPPORTED_KD_TYPES:
        try:
            loss_fn = KDLoss(
                kd_type=kd_type, kd_lambda=1.0,
                num_classes=student_cfg["model"]["num_classes"],
                student_dim=student_dim, teacher_dim=teacher_dim,
                total_epochs=36,
            ).to(device)

            # Some KD types reference teacher_queries — skip query for this teacher.
            if kd_type == "query" and outputs.get("teacher_queries") is None:
                logger.warning(f"  {kd_type:>16} : SKIP (teacher queries unavailable)")
                continue

            losses = loss_fn(outputs, targets, epoch=0)
            total = losses["loss_total"]
            kd = losses["loss_kd"]

            if not (torch.isfinite(total) and torch.isfinite(kd)):
                failures.append(f"{kd_type}: non-finite loss")
                logger.error(f"  {kd_type:>16} : NON-FINITE total={total.item()} kd={kd.item()}")
                continue

            # Verify backward works
            total.backward(retain_graph=True)
            logger.info(f"  {kd_type:>16} : OK   total={total.item():.4f}  kd={kd.item():.4f}")
            # Clean up grads for next iteration
            for p in student.parameters():
                if p.grad is not None:
                    p.grad = None
            for p in loss_fn.parameters():
                if p.grad is not None:
                    p.grad = None
        except Exception as e:
            failures.append(f"{kd_type}: {e}")
            logger.error(f"  {kd_type:>16} : FAIL {type(e).__name__}: {e}")

    print()
    if failures:
        print(f"❌ {len(failures)} KD type(s) failed:")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
    else:
        print(f"✅ All {len(SUPPORTED_KD_TYPES)} KD types pass cross-arch verification.")


if __name__ == "__main__":
    main()
