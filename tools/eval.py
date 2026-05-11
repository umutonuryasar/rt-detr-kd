#!/usr/bin/env python3
"""Evaluate a trained RT-DETR checkpoint on COCO val2017.

Usage
-----
python tools/eval.py \\
    --cfg configs/rtdetr_r18vd_coco.yml \\
    --weights runs/feature_kd_l1.0/checkpoint_best.pth \\
    --coco-val /data/coco/val2017 \\
    --val-ann /data/coco/annotations/instances_val2017.json

Outputs
-------
Prints COCO evaluation summary (AP, AP50, AP75, APs, APm, APl) to stdout
and saves a JSON results file alongside the checkpoint.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.models.rtdetr import build_rtdetr
from src.data.coco_dataset import COCODetection, collate_fn, _COCO_CATEGORIES_80
from src.data.transforms import build_transforms

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCO = True
except ImportError:
    HAS_COCO = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RT-DETR COCO Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cfg", default="configs/rtdetr_r18vd_coco.yml",
                   help="Model config YAML.")
    p.add_argument("--weights", required=True,
                   help="Path to checkpoint .pth file.")
    p.add_argument("--coco-val", default="/data/coco/val2017",
                   help="Path to COCO val2017 image directory.")
    p.add_argument("--val-ann",
                   default="/data/coco/annotations/instances_val2017.json",
                   help="Path to COCO val2017 annotations JSON.")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--score-thresh", type=float, default=0.01,
                   help="Minimum score to include in results.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use-amp", action="store_true", default=True)
    p.add_argument("--output", default=None,
                   help="Path to save JSON results. Defaults to weights_dir/results.json.")
    return p.parse_args()


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_thresh: float = 0.01,
    use_amp: bool = True,
) -> list[dict]:
    """Run model on all validation images and collect COCO-format predictions.

    Returns:
        List of COCO result dicts: {image_id, category_id, bbox, score}.
    """
    idx_to_coco_id = {i: cat_id for i, cat_id in enumerate(_COCO_CATEGORIES_80)}

    model.eval()
    results = []

    for images, targets in loader:
        images = images.to(device)

        with autocast(enabled=use_amp and device.type == "cuda"):
            outputs = model(images)

        pred_logits = outputs["pred_logits"]  # [B, Q, C]
        pred_boxes = outputs["pred_boxes"]    # [B, Q, 4]

        scores, labels = pred_logits.sigmoid().max(dim=-1)  # [B, Q]

        for i, (img_scores, img_labels, img_boxes, target) in enumerate(
            zip(scores, labels, pred_boxes, targets)
        ):
            img_id = target["image_id"]
            if isinstance(img_id, torch.Tensor):
                img_id = img_id.item()

            orig_h, orig_w = target["orig_size"]
            if isinstance(orig_h, torch.Tensor):
                orig_h, orig_w = orig_h.item(), orig_w.item()

            # Filter by score threshold
            keep = img_scores > score_thresh
            img_scores = img_scores[keep]
            img_labels = img_labels[keep]
            img_boxes = img_boxes[keep]

            if img_scores.numel() == 0:
                continue

            # Convert normalized cxcywh -> pixel xywh
            cx = img_boxes[:, 0] * orig_w
            cy = img_boxes[:, 1] * orig_h
            bw = img_boxes[:, 2] * orig_w
            bh = img_boxes[:, 3] * orig_h
            x0 = (cx - bw / 2).clamp(min=0)
            y0 = (cy - bh / 2).clamp(min=0)

            for j in range(len(img_scores)):
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
                    "score": round(img_scores[j].item(), 4),
                })

    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if not HAS_COCO:
        logger.error("pycocotools not installed. Run: pip install pycocotools")
        sys.exit(1)

    # Load config
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # Build model
    logger.info(f"Building model from config: {args.cfg}")
    model = build_rtdetr(cfg)
    logger.info(f"  Parameters: {model.num_parameters:,}")

    # Load weights
    logger.info(f"Loading weights from: {args.weights}")
    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model = model.to(device)

    # Build dataset
    val_transforms = build_transforms(train=False, img_size=args.img_size)
    val_dataset = COCODetection(
        img_folder=args.coco_val,
        ann_file=args.val_ann,
        transforms=val_transforms,
        remove_no_annotations=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    logger.info(f"Validation set: {len(val_dataset)} images")

    # Run inference
    logger.info("Running inference...")
    results = run_inference(
        model, val_loader, device,
        score_thresh=args.score_thresh,
        use_amp=args.use_amp,
    )
    logger.info(f"Total predictions: {len(results)}")

    if not results:
        logger.error("No predictions generated. Check model weights and threshold.")
        sys.exit(1)

    # Save results
    weights_path = Path(args.weights)
    output_path = args.output or str(weights_path.parent / "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f)
    logger.info(f"Saved predictions to: {output_path}")

    # COCO evaluation
    logger.info("Running COCO evaluation...")
    coco_gt = COCO(args.val_ann)
    coco_dt = coco_gt.loadRes(output_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats_names = [
        "AP@[.5:.95]", "AP@.50", "AP@.75",
        "AP-small", "AP-medium", "AP-large",
        "AR@1", "AR@10", "AR@100",
        "AR-small", "AR-medium", "AR-large",
    ]
    logger.info("\n" + "=" * 50)
    logger.info("COCO Evaluation Results")
    logger.info("=" * 50)
    for name, val in zip(stats_names, coco_eval.stats):
        logger.info(f"  {name:<20}: {val:.4f}")

    primary_map = float(coco_eval.stats[0])
    logger.info(f"\nPrimary metric (mAP@[.5:.95]): {primary_map:.4f}")


if __name__ == "__main__":
    main()
