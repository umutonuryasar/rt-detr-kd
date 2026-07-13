#!/usr/bin/env python
"""Carve a model-selection split out of a COCO training annotation file.

Produces two annotation JSONs from one input:
  <out_prefix>_train.json   — training annotations MINUS the selection images
  <out_prefix>_select.json  — the selection split (used for per-epoch eval
                              and best-checkpoint selection via --select-ann)

Why: best-checkpoint selection on the same val set that final numbers are
reported on leaks model-selection information into the reported metric.
This script keeps val untouched: selection images come FROM the training
pool and are REMOVED from it (no leakage in either direction).

Usage:
    python tools/make_select_split.py \
        --ann data/coco/annotations/instances_train2017_30k.json \
        --num-select 2500 \
        --seed 42 \
        --out-prefix data/coco/annotations/instances_train2017_30k

Then in run_ablation.sh:
    TRAIN_ANN=..._train.json
    SELECT_ANN=..._select.json
    (--select-img points at the same image folder as --coco-train.)
"""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ann", required=True, help="Input COCO annotation JSON.")
    p.add_argument("--num-select", type=int, default=2500,
                   help="Number of images in the selection split.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed — keep fixed across the whole ablation so "
                        "every run uses the identical split.")
    p.add_argument("--out-prefix", default=None,
                   help="Output prefix (default: input path without .json).")
    args = p.parse_args()

    ann_path = Path(args.ann)
    out_prefix = Path(args.out_prefix) if args.out_prefix else ann_path.with_suffix("")

    with open(ann_path) as f:
        coco = json.load(f)

    images = coco["images"]
    if args.num_select >= len(images):
        raise SystemExit(
            f"num_select={args.num_select} >= total images {len(images)}"
        )

    rng = random.Random(args.seed)
    select_ids = set(
        img["id"] for img in rng.sample(images, args.num_select)
    )

    def subset(keep_ids: set) -> dict:
        return {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": [im for im in coco["images"] if im["id"] in keep_ids],
            "annotations": [a for a in coco["annotations"]
                            if a["image_id"] in keep_ids],
        }

    all_ids = set(im["id"] for im in images)
    train_ids = all_ids - select_ids

    train_out = Path(f"{out_prefix}_train.json")
    select_out = Path(f"{out_prefix}_select.json")

    train_sub = subset(train_ids)
    select_sub = subset(select_ids)

    with open(train_out, "w") as f:
        json.dump(train_sub, f)
    with open(select_out, "w") as f:
        json.dump(select_sub, f)

    print(f"input   : {len(images)} images, {len(coco['annotations'])} anns")
    print(f"train   : {len(train_sub['images'])} images, "
          f"{len(train_sub['annotations'])} anns -> {train_out}")
    print(f"select  : {len(select_sub['images'])} images, "
          f"{len(select_sub['annotations'])} anns -> {select_out}")
    print(f"seed    : {args.seed} (keep identical across all runs)")
    # Sanity: disjoint and covering
    assert train_ids.isdisjoint(select_ids)
    assert train_ids | select_ids == all_ids
    print("OK: splits are disjoint and cover the input exactly.")


if __name__ == "__main__":
    main()
