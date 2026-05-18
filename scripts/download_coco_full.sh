#!/usr/bin/env bash
# Download full COCO 2017 dataset (118K train + 5K val + annotations).
#
# Usage:
#   bash scripts/download_coco_full.sh [OUTPUT_DIR]
#
# Output structure:
#   <OUTPUT_DIR>/
#     train2017/          (118K images, ~18GB)
#     val2017/            (5K images, ~1GB)
#     annotations/
#       instances_train2017.json
#       instances_val2017.json

set -euo pipefail

OUTPUT_DIR="${1:-$HOME/data/coco}"

echo "Downloading full COCO 2017 to: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/annotations"

# ---- Images ----
echo "[1/3] Downloading train2017 images (~18GB)..."
wget -q --show-progress -O "$OUTPUT_DIR/train2017.zip" \
    "http://images.cocodataset.org/zips/train2017.zip"
unzip -q "$OUTPUT_DIR/train2017.zip" -d "$OUTPUT_DIR"
rm "$OUTPUT_DIR/train2017.zip"
echo "  train2017: done ($(find "$OUTPUT_DIR/train2017" -name "*.jpg" | wc -l) images)"

echo "[2/3] Downloading val2017 images (~1GB)..."
wget -q --show-progress -O "$OUTPUT_DIR/val2017.zip" \
    "http://images.cocodataset.org/zips/val2017.zip"
unzip -q "$OUTPUT_DIR/val2017.zip" -d "$OUTPUT_DIR"
rm "$OUTPUT_DIR/val2017.zip"
echo "  val2017: done ($(find "$OUTPUT_DIR/val2017" -name "*.jpg" | wc -l) images)"

# ---- Annotations ----
echo "[3/3] Downloading annotations (~241MB)..."
wget -q --show-progress -O "$OUTPUT_DIR/annotations_trainval2017.zip" \
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
unzip -q "$OUTPUT_DIR/annotations_trainval2017.zip" -d "$OUTPUT_DIR"
rm "$OUTPUT_DIR/annotations_trainval2017.zip"
echo "  annotations: done"

echo ""
echo "Full COCO 2017 downloaded successfully."
echo "  Train: $OUTPUT_DIR/train2017"
echo "  Val:   $OUTPUT_DIR/val2017"
echo "  Ann:   $OUTPUT_DIR/annotations"
