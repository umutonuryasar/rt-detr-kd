#!/usr/bin/env bash
# Download a stratified 30K-image subset of COCO 2017 for RT-DETR training.
#
# The script:
#   1. Downloads the full COCO 2017 annotation files (~240MB).
#   2. Uses Python to sample 30K images stratified by category frequency.
#   3. Downloads only the selected images from the COCO S3 bucket.
#   4. Creates a filtered annotation JSON for the subset.
#
# Usage:
#   bash scripts/download_coco_subset.sh [OUTPUT_DIR]
#
# Output structure:
#   OUTPUT_DIR/
#     coco/
#       annotations/
#         instances_train2017_30k.json
#         instances_val2017.json
#       train2017_30k/     (30K sampled images)
#       val2017/           (full 5K val set)

set -euo pipefail

OUTPUT_DIR="${1:-/data}"
COCO_DIR="$OUTPUT_DIR/coco"
ANN_DIR="$COCO_DIR/annotations"
TRAIN_IMG_DIR="$COCO_DIR/train2017_30k"
VAL_IMG_DIR="$COCO_DIR/val2017"
NUM_TRAIN_IMAGES=30000

echo "=== RT-DETR COCO Subset Downloader ==="
echo "Output directory : $COCO_DIR"
echo "Train subset size: $NUM_TRAIN_IMAGES images"
echo ""

# Create directories
mkdir -p "$ANN_DIR" "$TRAIN_IMG_DIR" "$VAL_IMG_DIR"

# ------------------------------------------------------------------
# Step 1: Download annotation files
# ------------------------------------------------------------------
echo "[1/4] Downloading COCO 2017 annotations..."
ANN_ZIP="$OUTPUT_DIR/annotations_trainval2017.zip"
if [ ! -f "$ANN_ZIP" ]; then
    wget -q --show-progress \
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
        -O "$ANN_ZIP"
fi
echo "      Extracting annotations..."
unzip -q -o "$ANN_ZIP" -d "$COCO_DIR"
echo "      Done."

# ------------------------------------------------------------------
# Step 2: Sample 30K training images stratified by category
# ------------------------------------------------------------------
echo "[2/4] Sampling $NUM_TRAIN_IMAGES training images (stratified)..."
export ANN_DIR NUM_TRAIN_IMAGES
ANN_DIR="$ANN_DIR" NUM_TRAIN_IMAGES="$NUM_TRAIN_IMAGES" python3 - <<'PYEOF'
import json, os, random
from collections import defaultdict

ann_dir  = os.environ["ANN_DIR"]
n_images = int(os.environ["NUM_TRAIN_IMAGES"])
seed     = 42
random.seed(seed)

ann_file = os.path.join(ann_dir, "instances_train2017.json")
out_file = os.path.join(ann_dir, "instances_train2017_30k.json")

if os.path.exists(out_file):
    print(f"  Subset annotation already exists: {out_file}")
    with open(out_file) as f:
        subset = json.load(f)
    selected_ids = {img["id"] for img in subset["images"]}
else:
    with open(ann_file) as f:
        coco = json.load(f)
    cat_to_imgs = defaultdict(set)
    for ann in coco["annotations"]:
        cat_to_imgs[ann["category_id"]].add(ann["image_id"])
    img_id_set = {img["id"] for img in coco["images"]}
    n_cats = len(cat_to_imgs)
    quota_per_cat = max(1, n_images // n_cats)
    selected_ids = set()
    for cat_id, img_ids in cat_to_imgs.items():
        candidates = list(img_ids); random.shuffle(candidates)
        selected_ids.update(candidates[:quota_per_cat])
    remaining = list(img_id_set - selected_ids); random.shuffle(remaining)
    selected_ids.update(remaining[:max(0, n_images - len(selected_ids))])
    selected_ids = set(list(selected_ids)[:n_images])
    selected_imgs = [img for img in coco["images"] if img["id"] in selected_ids]
    selected_anns = [ann for ann in coco["annotations"] if ann["image_id"] in selected_ids]
    subset = {"info": coco.get("info", {}), "licenses": coco.get("licenses", []),
              "categories": coco["categories"], "images": selected_imgs, "annotations": selected_anns}
    with open(out_file, "w") as f:
        json.dump(subset, f)
    print(f"  Saved: {out_file}")

# Write image list for wget
id_to_fname = {}
ann_file2 = os.path.join(ann_dir, "instances_train2017.json")
with open(ann_file2) as f:
    all_imgs = json.load(f)["images"]
for img in all_imgs:
    if img["id"] in selected_ids:
        id_to_fname[img["id"]] = img["file_name"]

list_file = os.path.join(ann_dir, "train_subset_files.txt")
with open(list_file, "w") as f:
    for fname in id_to_fname.values():
        f.write(f"http://images.cocodataset.org/train2017/{fname}\n")
print(f"  Image URL list: {list_file}")
PYEOF

# ------------------------------------------------------------------
# Step 3: Download selected training images
# ------------------------------------------------------------------
echo "[3/4] Downloading $NUM_TRAIN_IMAGES training images (parallel, 8 jobs)..."
wget -q --show-progress \
    -P "$TRAIN_IMG_DIR" \
    --input-file="$ANN_DIR/train_subset_files.txt" \
    --tries=3 \
    --wait=1 \
    --random-wait \
    -nc  # skip already-downloaded files

echo "      Downloaded $(ls "$TRAIN_IMG_DIR" | wc -l) images."

# ------------------------------------------------------------------
# Step 4: Download val2017 images
# ------------------------------------------------------------------
echo "[4/4] Downloading COCO val2017 images (~1GB)..."
VAL_ZIP="$OUTPUT_DIR/val2017.zip"
if [ ! -f "$VAL_ZIP" ]; then
    wget -q --show-progress \
        "http://images.cocodataset.org/zips/val2017.zip" \
        -O "$VAL_ZIP"
fi
echo "      Extracting val images..."
unzip -q -o "$VAL_ZIP" -d "$COCO_DIR"
echo "      Done. Val images: $(ls "$VAL_IMG_DIR" | wc -l)"

echo ""
echo "=== Download complete ==="
echo ""
echo "Update your config files:"
echo "  data:"
echo "    train_ann: $ANN_DIR/instances_train2017_30k.json"
echo "    val_ann:   $ANN_DIR/instances_val2017.json"
echo "    train_img: $TRAIN_IMG_DIR"
echo "    val_img:   $VAL_IMG_DIR"
