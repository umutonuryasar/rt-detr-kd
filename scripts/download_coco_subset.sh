#!/usr/bin/env bash
# Download a stratified 30K-image subset of COCO 2017 for RT-DETR training.
#
# Strategy: download full train2017.zip once (~18 GB), extract to a temp dir,
# move only the 30K selected images, delete the rest. This is faster than
# fetching 30K individual files from the COCO S3 bucket.
#
# Usage:
#   bash scripts/download_coco_subset.sh [OUTPUT_DIR]
#
# Output structure:
#   OUTPUT_DIR/
#     annotations/
#       instances_train2017_30k.json
#       instances_val2017.json
#     train2017_30k/     (30K sampled images)
#     val2017/           (full 5K val set)

set -euo pipefail

OUTPUT_DIR="${1:-/data}"
COCO_DIR="$OUTPUT_DIR"
ANN_DIR="$COCO_DIR/annotations"
TRAIN_SUBSET_DIR="$COCO_DIR/train2017_30k"
TRAIN_FULL_DIR="$COCO_DIR/train2017"        # temp; deleted after subset extracted
VAL_IMG_DIR="$COCO_DIR/val2017"
NUM_TRAIN_IMAGES=30000

echo "=== RT-DETR COCO Subset Downloader (zip-based) ==="
echo "Output directory : $COCO_DIR"
echo "Train subset size: $NUM_TRAIN_IMAGES images"
echo ""

mkdir -p "$ANN_DIR" "$TRAIN_SUBSET_DIR" "$VAL_IMG_DIR"

# ------------------------------------------------------------------
# Step 1: Annotations
# ------------------------------------------------------------------
echo "[1/5] Downloading COCO 2017 annotations (~241 MB)..."
ANN_ZIP="$OUTPUT_DIR/annotations_trainval2017.zip"
if [ ! -f "$ANN_ZIP" ]; then
    wget -q --show-progress \
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
        -O "$ANN_ZIP"
fi
unzip -q -o "$ANN_ZIP" -d "$COCO_DIR"
rm -f "$ANN_ZIP"
echo "  Done."

# ------------------------------------------------------------------
# Step 2: Sample 30K training images stratified by category → write list
# ------------------------------------------------------------------
echo "[2/5] Sampling $NUM_TRAIN_IMAGES training images (stratified)..."
export ANN_DIR NUM_TRAIN_IMAGES
python3 - <<'PYEOF'
import json, os, random
from collections import defaultdict

ann_dir  = os.environ["ANN_DIR"]
n_images = int(os.environ["NUM_TRAIN_IMAGES"])
random.seed(42)

ann_file = os.path.join(ann_dir, "instances_train2017.json")
out_file = os.path.join(ann_dir, "instances_train2017_30k.json")

with open(ann_file) as f:
    coco = json.load(f)

if os.path.exists(out_file):
    print(f"  Subset annotation already exists: {out_file}")
    with open(out_file) as f:
        subset = json.load(f)
    selected_ids = {img["id"] for img in subset["images"]}
else:
    cat_to_imgs = defaultdict(set)
    for ann in coco["annotations"]:
        cat_to_imgs[ann["category_id"]].add(ann["image_id"])
    img_id_set = {img["id"] for img in coco["images"]}
    quota_per_cat = max(1, n_images // len(cat_to_imgs))
    selected_ids = set()
    for img_ids in cat_to_imgs.values():
        candidates = list(img_ids); random.shuffle(candidates)
        selected_ids.update(candidates[:quota_per_cat])
    remaining = list(img_id_set - selected_ids); random.shuffle(remaining)
    selected_ids.update(remaining[:max(0, n_images - len(selected_ids))])
    selected_ids = set(list(selected_ids)[:n_images])

    selected_imgs = [img for img in coco["images"] if img["id"] in selected_ids]
    selected_anns = [ann for ann in coco["annotations"] if ann["image_id"] in selected_ids]
    subset = {
        "info": coco.get("info", {}), "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": selected_imgs, "annotations": selected_anns,
    }
    with open(out_file, "w") as f:
        json.dump(subset, f)
    print(f"  Saved: {out_file}")

# Write selected filenames for the prune step
list_file = os.path.join(ann_dir, "train_subset_filenames.txt")
id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}
with open(list_file, "w") as f:
    for img_id in selected_ids:
        f.write(id_to_fname[img_id] + "\n")
print(f"  Filenames list written: {list_file}  ({len(selected_ids)} entries)")
PYEOF

# ------------------------------------------------------------------
# Step 3: Download full train2017.zip (~18 GB) — one fast stream
# ------------------------------------------------------------------
TRAIN_ZIP="$OUTPUT_DIR/train2017.zip"
if [ -d "$TRAIN_FULL_DIR" ] && [ "$(ls "$TRAIN_FULL_DIR" | wc -l)" -ge 118000 ]; then
    echo "[3/5] Full train2017 already extracted — skipping zip download."
elif [ -f "$TRAIN_ZIP" ]; then
    echo "[3/5] train2017.zip already present — skipping download."
else
    echo "[3/5] Downloading train2017.zip (~18 GB)..."
    wget -q --show-progress \
        "http://images.cocodataset.org/zips/train2017.zip" \
        -O "$TRAIN_ZIP"
fi

# ------------------------------------------------------------------
# Step 4: Extract zip, keep only the 30K subset, delete the rest
# ------------------------------------------------------------------
echo "[4/5] Extracting train2017.zip and pruning to ${NUM_TRAIN_IMAGES} images..."
if [ ! -d "$TRAIN_FULL_DIR" ] || [ "$(ls "$TRAIN_FULL_DIR" | wc -l)" -lt 1000 ]; then
    unzip -q "$TRAIN_ZIP" -d "$COCO_DIR"
fi
rm -f "$TRAIN_ZIP"

# Move selected files to train2017_30k/, delete the rest
FILELIST="$ANN_DIR/train_subset_filenames.txt"
echo "  Moving selected images to $TRAIN_SUBSET_DIR ..."
while IFS= read -r fname; do
    src="$TRAIN_FULL_DIR/$fname"
    if [ -f "$src" ]; then
        mv "$src" "$TRAIN_SUBSET_DIR/$fname"
    fi
done < "$FILELIST"

echo "  Removing remaining full train2017/ (~88K images)..."
rm -rf "$TRAIN_FULL_DIR"
echo "  Subset: $(ls "$TRAIN_SUBSET_DIR" | wc -l) images in $TRAIN_SUBSET_DIR"

# ------------------------------------------------------------------
# Step 5: val2017 (~1 GB zip, fast)
# ------------------------------------------------------------------
echo "[5/5] Downloading val2017.zip (~1 GB)..."
VAL_ZIP="$OUTPUT_DIR/val2017.zip"
if [ ! -d "$VAL_IMG_DIR" ] || [ "$(ls "$VAL_IMG_DIR" | wc -l)" -lt 5000 ]; then
    wget -q --show-progress \
        "http://images.cocodataset.org/zips/val2017.zip" \
        -O "$VAL_ZIP"
    unzip -q "$VAL_ZIP" -d "$COCO_DIR"
    rm -f "$VAL_ZIP"
fi
echo "  Val images: $(ls "$VAL_IMG_DIR" | wc -l)"

echo ""
echo "=== Download complete ==="
echo "  Train subset : $TRAIN_SUBSET_DIR"
echo "  Val          : $VAL_IMG_DIR"
echo "  Annotations  : $ANN_DIR"
echo ""
echo "Config paths:"
echo "  train_ann: $ANN_DIR/instances_train2017_30k.json"
echo "  val_ann:   $ANN_DIR/instances_val2017.json"
echo "  train_img: $TRAIN_SUBSET_DIR"
echo "  val_img:   $VAL_IMG_DIR"
