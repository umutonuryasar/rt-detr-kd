#!/usr/bin/env bash
# Downloads canonical RT-DETR teacher weights from the lyuwenyu/storage GitHub
# release (the official PyTorch checkpoints converted from PaddleDetection).
#
# The lyuwenyu/RT-DETR HuggingFace repo is private; the public source is:
#   https://github.com/lyuwenyu/storage/releases/tag/v0.1
#
# Weights saved to:  weights/
# Env-var exports:   TEACHER_WEIGHTS, TEACHER_WEIGHTS_R34
set -euo pipefail

BASE_URL="https://github.com/lyuwenyu/storage/releases/download/v0.1"

R50_FILE="rtdetr_r50vd_2x_coco_objects365_from_paddle.pth"   # RT-DETR-L, 53.1 mAP (COCO+O365 pretrain)
R34_FILE="rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"         # RT-DETR-M, 48.9 mAP (COCO only)

WEIGHTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/weights"
mkdir -p "$WEIGHTS_DIR"

download() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "[skip] $(basename "$dest") already exists"
        return
    fi
    echo "[download] $(basename "$dest")"
    echo "  <- $url"
    curl -L --progress-bar --fail -o "$dest" "$url"
}

download "$BASE_URL/$R50_FILE" "$WEIGHTS_DIR/$R50_FILE"
download "$BASE_URL/$R34_FILE" "$WEIGHTS_DIR/$R34_FILE"

echo ""
echo "=== File sizes ==="
du -sh "$WEIGHTS_DIR/$R50_FILE" "$WEIGHTS_DIR/$R34_FILE"

echo ""
echo "=== SHA-256 checksums ==="
sha256sum "$WEIGHTS_DIR/$R50_FILE" "$WEIGHTS_DIR/$R34_FILE"

echo ""
echo "=== Export commands ==="
echo "export TEACHER_WEIGHTS=\"$WEIGHTS_DIR/$R50_FILE\""
echo "export TEACHER_WEIGHTS_R34=\"$WEIGHTS_DIR/$R34_FILE\""
