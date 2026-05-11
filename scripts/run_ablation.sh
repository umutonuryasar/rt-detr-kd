#!/usr/bin/env bash
# Run all 12 ablation configurations for the RT-DETR KD CS229 project.
#
# Ablation grid:
#   Run 0  : Baseline (no KD)
#   Run 1-6: Logit-KD  (λ ∈ {0.5, 1.0} × T ∈ {2, 4, 8})
#   Run 7-8: Feature-KD (λ ∈ {0.5, 1.0})
#   Run 9-11: Reserved for extended/retrained configs
#
# Usage:
#   bash scripts/run_ablation.sh [COCO_ROOT] [OUTPUT_ROOT]
#
# Prerequisites:
#   - COCO data downloaded (see scripts/download_coco_subset.sh)
#   - Teacher weights available at $TEACHER_WEIGHTS

set -euo pipefail

# ---- Configuration ----
COCO_ROOT="${1:-$HOME/data/coco}"
OUTPUT_ROOT="${2:-runs}"
STUDENT_CFG="configs/rtdetr_r18vd_coco.yml"
TEACHER_CFG="configs/rtdetr_r50vd_coco.yml"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-}"   # set externally or pass via env
EPOCHS=36
BATCH_SIZE=4
IMG_SIZE=512   # 640 OOMs on RTX 3050 with teacher+student; 512 fits in 4GB fp16

TRAIN_ANN="$COCO_ROOT/annotations/instances_train2017_30k.json"
VAL_ANN="$COCO_ROOT/annotations/instances_val2017.json"
TRAIN_IMG="$COCO_ROOT/train2017_30k"
VAL_IMG="$COCO_ROOT/val2017"

# ---- Helpers ----
run_experiment() {
    local run_id="$1"
    local kd_type="$2"
    local kd_lambda="$3"
    local temperature="$4"
    local tag="$5"
    local output_dir="$OUTPUT_ROOT/$tag"

    echo ""
    echo "================================================================"
    echo " Run $run_id: $tag"
    echo "================================================================"
    echo " KD type    : $kd_type"
    echo " KD lambda  : $kd_lambda"
    echo " Temperature: $temperature"
    echo " Output dir : $output_dir"
    echo "================================================================"

    local teacher_flag=""
    if [ "$kd_type" != "none" ] && [ -n "$TEACHER_WEIGHTS" ]; then
        teacher_flag="--teacher-weights $TEACHER_WEIGHTS"
    fi

    python tools/train_kd.py \
        --student-cfg "$STUDENT_CFG" \
        --teacher-cfg "$TEACHER_CFG" \
        --kd-type "$kd_type" \
        --kd-lambda "$kd_lambda" \
        --temperature "$temperature" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --img-size "$IMG_SIZE" \
        --output-dir "$output_dir" \
        --coco-train "$TRAIN_IMG" \
        --coco-val "$VAL_IMG" \
        --train-ann "$TRAIN_ANN" \
        --val-ann "$VAL_ANN" \
        --use-amp \
        $teacher_flag \
        2>&1 | tee "$output_dir/train.log"

    echo ""
    echo "  Benchmarking FPS for $tag..."
    python tools/benchmark_fps.py \
        --cfg "$STUDENT_CFG" \
        --weights "$output_dir/checkpoint_best.pth" \
        --input-size "$IMG_SIZE" \
        --warmup 50 \
        --iters 500 \
        --device cuda \
        2>&1 | tee "$output_dir/fps.log"

    echo ""
    echo "  Evaluating $tag on COCO val..."
    python tools/eval.py \
        --cfg "$STUDENT_CFG" \
        --weights "$output_dir/checkpoint_best.pth" \
        --coco-val "$VAL_IMG" \
        --val-ann "$VAL_ANN" \
        --img-size "$IMG_SIZE" \
        2>&1 | tee "$output_dir/eval.log"

    echo "  Finished $tag."
}

mkdir -p "$OUTPUT_ROOT"

# Track start time
ABLATION_START=$(date +%s)
echo "Starting ablation study at $(date)"
echo "Output root: $OUTPUT_ROOT"
echo ""

# ---- Run 0: Baseline (no KD) ----
run_experiment 0 "none" "0.0" "4" "run00_baseline"

# ---- Runs 1-6: Logit-KD ----
# lambda=0.5
run_experiment 1 "logit" "0.5" "2" "run01_logit_l0.5_t2"
run_experiment 2 "logit" "0.5" "4" "run02_logit_l0.5_t4"
run_experiment 3 "logit" "0.5" "8" "run03_logit_l0.5_t8"

# lambda=1.0
run_experiment 4 "logit" "1.0" "2" "run04_logit_l1.0_t2"
run_experiment 5 "logit" "1.0" "4" "run05_logit_l1.0_t4"
run_experiment 6 "logit" "1.0" "8" "run06_logit_l1.0_t8"

# ---- Runs 7-8: Feature-KD ----
run_experiment 7 "feature" "0.5" "4" "run07_feature_l0.5"
run_experiment 8 "feature" "1.0" "4" "run08_feature_l1.0"

# ---- Runs 9-11: Extended (full 72-epoch retrain of best config) ----
# Uncomment when you have identified the best config from runs 0-8.
#
# echo ""
# echo "Extended run: Feature-KD l=1.0, 72 epochs"
# python tools/train_kd.py \
#     --student-cfg "$STUDENT_CFG" \
#     --teacher-cfg "$TEACHER_CFG" \
#     --kd-type feature \
#     --kd-lambda 1.0 \
#     --epochs 72 \
#     --batch-size "$BATCH_SIZE" \
#     --img-size "$IMG_SIZE" \
#     --output-dir "$OUTPUT_ROOT/run09_feature_l1.0_e72" \
#     --coco-train "$TRAIN_IMG" \
#     --coco-val "$VAL_IMG" \
#     --train-ann "$TRAIN_ANN" \
#     --val-ann "$VAL_ANN" \
#     --mosaic \
#     --use-amp \
#     ${TEACHER_WEIGHTS:+--teacher-weights "$TEACHER_WEIGHTS"}

# ---- Summary ----
ABLATION_END=$(date +%s)
ELAPSED=$(( (ABLATION_END - ABLATION_START) / 60 ))

echo ""
echo "================================================================"
echo " Ablation study complete!"
echo " Total wall time: ${ELAPSED} minutes"
echo "================================================================"
echo ""
echo "Collect results:"
echo "  for d in $OUTPUT_ROOT/run*/; do"
echo "    echo \"\$(basename \$d): \$(grep 'mAP@' \$d/eval.log | tail -1)\""
echo "  done"
echo ""
echo "Or launch the notebook:"
echo "  jupyter notebook notebooks/ablation_analysis.ipynb"
