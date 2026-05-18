#!/usr/bin/env bash
# Run all 18 ablation configurations for the RT-DETR KD project (Phase 2A).
#
# Ablation grid:
#   Run 0  : Baseline (no KD)
#   Run 1-6: Logit-KD  (λ ∈ {0.5, 1.0} × T ∈ {2, 4, 8})
#   Run 7-8: Feature-KD (λ ∈ {0.5, 1.0})
#   Run 9  : Combined-KD (logit + feature, λ=1.0, T=4)
#   Run 10 : Encoder-only partial KD (MSE only, λ=1.0)
#   Run 11 : Attention-only partial KD (cosine only, λ=1.0)
#   Run 12 : Feature-KD, teacher=R34 (capacity analysis)
#   Run 13 : Feature-KD, teacher=R50 (capacity upper bound)
#   Run 14 : CWD (Channel-Wise Distillation, ICCV'21 baseline)
#   Run 15 : MGD (Masked Generative Distillation, ECCV'22 baseline)
#   Run 16 : Query-KD (novel: decoder object query distillation)
#   Run 17 : Stage-Adaptive KD (novel: curriculum weighting)
#
# Usage:
#   bash scripts/run_ablation.sh [COCO_ROOT] [OUTPUT_ROOT]
#
# Prerequisites:
#   - COCO data downloaded (see scripts/download_coco_subset.sh)
#   - Teacher weights available at $TEACHER_WEIGHTS / $TEACHER_WEIGHTS_R34

set -euo pipefail

# ---- Configuration ----
COCO_ROOT="${1:-$HOME/data/coco}"
OUTPUT_ROOT="${2:-runs}"
STUDENT_CFG="configs/rtdetr_r18vd_coco.yml"
TEACHER_CFG="configs/rtdetr_r50vd_coco.yml"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-}"      # R50 teacher weights (set externally)
TEACHER_WEIGHTS_R34="${TEACHER_WEIGHTS_R34:-}"  # R34 teacher weights (set externally)
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
    local kd_cfg="${6:-}"         # optional: path to kd config yaml
    local teacher_cfg="${7:-$TEACHER_CFG}"   # optional: override teacher config
    local teacher_weights="${8:-$TEACHER_WEIGHTS}"  # optional: override teacher weights
    local output_dir="$OUTPUT_ROOT/$tag"

    echo ""
    echo "================================================================"
    echo " Run $run_id: $tag"
    echo "================================================================"
    echo " KD type    : $kd_type"
    echo " KD lambda  : $kd_lambda"
    echo " Temperature: $temperature"
    echo " KD cfg     : ${kd_cfg:-<none>}"
    echo " Teacher cfg: $teacher_cfg"
    echo " Output dir : $output_dir"
    echo "================================================================"

    local teacher_flag=""
    if [ "$kd_type" != "none" ] && [ -n "$teacher_weights" ]; then
        teacher_flag="--teacher-weights $teacher_weights"
    fi

    local kd_cfg_flag=""
    if [ -n "$kd_cfg" ]; then
        kd_cfg_flag="--kd-cfg $kd_cfg"
    fi

    python tools/train_kd.py \
        --student-cfg "$STUDENT_CFG" \
        --teacher-cfg "$teacher_cfg" \
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
        $kd_cfg_flag \
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

# ---- Run 9: Combined-KD (logit + feature) ----
run_experiment 9 "combined" "1.0" "4" "run09_combined_l1.0_t4" \
    "configs/kd/combined_kd.yml"

# ---- Run 10: Encoder-only partial KD ----
run_experiment 10 "feature" "1.0" "4" "run10_encoder_only_l1.0" \
    "configs/kd/encoder_only_kd.yml"

# ---- Run 11: Attention-only partial KD ----
run_experiment 11 "feature" "1.0" "4" "run11_attention_only_l1.0" \
    "configs/kd/attention_only_kd.yml"

# ---- Run 12: Feature-KD, teacher=R34 (capacity analysis) ----
run_experiment 12 "feature" "1.0" "4" "run12_feature_teacher_r34" \
    "" "configs/rtdetr_r34vd_coco.yml" "$TEACHER_WEIGHTS_R34"

# ---- Run 13: Feature-KD, teacher=R50 (capacity upper bound) ----
run_experiment 13 "feature" "1.0" "4" "run13_feature_teacher_r50" \
    "" "$TEACHER_CFG" "$TEACHER_WEIGHTS"

# ---- Run 14: CWD (Channel-Wise Distillation, ICCV'21 baseline) ----
run_experiment 14 "cwd" "1.0" "4" "run14_cwd_l1.0" \
    "configs/kd/cwd_kd.yml"

# ---- Run 15: MGD (Masked Generative Distillation, ECCV'22 baseline) ----
run_experiment 15 "mgd" "1.0" "4" "run15_mgd_l1.0" \
    "configs/kd/mgd_kd.yml"

# ---- Run 16: Query-KD (novel: decoder object query distillation) ----
run_experiment 16 "query" "1.0" "4" "run16_query_kd_l1.0" \
    "configs/kd/query_kd.yml"

# ---- Run 17: Stage-Adaptive KD (novel: curriculum weighting) ----
run_experiment 17 "stage_adaptive" "1.0" "4" "run17_stage_adaptive_l1.0" \
    "configs/kd/stage_adaptive_kd.yml"

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
