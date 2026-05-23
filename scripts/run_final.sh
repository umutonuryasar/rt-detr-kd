#!/usr/bin/env bash
# Run final paper experiments on full COCO 118K for 72 epochs.
#
# This script is run AFTER Phase 2A ablation is complete. Edit the
# FINAL_RUNS list below to reflect the top configs selected from Phase 2A.
#
# Includes:
#   - run00: Baseline (no KD)              — always included
#   - run08: Feature-KD λ=1.0             — best projected from ablation
#   - run14: CWD                           — SOTA baseline comparison
#   - run15: MGD                           — SOTA baseline comparison
#   - runXX: Best logit-KD config          — fill in after Phase 2A
#   - runXX: Best novel method (16 or 17)  — fill in after Phase 2A
#
# Each selected run is repeated with 3 random seeds for mean ± std reporting.
#
# Usage:
#   bash scripts/run_final.sh [COCO_ROOT] [OUTPUT_ROOT]
#
# Prerequisites:
#   - Full COCO downloaded (see scripts/download_coco_full.sh)
#   - Phase 2A complete; edit FINAL_RUNS below

set -euo pipefail

COCO_ROOT="${1:-$HOME/data/coco}"
OUTPUT_ROOT="${2:-runs_final}"
STUDENT_CFG="configs/rtdetr_r18vd_coco.yml"
TEACHER_CFG="configs/rtdetr_r50vd_coco.yml"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-}"
TEACHER_WEIGHTS_R34="${TEACHER_WEIGHTS_R34:-}"
EPOCHS=72
BATCH_SIZE=16   # A100 40GB
IMG_SIZE=640

TRAIN_ANN="$COCO_ROOT/annotations/instances_train2017.json"
VAL_ANN="$COCO_ROOT/annotations/instances_val2017.json"
TRAIN_IMG="$COCO_ROOT/train2017"
VAL_IMG="$COCO_ROOT/val2017"

# ---- Helpers ----
run_experiment() {
    local run_id="$1"
    local seed="$2"
    local kd_type="$3"
    local kd_lambda="$4"
    local temperature="$5"
    local tag="$6"
    local kd_cfg="${7:-}"
    local teacher_cfg="${8:-$TEACHER_CFG}"
    local teacher_weights="${9:-$TEACHER_WEIGHTS}"
    local output_dir="$OUTPUT_ROOT/${tag}_seed${seed}"

    echo ""
    echo "================================================================"
    echo " Run $run_id (seed $seed): $tag"
    echo "================================================================"

    mkdir -p "$output_dir"

    # Skip-if-done (resilience to Colab session drops).
    if [ -f "$output_dir/checkpoint_best.pth" ] \
       && [ -f "$output_dir/eval.log" ] \
       && grep -q "AP@\[.5:.95\]" "$output_dir/eval.log" 2>/dev/null; then
        echo "  ✓ Already complete — skipping ($output_dir)"
        return 0
    fi

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
        --seed "$seed" \
        --use-amp \
        $teacher_flag \
        $kd_cfg_flag \
        2>&1 | tee "$output_dir/train.log"

    python tools/eval.py \
        --cfg "$STUDENT_CFG" \
        --weights "$output_dir/checkpoint_best.pth" \
        --coco-val "$VAL_IMG" \
        --val-ann "$VAL_ANN" \
        --img-size "$IMG_SIZE" \
        2>&1 | tee "$output_dir/eval.log"

    python tools/benchmark_fps.py \
        --cfg "$STUDENT_CFG" \
        --weights "$output_dir/checkpoint_best.pth" \
        --input-size "$IMG_SIZE" \
        --warmup 50 \
        --iters 500 \
        --device cuda \
        2>&1 | tee "$output_dir/fps.log"
}

mkdir -p "$OUTPUT_ROOT"
FINAL_START=$(date +%s)
echo "Starting final paper runs at $(date)"
echo "Full COCO: $COCO_ROOT | Epochs: $EPOCHS | Output: $OUTPUT_ROOT"
echo ""

# ---- Phase 2D + 2E: Run each config with 3 seeds ----
# TODO: After Phase 2A, fill in the best logit config and best novel config.

for SEED in 42 1337 2025; do
    # run00: Baseline (no KD)
    run_experiment 0 "$SEED" "none" "0.0" "4" "run00_baseline"

    # run08: Feature-KD λ=1.0
    run_experiment 8 "$SEED" "feature" "1.0" "4" "run08_feature_l1.0"

    # run14: CWD
    run_experiment 14 "$SEED" "cwd" "1.0" "4" "run14_cwd" \
        "configs/kd/cwd_kd.yml"

    # run15: MGD
    run_experiment 15 "$SEED" "mgd" "1.0" "4" "run15_mgd" \
        "configs/kd/mgd_kd.yml"

    # TODO: Add best logit config from Phase 2A
    # run_experiment XX "$SEED" "logit" "X.X" "X" "runXX_logit_best"

    # TODO: Add best novel config from Phase 2A (run16 or run17)
    # run_experiment 16 "$SEED" "query" "1.0" "4" "run16_query_kd" \
    #     "configs/kd/query_kd.yml"
    # run_experiment 17 "$SEED" "stage_adaptive" "1.0" "4" "run17_stage_adaptive" \
    #     "configs/kd/stage_adaptive_kd.yml"
done

FINAL_END=$(date +%s)
ELAPSED=$(( (FINAL_END - FINAL_START) / 60 ))

echo ""
echo "================================================================"
echo " Final runs complete! Total wall time: ${ELAPSED} minutes"
echo "================================================================"
echo ""
echo "Aggregate results (mean ± std) across seeds:"
echo "  python tools/aggregate_results.py --runs-dir $OUTPUT_ROOT"
