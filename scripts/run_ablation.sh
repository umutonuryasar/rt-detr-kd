#!/usr/bin/env bash
# Run 6 focused ablation configurations for the RT-DETR KD project (Phase 2A).
#
# Ablation grid (production-oriented showcase):
#   Run 0  : Baseline (no KD)
#   Run 5  : Logit-KD  (λ=1.0, T=4)
#   Run 8  : Feature-KD (λ=1.0)
#   Run 14 : CWD (Channel-Wise Distillation, ICCV'21 baseline)
#   Run 16 : Query-KD (novel: decoder object query distillation)
#   Run 17 : Stage-Adaptive KD, cosine schedule (novel: curriculum weighting)
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
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-}"      # R50 teacher weights (set externally)

# ---- Teacher source (B1 cross-architecture KD vs. own simplified teacher) ----
# Set TEACHER_SOURCE=lyuwenyu to use the canonical RT-DETR teacher from the
# lyuwenyu/RT-DETR submodule. Set LYUWENYU_CFG to one of their YAMLs.
# Example:
#   TEACHER_SOURCE=lyuwenyu \
#   LYUWENYU_CFG=third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
#   TEACHER_WEIGHTS=weights/rtdetr_r50vd_6x_coco_from_paddle.pth \
#   TEACHER_MIN_MAP=0.45 \
#   bash scripts/run_ablation.sh /data/coco runs
TEACHER_SOURCE="${TEACHER_SOURCE:-own}"      # own | lyuwenyu
LYUWENYU_CFG="${LYUWENYU_CFG:-}"
TEACHER_MIN_MAP="${TEACHER_MIN_MAP:-0.0}"    # 0.0 disables the gate
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"     # power-user escape hatch

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

    # tee opens log files at pipeline construction time, before Python creates
    # the output dir — explicitly create it first to avoid first-run failures.
    mkdir -p "$output_dir"

    # ---- Skip-if-done (resilience to Colab session drops) ----
    # A run is considered complete when checkpoint_best.pth exists AND eval.log
    # contains a COCO mAP result. Re-running the script after a session drop
    # should not re-train completed runs from scratch.
    if [ -f "$output_dir/checkpoint_best.pth" ] \
       && [ -f "$output_dir/eval.log" ] \
       && grep -q "AP@\[.5:.95\]" "$output_dir/eval.log" 2>/dev/null; then
        echo "  ✓ Already complete — skipping ($output_dir/checkpoint_best.pth + eval.log present)"
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

    # ---- Cross-architecture (lyuwenyu) teacher flags ----
    local lyuwenyu_flag=""
    local min_map="${TEACHER_MIN_MAP}"
    if [ "$kd_type" != "none" ] \
       && [ "$TEACHER_SOURCE" = "lyuwenyu" ] \
       && [ "$teacher_cfg" = "$TEACHER_CFG" ]; then
        if [ -z "$LYUWENYU_CFG" ]; then
            echo "ERROR: TEACHER_SOURCE=lyuwenyu but LYUWENYU_CFG is not set." >&2
            exit 1
        fi
        lyuwenyu_flag="--teacher-source lyuwenyu --lyuwenyu-cfg $LYUWENYU_CFG"
    fi

    local map_gate_flag=""
    if [ "$kd_type" != "none" ]; then
        map_gate_flag="--teacher-min-map $min_map"
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
        $lyuwenyu_flag \
        $map_gate_flag \
        $EXTRA_TRAIN_ARGS \
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

# ---- Run 5: Logit-KD (λ=1.0, T=4) ----
run_experiment 5 "logit" "1.0" "4" "run05_logit_l1.0_t4"

# ---- Run 8: Feature-KD (λ=1.0) ----
run_experiment 8 "feature" "1.0" "4" "run08_feature_l1.0"

# ---- Run 14: CWD (Channel-Wise Distillation, ICCV'21 baseline) ----
run_experiment 14 "cwd" "1.0" "4" "run14_cwd_l1.0" \
    "configs/kd/cwd_kd.yml"

# ---- Run 16: Query-KD (novel: decoder object query distillation) ----
run_experiment 16 "query" "1.0" "4" "run16_query_kd_l1.0" \
    "configs/kd/query_kd.yml"

# ---- Run 17: Stage-Adaptive KD, cosine (novel: curriculum weighting) ----
run_experiment 17 "stage_adaptive" "1.0" "4" "run17_stage_adaptive_cosine" \
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
