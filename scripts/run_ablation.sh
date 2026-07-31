#!/usr/bin/env bash
# Run the Phase 2A ablation for the RT-DETR KD project (tech-report scope).
#
# Ablation grid — 9 runs + 1 optional:
#   Run 0 : Baseline (no KD)
#   Run 1 : Logit-KD, binary KL          (sigmoid-matched — default formulation)
#   Run 2 : Logit-KD, softmax KL         (formulation ablation)
#   Run 3 : Feature-KD (enc MSE + attn)  (attn term requires TEACHER_SOURCE=own)
#   Run 4 : CWD (Shu et al., ICCV'21 literature baseline)
#   Run 5 : Query-KD, hungarian matching (novel #1 — requires TEACHER_SOURCE=own)
#   Run 6 : Query-KD, index matching     (matching-contribution ablation)
#   Run 7 : Stage-Adaptive, cosine       (novel #2 — curriculum weighting)
#   Run 8 : Stage-Adaptive, inverse_cosine (curriculum-DIRECTION control:
#           if ≈ cosine, the curriculum claim does not hold — report honestly)
#   Run 9 : MGD (optional extra literature baseline — commented out below)
#
# Model selection: per-epoch eval + best checkpoint use a selection split
# carved FROM the training pool (tools/make_select_split.py); the val set is
# evaluated once at the end and stays untouched by checkpoint selection.
#
# Usage:
#   # one-time, before the first run (fixed seed — same split for ALL runs):
#   python tools/make_select_split.py \
#       --ann $COCO_ROOT/annotations/instances_train2017_30k.json \
#       --num-select 2500 --seed 42
#   bash scripts/run_ablation.sh [COCO_ROOT] [OUTPUT_ROOT]
#
# Prerequisites:
#   - COCO data downloaded (see scripts/download_coco_subset.sh)
#   - Selection split generated (command above)
#   - OWN teacher weights at $TEACHER_WEIGHTS (Query-KD and the Feature-KD
#     attention term are inactive against the lyuwenyu teacher)

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
TEACHER_MIN_MAP="${TEACHER_MIN_MAP:-0.10}"   # own teacher scores 0.142; gate ~5pts below
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"     # power-user escape hatch

EPOCHS=36
BATCH_SIZE="${BATCH_SIZE:-4}"    # 4 on RTX 3050 (4GB); 16 on Colab A100
IMG_SIZE="${IMG_SIZE:-512}"      # 640 OOMs on RTX 3050 with teacher+student
SEED=42                          # single fixed seed for the whole ablation

# ---- Per-method KD lambda (from tools/calibrate_lambda.py) ----------------
# lambda_method = median(L_det)/median(L_KD), measured at init so every KD term
# STARTS at detection-loss scale. Running all methods at lambda=1.0 would
# compare them at wildly different effective KD strengths (raw magnitudes span
# ~1e5x). Paste the calibrated values below; keep them here (visible) rather
# than in YAML, where a key can silently override the CLI flag.
#
# >>> REPLACE THESE PLACEHOLDERS with the calibrate_lambda.py output <<<
LAM_LOGIT_BINARY="${LAM_LOGIT_BINARY:-1.0}"
LAM_LOGIT_SOFTMAX="${LAM_LOGIT_SOFTMAX:-1.0}"
LAM_FEATURE="${LAM_FEATURE:-1.0}"
LAM_CWD="${LAM_CWD:-1.0}"
LAM_QUERY_HUNGARIAN="${LAM_QUERY_HUNGARIAN:-1.0}"
LAM_QUERY_INDEX="${LAM_QUERY_INDEX:-1.0}"
LAM_STAGE_COSINE="${LAM_STAGE_COSINE:-1.0}"
LAM_STAGE_INVCOS="${LAM_STAGE_INVCOS:-1.0}"

# Learning rate: 1e-3 (train_kd.py's default) collapses this architecture —
# the teacher scored 0.027 at 1e-3 vs 0.142 at 1e-4. All runs use 1e-4.
LR_HEAD="${LR_HEAD:-1e-4}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"

# Leakage-free splits (produced by tools/make_select_split.py, seed 42):
#   *_train.json  = 30K subset MINUS the 2.5K selection images
#   *_select.json = the 2.5K selection split (checkpoint selection only)
TRAIN_ANN="$COCO_ROOT/annotations/instances_train2017_30k_train.json"
SELECT_ANN="$COCO_ROOT/annotations/instances_train2017_30k_select.json"
VAL_ANN="$COCO_ROOT/annotations/instances_val2017.json"
TRAIN_IMG="$COCO_ROOT/train2017_30k"
SELECT_IMG="$TRAIN_IMG"          # selection images live in the same folder
VAL_IMG="$COCO_ROOT/val2017"

if [ ! -f "$SELECT_ANN" ]; then
    echo "ERROR: $SELECT_ANN not found." >&2
    echo "Generate the selection split first (fixed seed 42):" >&2
    echo "  python tools/make_select_split.py --ann $COCO_ROOT/annotations/instances_train2017_30k.json --num-select 2500 --seed 42" >&2
    exit 1
fi

# ---- Helpers ----
run_experiment() {
    local run_id="$1"
    local kd_type="$2"
    local kd_lambda="$3"
    local temperature="$4"
    local tag="$5"
    local kd_cfg="${6:-}"         # optional: path to kd config yaml
    local run_extra_args="${7:-}" # optional: per-run flags (e.g. --logit-mode softmax)
    local teacher_cfg="${8:-$TEACHER_CFG}"   # optional: override teacher config
    local teacher_weights="${9:-$TEACHER_WEIGHTS}"  # optional: override teacher weights
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
        --lr-head "$LR_HEAD" \
        --lr-backbone "$LR_BACKBONE" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --img-size "$IMG_SIZE" \
        --output-dir "$output_dir" \
        --coco-train "$TRAIN_IMG" \
        --coco-val "$VAL_IMG" \
        --train-ann "$TRAIN_ANN" \
        --val-ann "$VAL_ANN" \
        --select-img "$SELECT_IMG" \
        --select-ann "$SELECT_ANN" \
        --seed "$SEED" \
        --use-amp \
        $teacher_flag \
        $kd_cfg_flag \
        $lyuwenyu_flag \
        $map_gate_flag \
        $run_extra_args \
        $EXTRA_TRAIN_ARGS \
        2>&1 | tee "$output_dir/train.log" || return $?

    echo ""
    echo "  Benchmarking FPS for $tag..."
    python tools/benchmark_fps.py \
        --cfg "$STUDENT_CFG" \
        --weights "$output_dir/checkpoint_best.pth" \
        --input-size "$IMG_SIZE" \
        --warmup 50 \
        --iters 500 \
        --device cuda \
        2>&1 | tee "$output_dir/fps.log" || return $?

    echo ""
    echo "  Evaluating $tag on COCO val..."
    python tools/eval.py \
        --cfg "$STUDENT_CFG" \
        --weights "$output_dir/checkpoint_best.pth" \
        --coco-val "$VAL_IMG" \
        --val-ann "$VAL_ANN" \
        --img-size "$IMG_SIZE" \
        2>&1 | tee "$output_dir/eval.log" || return $?

    echo "  Finished $tag."
}

# ---- Failure isolation ----
# One dead run (OOM, dropped mount, corrupt image) must not cancel the runs
# after it — the whole point of unattended overnight execution. Failures are
# recorded and the ablation continues; the script exits non-zero at the end.
#
# NOTE on `set -e`: invoking a function under `||` disables errexit inside its
# entire body, which is why every stage in run_experiment carries an explicit
# `|| return $?`. Without those, a failed training run would fall through to
# benchmarking and evaluating a checkpoint that does not exist.
run_or_record() {
    local run_id="$1"
    local tag="$5"
    local status=0

    run_experiment "$@" || status=$?

    if [ "$status" -eq 0 ]; then
        return 0
    fi

    echo ""
    echo "  ✗ FAILED — run $run_id ($tag) exited with status $status."
    echo "    Continuing with the next run. Log: $OUTPUT_ROOT/$tag/train.log"
    printf 'run %s\t%s\texit=%s\n' "$run_id" "$tag" "$status" >> "$FAILURES_FILE"
    return 0
}

mkdir -p "$OUTPUT_ROOT"

# Start each invocation with a clean failure ledger, otherwise a re-run that
# skips every already-completed run would still exit non-zero from stale
# entries.
FAILURES_FILE="$OUTPUT_ROOT/failures.txt"
rm -f "$FAILURES_FILE"

# Track start time
ABLATION_START=$(date +%s)
echo "Starting ablation study at $(date)"
echo "Output root: $OUTPUT_ROOT"
echo ""

# ---- Run 0: Baseline (no KD) ----
run_or_record 0 "none" "0.0" "4" "run00_baseline"

# ---- Run 1: Logit-KD, binary KL (sigmoid-matched default) ----
run_or_record 1 "logit" "$LAM_LOGIT_BINARY" "4" "run01_logit_binary_t4" \
    "" "--logit-mode binary"

# ---- Run 2: Logit-KD, softmax KL (formulation ablation) ----
run_or_record 2 "logit" "$LAM_LOGIT_SOFTMAX" "4" "run02_logit_softmax_t4" \
    "" "--logit-mode softmax"

# ---- Run 3: Feature-KD (enc MSE + attention; attn needs own teacher) ----
run_or_record 3 "feature" "$LAM_FEATURE" "4" "run03_feature"

# ---- Run 4: CWD (Shu et al., ICCV'21 literature baseline) ----
run_or_record 4 "cwd" "$LAM_CWD" "4" "run04_cwd" \
    "configs/kd/cwd_kd.yml"

# ---- Run 5: Query-KD, hungarian matching (novel #1) ----
run_or_record 5 "query" "$LAM_QUERY_HUNGARIAN" "4" "run05_query_hungarian" \
    "configs/kd/query_kd.yml" "--query-matching hungarian"

# ---- Run 6: Query-KD, index matching (matching-contribution ablation) ----
run_or_record 6 "query" "$LAM_QUERY_INDEX" "4" "run06_query_index" \
    "configs/kd/query_kd.yml" "--query-matching index"

# ---- Run 7: Stage-Adaptive, cosine (novel #2) ----
run_or_record 7 "stage_adaptive" "$LAM_STAGE_COSINE" "4" "run07_stage_adaptive_cosine" \
    "configs/kd/stage_adaptive_kd.yml" "--schedule cosine"

# ---- Run 8: Stage-Adaptive, inverse_cosine (curriculum-direction control) ----
run_or_record 8 "stage_adaptive" "$LAM_STAGE_INVCOS" "4" "run08_stage_adaptive_invcos" \
    "configs/kd/stage_adaptive_kd.yml" "--schedule inverse_cosine"

# ---- Run 9 (OPTIONAL): MGD extra literature baseline ----
# run_or_record 9 "mgd" "1.0" "4" "run09_mgd" \
#     "configs/kd/archive/mgd_kd.yml"

# ---- Summary ----
ABLATION_END=$(date +%s)
ELAPSED=$(( (ABLATION_END - ABLATION_START) / 60 ))

echo ""
echo "================================================================"
if [ -s "$FAILURES_FILE" ]; then
    echo " Ablation finished WITH FAILURES"
    echo " Total wall time: ${ELAPSED} minutes"
    echo "================================================================"
    echo ""
    echo "Failed runs ($FAILURES_FILE):"
    cat "$FAILURES_FILE"
    echo ""
    echo "Re-running this script retries only the failed runs — completed runs"
    echo "are skipped via checkpoint_best.pth + eval.log."
    FINAL_STATUS=1
else
    echo " Ablation study complete!"
    echo " Total wall time: ${ELAPSED} minutes"
    echo "================================================================"
    FINAL_STATUS=0
fi
echo ""
echo "Collect results:"
echo "  for d in $OUTPUT_ROOT/run*/; do"
echo "    echo \"\$(basename \$d): \$(grep 'mAP@' \$d/eval.log | tail -1)\""
echo "  done"
echo ""
echo "Or launch the notebook:"
echo "  jupyter notebook notebooks/ablation_analysis.ipynb"

exit "$FINAL_STATUS"