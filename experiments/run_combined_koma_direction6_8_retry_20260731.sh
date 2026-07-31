#!/bin/bash
# Retry Direction 6 (dualhead) and Direction 8 (hed) with a residual-anchored
# output (out = aux_logits + small bounded correction), matching the
# ResidualCleanupGenerator/DarkenOnlyCleanupGenerator pattern that already
# works for this dataset/epoch budget.
#
# Root cause of the first attempt's failure (both landed in the same
# under-inked ~0.19-0.20 F1@2px range, far below cleanup/cleanupdark's ~0.41):
# both DualHeadRefinerGenerator and HedUNetGenerator reconstructed ink_logits
# from scratch with no anchor to the aux (atari) channel, so they started at
# ~sigmoid(0)=0.5 everywhere and had to learn ink density from nothing in only
# 3 epochs -- a budget that was calibrated for residual-anchored models, which
# start already close to a decent output and only learn a small correction.
# This run isolates that one variable: same architectures (dual-head /
# multi-scale side outputs), same 3-epoch budget, same loss weights, but now
# anchored to aux like every adopted model in this family.
#
# Chains both retries sequentially on the single GPU, then builds one combined
# montage/metrics table with the v1 (unanchored) and v2 (anchored) outputs
# side by side against the established baselines, so the fix is directly
# verifiable rather than just asserted.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
PREV_TAG=combined_koma_lucy_mild_msgan_20260729
DARK_TAG=combined_koma_cleanupdark_20260730
DUALHEAD_V1_TAG=combined_koma_dualhead_20260731
HED_V1_TAG=combined_koma_hed_20260731
DUALHEAD_V2_TAG=combined_koma_dualhead_v2_20260731
HED_V2_TAG=combined_koma_hed_v2_20260731
COMBINED_TAG=combined_koma_direction6_8_retry_20260731
EPOCHS=${EPOCHS:-3}
TRAIN_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
TRAIN_RAW_ROUGH=dataset/pairs_480/train/rough
TRAIN_LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough

LOG=logs/${COMBINED_TAG}.log
DONE=logs/${COMBINED_TAG}.done
METRICS=results/fixed_output_metrics_${COMBINED_TAG}_compare.csv
MONTAGE=results/compare_${COMBINED_TAG}.png

LUCY_MILD_TRAIN_AUX=results/${PREV_TAG}_lucy_mild_train
LUCY_MILD_EVAL_AUX=results/${PREV_TAG}_lucy_mild_eval
BASE_MODEL_NAME=${PREV_TAG}_baseline_bce_eval8
BASE_OUT=results/${BASE_MODEL_NAME}
PREV_MODEL_NAME=${PREV_TAG}
PREV_OUT=results/${PREV_TAG}
DARK_MODEL_NAME=${DARK_TAG}
DARK_OUT=results/${DARK_TAG}
DUALHEAD_V1_OUT=results/${DUALHEAD_V1_TAG}
HED_V1_OUT=results/${HED_V1_TAG}

for d in "$LUCY_MILD_TRAIN_AUX" "$LUCY_MILD_EVAL_AUX" "$BASE_OUT" "$PREV_OUT" "$DARK_OUT" "$DUALHEAD_V1_OUT" "$HED_V1_OUT"; do
  if [ ! -d "$d" ]; then
    echo "missing expected reused artifact dir: $d" >&2
    exit 1
  fi
done

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] direction6+8 residual-anchor retry start"

echo "=== train dualhead v2 (residual-anchored, skeleton_weight=0.06) on koma data ==="
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "checkpoints/${DUALHEAD_V2_TAG}" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --line-dir "$TRAIN_LINE_DIR" \
  --aux-dir "$LUCY_MILD_TRAIN_AUX" \
  --epochs "$EPOCHS" \
  --workers 0 \
  --model dualhead \
  --gan --multiscale-gan \
  --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
  --bce-weight 0.75 --l1-weight 0.03 \
  --shape-weight 0.08 --ink-weight 0.14 \
  --binary-weight 0.10 --structure-weight 0.04 \
  --skeleton-weight 0.06 \
  --adv-weight 0.03 --feature-match-weight 0.08 \
  --require-cuda

echo "=== infer dualhead v2 (koma) on eval list ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "checkpoints/${DUALHEAD_V2_TAG}/best.pth" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --aux-dir "$LUCY_MILD_EVAL_AUX" \
  --output-dir "results/${DUALHEAD_V2_TAG}" \
  --autocontrast

echo "=== train hed v2 (residual-anchored, side_weight=0.06) on koma data ==="
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "checkpoints/${HED_V2_TAG}" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --line-dir "$TRAIN_LINE_DIR" \
  --aux-dir "$LUCY_MILD_TRAIN_AUX" \
  --epochs "$EPOCHS" \
  --workers 0 \
  --model hed \
  --gan --multiscale-gan \
  --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
  --bce-weight 0.75 --l1-weight 0.03 \
  --shape-weight 0.08 --ink-weight 0.14 \
  --binary-weight 0.10 --structure-weight 0.04 \
  --side-weight 0.06 \
  --adv-weight 0.03 --feature-match-weight 0.08 \
  --require-cuda

echo "=== infer hed v2 (koma) on eval list ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "checkpoints/${HED_V2_TAG}/best.pth" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --aux-dir "$LUCY_MILD_EVAL_AUX" \
  --output-dir "results/${HED_V2_TAG}" \
  --autocontrast

echo "=== montage (baseline / lucy_mild / cleanupdark / dualhead_v1 / dualhead_v2 / hed_v1 / hed_v2 / GT) ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model plain_bce_koma="$BASE_OUT" \
  --model lucy_mild_koma="$PREV_OUT" \
  --model cleanupdark_koma="$DARK_OUT" \
  --model dualhead_v1="$DUALHEAD_V1_OUT" \
  --model dualhead_v2="results/${DUALHEAD_V2_TAG}" \
  --model hed_v1="$HED_V1_OUT" \
  --model hed_v2="results/${HED_V2_TAG}" \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    "$BASE_MODEL_NAME" \
    "$PREV_MODEL_NAME" \
    "$DARK_MODEL_NAME" \
    "$DUALHEAD_V1_TAG" \
    "$DUALHEAD_V2_TAG" \
    "$HED_V1_TAG" \
    "$HED_V2_TAG" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$COMBINED_TAG"
  echo "epochs=$EPOCHS"
  echo "train_list=$TRAIN_LIST"
  echo "train_line_dir=$TRAIN_LINE_DIR"
  echo "dualhead_v2=residual-anchored skeleton_weight=0.06 skeleton_gain=1.5"
  echo "hed_v2=residual-anchored side_weight=0.06"
  echo "montage=$MONTAGE"
  echo "metrics=$METRICS"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart Direction 6+8 residual-anchor retry complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] direction6+8 residual-anchor retry complete"
