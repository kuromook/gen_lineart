#!/bin/bash
# Retrain the adopted lucy_mild_aux_msgan recipe (model=cleanup, the current
# best koma checkpoint: combined_koma_lucy_mild_msgan_20260729) with one
# addition: an adversarial-only branch fed from the unpaired-rough skima pool
# (dataset/unpaired_rough/skima, 4917 tiled, no paired line-art GT -- see
# memory project_unpaired_data_pools.md).
#
# Every reconstruction loss (bce/l1/shape/ink/binary/structure/etc.) still
# only sees the paired koma batch, since skima has no line-art target. Each
# step, an extra skima rough+aux batch is passed through G and only
# adversarial_mse(D(rough, G(rough)), 1.0) is added to the generator loss,
# weighted by --unpaired-weight -- this is legal without a GT target because
# the discriminator is judging "does this look like real line art", not
# comparing against a specific target. Isolates one variable against the
# already-adopted lucy_mild checkpoint: does exposing G to a wider pool of
# rough styles (skima is a different, unpaired source) via this cheap
# adversarial-only path change the result.
#
# --unpaired-weight 0.03 matches --adv-weight (the paired adv term), so the
# two adversarial pressures are the same order of magnitude.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_lucy_mild_unpaired_skima_20260731
PREV_TAG=combined_koma_lucy_mild_msgan_20260729
EPOCHS=${EPOCHS:-3}
UNPAIRED_WEIGHT=${UNPAIRED_WEIGHT:-0.03}
TRAIN_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
TRAIN_RAW_ROUGH=dataset/pairs_480/train/rough
TRAIN_LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough

UNPAIRED_FILE_LIST=dataset/pairs_480/valid_train_unpaired_skima.txt
UNPAIRED_ROUGH_DIR=dataset/pairs_480/train/rough_unpaired_skima
UNPAIRED_AUX_DIR=results/unpaired_skima_lucy_mild_aux

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

LUCY_MILD_TRAIN_AUX=results/${PREV_TAG}_lucy_mild_train
LUCY_MILD_EVAL_AUX=results/${PREV_TAG}_lucy_mild_eval
BASE_MODEL_NAME=${PREV_TAG}_baseline_bce_eval8
BASE_OUT=results/${BASE_MODEL_NAME}
PREV_MODEL_NAME=${PREV_TAG}
PREV_OUT=results/${PREV_TAG}

CKPT=checkpoints/${TAG}
OUT=results/${TAG}

for d in "$LUCY_MILD_TRAIN_AUX" "$LUCY_MILD_EVAL_AUX" "$BASE_OUT" "$PREV_OUT" "$UNPAIRED_ROUGH_DIR" "$UNPAIRED_AUX_DIR"; do
  if [ ! -d "$d" ]; then
    echo "missing expected reused artifact dir: $d" >&2
    exit 1
  fi
done
if [ ! -f "$UNPAIRED_FILE_LIST" ]; then
  echo "missing $UNPAIRED_FILE_LIST" >&2
  exit 1
fi

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] combined_koma lucy_mild + unpaired skima start"
echo "tag=$TAG epochs=$EPOCHS unpaired_weight=$UNPAIRED_WEIGHT"
echo "unpaired_file_list=$UNPAIRED_FILE_LIST rows=$(wc -l < "$UNPAIRED_FILE_LIST")"

echo "=== train cleanup + unpaired skima adversarial branch on koma data ==="
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "$CKPT" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --line-dir "$TRAIN_LINE_DIR" \
  --aux-dir "$LUCY_MILD_TRAIN_AUX" \
  --unpaired-rough-file-list "$UNPAIRED_FILE_LIST" \
  --unpaired-rough-dir "$UNPAIRED_ROUGH_DIR" \
  --unpaired-rough-aux-dir "$UNPAIRED_AUX_DIR" \
  --unpaired-weight "$UNPAIRED_WEIGHT" \
  --epochs "$EPOCHS" \
  --workers 0 \
  --model cleanup \
  --gan --multiscale-gan \
  --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
  --bce-weight 0.75 --l1-weight 0.03 \
  --shape-weight 0.08 --ink-weight 0.14 \
  --binary-weight 0.10 --structure-weight 0.04 \
  --adv-weight 0.03 --feature-match-weight 0.08 \
  --require-cuda

echo "=== infer (koma) on eval list ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$CKPT/best.pth" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --aux-dir "$LUCY_MILD_EVAL_AUX" \
  --output-dir "$OUT" \
  --autocontrast

echo "=== montage (baseline / lucy_mild / lucy_mild+unpaired_skima / GT) ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model plain_bce_koma="$BASE_OUT" \
  --model lucy_mild_koma="$PREV_OUT" \
  --model lucy_mild_unpaired_skima="$OUT" \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    "$BASE_MODEL_NAME" \
    "$PREV_MODEL_NAME" \
    "$TAG" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "unpaired_weight=$UNPAIRED_WEIGHT"
  echo "unpaired_file_list=$UNPAIRED_FILE_LIST"
  echo "montage=$MONTAGE"
  echo "metrics=$METRICS"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart combined_koma lucy_mild + unpaired skima retrain complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma lucy_mild + unpaired skima retrain complete"
