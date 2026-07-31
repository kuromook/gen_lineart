#!/bin/bash
# Test the HED/DexiNed-style multi-scale side-output refiner (HedUNetGenerator,
# model=hed, model_directions.md Direction 8) on the same 2026-07-29 5-source
# koma dataset used by combined_koma_lucy_mild_msgan_20260729 (model=cleanup),
# combined_koma_cleanupdark_20260730 (model=cleanupdark), and
# combined_koma_dualhead_20260731 (model=dualhead), with every other
# loss/recipe setting held identical.
#
# Isolates one variable: whether supervising two extra 1x1-conv heads reading
# the 1/4- and 1/2-resolution decoder feature maps (each against the GT line
# target downsampled to that resolution) pushes the shared decoder trunk to
# commit to correct line structure at every scale, instead of relying only on
# the final full-res 1x1 conv -- the standard HED side-output trick. The final
# output itself is architecturally identical to the plain 2ch refiner_unet
# (`unet` model + aux_dir); the side heads only shape training gradients.
#
# --side-weight 0.06 matches the magnitude used for the dualhead run's
# --skeleton-weight, so the auxiliary loss contribution is comparable in scale
# across the two experiments.
#
# Reuses the already-materialized atari/lucy_mild aux dirs and the plain-bce
# koma baseline + prior cleanup-family eval outputs rather than regenerating
# them.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_hed_20260731
PREV_TAG=combined_koma_lucy_mild_msgan_20260729
DARK_TAG=combined_koma_cleanupdark_20260730
DUALHEAD_TAG=combined_koma_dualhead_20260731
EPOCHS=${EPOCHS:-3}
TRAIN_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
TRAIN_RAW_ROUGH=dataset/pairs_480/train/rough
TRAIN_LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

# Reused, already-materialized from the earlier koma cleanup-family runs.
LUCY_MILD_TRAIN_AUX=results/${PREV_TAG}_lucy_mild_train
LUCY_MILD_EVAL_AUX=results/${PREV_TAG}_lucy_mild_eval
BASE_MODEL_NAME=${PREV_TAG}_baseline_bce_eval8
BASE_OUT=results/${BASE_MODEL_NAME}
PREV_MODEL_NAME=${PREV_TAG}
PREV_OUT=results/${PREV_TAG}
DARK_MODEL_NAME=${DARK_TAG}
DARK_OUT=results/${DARK_TAG}
DUALHEAD_MODEL_NAME=${DUALHEAD_TAG}
DUALHEAD_OUT=results/${DUALHEAD_TAG}

CKPT=checkpoints/${TAG}
OUT=results/${TAG}

for d in "$LUCY_MILD_TRAIN_AUX" "$LUCY_MILD_EVAL_AUX" "$BASE_OUT" "$PREV_OUT" "$DARK_OUT" "$DUALHEAD_OUT"; do
  if [ ! -d "$d" ]; then
    echo "missing expected reused artifact dir: $d" >&2
    exit 1
  fi
done

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] combined_koma hed (multi-scale side outputs) start"
echo "tag=$TAG epochs=$EPOCHS train_list=$TRAIN_LIST line_dir=$TRAIN_LINE_DIR"
echo "reused_lucy_mild_train_aux=$LUCY_MILD_TRAIN_AUX"
echo "reused_lucy_mild_eval_aux=$LUCY_MILD_EVAL_AUX"

echo "=== train hed (multi-scale side outputs, side_weight=0.06) on koma data ==="
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "$CKPT" \
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

echo "=== infer hed (koma) on eval list ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$CKPT/best.pth" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --aux-dir "$LUCY_MILD_EVAL_AUX" \
  --output-dir "$OUT" \
  --autocontrast

echo "=== montage (baseline / lucy_mild bidirectional / cleanupdark darken-only / dualhead / hed / GT) ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model plain_bce_koma="$BASE_OUT" \
  --model lucy_mild_koma="$PREV_OUT" \
  --model cleanupdark_koma="$DARK_OUT" \
  --model dualhead_koma="$DUALHEAD_OUT" \
  --model hed_koma="$OUT" \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    "$BASE_MODEL_NAME" \
    "$PREV_MODEL_NAME" \
    "$DARK_MODEL_NAME" \
    "$DUALHEAD_MODEL_NAME" \
    "$TAG" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "train_list=$TRAIN_LIST"
  echo "train_line_dir=$TRAIN_LINE_DIR"
  echo "model=hed side_weight=0.06"
  echo "montage=$MONTAGE"
  echo "metrics=$METRICS"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart combined_koma hed (multi-scale side outputs) retrain complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma hed (multi-scale side outputs) retrain complete"
