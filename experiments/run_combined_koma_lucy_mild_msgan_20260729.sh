#!/bin/bash
# Retrain the lucy_mild_aux_msgan recipe (structure loss + adversarial +
# atari/lucy aux hint, from run_lucy_mask_deep_survey.sh /
# run_badrough_retrain_survey.sh) on the 2026-07-29 5-source koma dataset
# (ako5ver2/hamlabi/fitness/gakuen/housei, 1489 tiles), to test whether the
# koma alignment work helps under a recipe that has already shown real
# binary line output (unlike the plain warm_clean_bce unet recipe, which
# reproduces a soft/density-map ceiling regardless of data quality).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_lucy_mild_msgan_20260729
EPOCHS=${EPOCHS:-3}
TRAIN_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
TRAIN_RAW_ROUGH=dataset/pairs_480/train/rough
TRAIN_LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough
ATARI_CKPT=checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

ATARI_TRAIN=results/${TAG}_atari_train
ATARI_EVAL=results/${TAG}_atari_eval
LUCY_MILD_TRAIN_AUX=results/${TAG}_lucy_mild_train
LUCY_MILD_EVAL_AUX=results/${TAG}_lucy_mild_eval

CKPT=checkpoints/${TAG}
OUT=results/${TAG}

# Reuse the already-trained plain warm_clean_bce koma checkpoint's outputs
# on this same eval list, for a direct before/after comparison. Note: the
# old lucy_mask_deep_e2_lucy_mild_aux_msgan reference outputs no longer
# exist (deleted in the 2026-07-25 results image cleanup), so this run only
# compares against the plain-bce koma baseline, not the pre-koma lucy_mild.
BASE_CKPT=checkpoints/combined_koma_20260729_480_warm_clean_bce_e10/best.pth
BASE_MODEL_NAME=${TAG}_baseline_bce_eval8
BASE_OUT=results/${BASE_MODEL_NAME}

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] combined_koma lucy_mild msgan retrain start"
echo "tag=$TAG epochs=$EPOCHS train_list=$TRAIN_LIST line_dir=$TRAIN_LINE_DIR"

echo "=== infer plain-bce koma baseline on eval list (for comparison) ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$BASE_CKPT" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --output-dir "$BASE_OUT" \
  --autocontrast

echo "=== materialize atari aux (train + eval) ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$ATARI_CKPT" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --output-dir "$ATARI_TRAIN" \
  --autocontrast
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$ATARI_CKPT" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --output-dir "$ATARI_EVAL" \
  --autocontrast

echo "=== preprocess lucy_mild aux (train + eval) ==="
"$PY" scripts/preprocess_atari_aux.py \
  --input-dir "$ATARI_TRAIN" --output-dir "$LUCY_MILD_TRAIN_AUX" --mode lucy_mild
"$PY" scripts/preprocess_atari_aux.py \
  --input-dir "$ATARI_EVAL" --output-dir "$LUCY_MILD_EVAL_AUX" --mode lucy_mild

echo "=== train lucy_mild_aux_msgan on koma data ==="
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "$CKPT" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --line-dir "$TRAIN_LINE_DIR" \
  --aux-dir "$LUCY_MILD_TRAIN_AUX" \
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

echo "=== infer lucy_mild_aux_msgan (koma) on eval list ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$CKPT/best.pth" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --aux-dir "$LUCY_MILD_EVAL_AUX" \
  --output-dir "$OUT" \
  --autocontrast

echo "=== montage ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model plain_bce_koma="$BASE_OUT" \
  --model lucy_mild_koma="$OUT" \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    "$BASE_MODEL_NAME" \
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
  echo "atari_checkpoint=$ATARI_CKPT"
  echo "montage=$MONTAGE"
  echo "metrics=$METRICS"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart combined_koma lucy_mild msgan retrain complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma lucy_mild msgan retrain complete"
