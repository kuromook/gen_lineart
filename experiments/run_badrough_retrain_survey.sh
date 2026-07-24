#!/bin/bash
# Retrain current key candidates after excluding old ako5 bad-rough tiles.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-./venv/bin/python}
TAG=${1:-badrough_retrain_e3}
EPOCHS=${EPOCHS:-3}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean_no_ako5_badrough.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
TRAIN_RAW_ROUGH=${TRAIN_RAW_ROUGH:-dataset/pairs_480/train/rough}
EVAL_RAW_ROUGH=${EVAL_RAW_ROUGH:-dataset/pairs_480/test/rough}
TRAIN_LINE_DIR=${TRAIN_LINE_DIR:-dataset/pairs_480/train/line}
ATARI_CKPT=${ATARI_CKPT:-checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth}

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
HAZE_METRICS=results/haze_uncertainty_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

EDGE_TRAIN_AUX=results/${TAG}_rough_train_edge_preserve
EDGE_EVAL_AUX=results/${TAG}_rough_eval_edge_preserve
ATARI_TRAIN=results/${TAG}_atari_train
ATARI_EVAL=results/${TAG}_atari_eval
DOG_TRAIN_AUX=results/${TAG}_dog_train
DOG_EVAL_AUX=results/${TAG}_dog_eval
LUCY_MILD_TRAIN_AUX=results/${TAG}_lucy_mild_train
LUCY_MILD_EVAL_AUX=results/${TAG}_lucy_mild_eval
LUCY_THIN_TRAIN_AUX=results/${TAG}_lucy_thin_train
LUCY_THIN_EVAL_AUX=results/${TAG}_lucy_thin_eval

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] badrough retrain survey start"
echo "tag=$TAG epochs=$EPOCHS train_list=$TRAIN_LIST"

echo "=== materialize edge_preserve rough aux ==="
"$PY" tools/preprocess/clean_rough_input.py \
  --file-list "$TRAIN_LIST" \
  --input-dir "$TRAIN_RAW_ROUGH" \
  --output-dir "$EDGE_TRAIN_AUX" \
  --mode edge_preserve
"$PY" tools/preprocess/clean_rough_input.py \
  --file-list "$EVAL_LIST" \
  --input-dir "$EVAL_RAW_ROUGH" \
  --output-dir "$EDGE_EVAL_AUX" \
  --mode edge_preserve

echo "=== train edge_bghaze ==="
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir checkpoints/${TAG}_edge_bghaze \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --line-dir "$TRAIN_LINE_DIR" \
  --aux-dir "$EDGE_TRAIN_AUX" \
  --epochs "$EPOCHS" \
  --workers 0 \
  --model unet \
  --lr 7e-5 --pos-weight 6.0 \
  --bce-weight 0.85 --l1-weight 0.04 \
  --shape-weight 0.08 --ink-weight 0.14 \
  --binary-weight 0.06 --adv-weight 0.0 \
  --background-haze-weight 0.06 --background-haze-radius 9 \
  --no-autocontrast

echo "=== infer edge_bghaze ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint checkpoints/${TAG}_edge_bghaze/best.pth \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --aux-dir "$EDGE_EVAL_AUX" \
  --output-dir results/${TAG}_edge_bghaze

echo "=== materialize atari aux ==="
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

echo "=== preprocess atari aux variants ==="
"$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_TRAIN" --output-dir "$DOG_TRAIN_AUX" --mode dog
"$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_EVAL" --output-dir "$DOG_EVAL_AUX" --mode dog
"$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_TRAIN" --output-dir "$LUCY_MILD_TRAIN_AUX" --mode lucy_mild
"$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_EVAL" --output-dir "$LUCY_MILD_EVAL_AUX" --mode lucy_mild
"$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_TRAIN" --output-dir "$LUCY_THIN_TRAIN_AUX" --mode lucy_thin
"$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_EVAL" --output-dir "$LUCY_THIN_EVAL_AUX" --mode lucy_thin

train_cleanup() {
  local label=$1
  local train_aux=$2
  local eval_aux=$3
  local ckpt=checkpoints/${TAG}_${label}
  local out=results/${TAG}_${label}
  echo "=== train ${label} ==="
  "$PY" scripts/train_i2i_survey.py \
    --checkpoint-dir "$ckpt" \
    --file-list "$TRAIN_LIST" \
    --rough-dir "$TRAIN_RAW_ROUGH" \
    --line-dir "$TRAIN_LINE_DIR" \
    --aux-dir "$train_aux" \
    --epochs "$EPOCHS" \
    --workers 0 \
    --model cleanup \
    --gan --multiscale-gan \
    --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
    --bce-weight 0.75 --l1-weight 0.03 \
    --shape-weight 0.08 --ink-weight 0.14 \
    --binary-weight 0.10 --structure-weight 0.04 \
    --adv-weight 0.03 --feature-match-weight 0.08

  echo "=== infer ${label} ==="
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "$ckpt/best.pth" \
    --file-list "$EVAL_LIST" \
    --rough-dir "$EVAL_RAW_ROUGH" \
    --aux-dir "$eval_aux" \
    --output-dir "$out" \
    --autocontrast
}

train_cleanup dog_aux_msgan "$DOG_TRAIN_AUX" "$DOG_EVAL_AUX"
train_cleanup lucy_mild_aux_msgan "$LUCY_MILD_TRAIN_AUX" "$LUCY_MILD_EVAL_AUX"
train_cleanup lucy_thin_aux_msgan "$LUCY_THIN_TRAIN_AUX" "$LUCY_THIN_EVAL_AUX"

echo "=== montage ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model old_edge_bghaze=results/2ch_haze_control_e3_edge_preserve_2ch_bghaze06_e3 \
  --model new_edge_bghaze=results/${TAG}_edge_bghaze \
  --model old_dog=results/halo_mitigation_e2_dog_aux_msgan \
  --model new_dog=results/${TAG}_dog_aux_msgan \
  --model old_lucy_mild=results/lucy_mask_deep_e2_lucy_mild_aux_msgan \
  --model new_lucy_mild=results/${TAG}_lucy_mild_aux_msgan \
  --model old_lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan \
  --model new_lucy_thin=results/${TAG}_lucy_thin_aux_msgan \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    2ch_haze_control_e3_edge_preserve_2ch_bghaze06_e3 \
    ${TAG}_edge_bghaze \
    halo_mitigation_e2_dog_aux_msgan \
    ${TAG}_dog_aux_msgan \
    lucy_mask_deep_e2_lucy_mild_aux_msgan \
    ${TAG}_lucy_mild_aux_msgan \
    lucy_mask_deep_e2_lucy_thin_aux_msgan \
    ${TAG}_lucy_thin_aux_msgan \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

echo "=== haze metrics ==="
"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --models \
    old_edge_bghaze=results/2ch_haze_control_e3_edge_preserve_2ch_bghaze06_e3 \
    new_edge_bghaze=results/${TAG}_edge_bghaze \
    old_dog=results/halo_mitigation_e2_dog_aux_msgan \
    new_dog=results/${TAG}_dog_aux_msgan \
    old_lucy_mild=results/lucy_mask_deep_e2_lucy_mild_aux_msgan \
    new_lucy_mild=results/${TAG}_lucy_mild_aux_msgan \
    old_lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan \
    new_lucy_thin=results/${TAG}_lucy_thin_aux_msgan \
  --output-csv "$HAZE_METRICS"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "train_list=$TRAIN_LIST"
  echo "atari_checkpoint=$ATARI_CKPT"
  echo "montage=$MONTAGE"
  echo "metrics=$METRICS"
  echo "haze_metrics=$HAZE_METRICS"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart badrough retrain survey complete" \
  "Review $MONTAGE, $METRICS, and $HAZE_METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] badrough retrain survey complete"
