#!/bin/bash
# Retrain cleanup candidates on the ako5-badrough-cleaned list with stricter
# ink/width controls to counter the over-recall behavior seen in badrough_retrain_e3.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-./venv/bin/python}
TAG=${1:-badrough_inkwidth_e3}
EPOCHS=${EPOCHS:-3}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean_no_ako5_badrough.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
TRAIN_RAW_ROUGH=${TRAIN_RAW_ROUGH:-dataset/pairs_480/train/rough}
EVAL_RAW_ROUGH=${EVAL_RAW_ROUGH:-dataset/pairs_480/test/rough}
TRAIN_LINE_DIR=${TRAIN_LINE_DIR:-dataset/pairs_480/train/line}
ATARI_CKPT=${ATARI_CKPT:-checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth}

# Reuse the aux materialized during badrough_retrain_e3 when present.
AUX_SOURCE_TAG=${AUX_SOURCE_TAG:-badrough_retrain_e3}

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
HAZE_METRICS=results/haze_uncertainty_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

ATARI_TRAIN=results/${AUX_SOURCE_TAG}_atari_train
ATARI_EVAL=results/${AUX_SOURCE_TAG}_atari_eval
DOG_TRAIN_AUX=results/${AUX_SOURCE_TAG}_dog_train
DOG_EVAL_AUX=results/${AUX_SOURCE_TAG}_dog_eval
LUCY_MILD_TRAIN_AUX=results/${AUX_SOURCE_TAG}_lucy_mild_train
LUCY_MILD_EVAL_AUX=results/${AUX_SOURCE_TAG}_lucy_mild_eval
LUCY_THIN_TRAIN_AUX=results/${AUX_SOURCE_TAG}_lucy_thin_train
LUCY_THIN_EVAL_AUX=results/${AUX_SOURCE_TAG}_lucy_thin_eval

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] badrough ink/width survey start"
echo "tag=$TAG epochs=$EPOCHS train_list=$TRAIN_LIST aux_source_tag=$AUX_SOURCE_TAG"

ensure_dir_files() {
  local dir=$1
  local min_count=$2
  if [ -d "$dir" ]; then
    local count
    count=$(find "$dir" -maxdepth 1 -type f | wc -l)
    if [ "$count" -ge "$min_count" ]; then
      echo "reuse $dir files=$count"
      return 0
    fi
  fi
  return 1
}

echo "=== ensure atari aux ==="
if ! ensure_dir_files "$ATARI_TRAIN" 728; then
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "$ATARI_CKPT" \
    --file-list "$TRAIN_LIST" \
    --rough-dir "$TRAIN_RAW_ROUGH" \
    --output-dir "$ATARI_TRAIN" \
    --autocontrast
fi
if ! ensure_dir_files "$ATARI_EVAL" 8; then
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "$ATARI_CKPT" \
    --file-list "$EVAL_LIST" \
    --rough-dir "$EVAL_RAW_ROUGH" \
    --output-dir "$ATARI_EVAL" \
    --autocontrast
fi

echo "=== ensure atari aux variants ==="
if ! ensure_dir_files "$DOG_TRAIN_AUX" 728; then
  "$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_TRAIN" --output-dir "$DOG_TRAIN_AUX" --mode dog
fi
if ! ensure_dir_files "$DOG_EVAL_AUX" 8; then
  "$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_EVAL" --output-dir "$DOG_EVAL_AUX" --mode dog
fi
if ! ensure_dir_files "$LUCY_MILD_TRAIN_AUX" 728; then
  "$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_TRAIN" --output-dir "$LUCY_MILD_TRAIN_AUX" --mode lucy_mild
fi
if ! ensure_dir_files "$LUCY_MILD_EVAL_AUX" 8; then
  "$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_EVAL" --output-dir "$LUCY_MILD_EVAL_AUX" --mode lucy_mild
fi
if ! ensure_dir_files "$LUCY_THIN_TRAIN_AUX" 728; then
  "$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_TRAIN" --output-dir "$LUCY_THIN_TRAIN_AUX" --mode lucy_thin
fi
if ! ensure_dir_files "$LUCY_THIN_EVAL_AUX" 8; then
  "$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_EVAL" --output-dir "$LUCY_THIN_EVAL_AUX" --mode lucy_thin
fi

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
    --aux-dropout 0.25 --aux-scale-min 0.75 \
    --epochs "$EPOCHS" \
    --workers 0 \
    --model cleanup \
    --gan --multiscale-gan \
    --lr 7e-5 --lr-d 2e-5 --pos-weight 4.0 \
    --bce-weight 0.72 --l1-weight 0.04 \
    --shape-weight 0.08 --ink-weight 0.18 \
    --binary-weight 0.16 --width-weight 0.08 \
    --structure-weight 0.06 \
    --adv-weight 0.025 --feature-match-weight 0.08

  echo "=== infer ${label} ==="
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "$ckpt/best.pth" \
    --file-list "$EVAL_LIST" \
    --rough-dir "$EVAL_RAW_ROUGH" \
    --aux-dir "$eval_aux" \
    --output-dir "$out" \
    --autocontrast
}

train_cleanup dog_inkwidth_aux_msgan "$DOG_TRAIN_AUX" "$DOG_EVAL_AUX"
train_cleanup lucy_mild_inkwidth_aux_msgan "$LUCY_MILD_TRAIN_AUX" "$LUCY_MILD_EVAL_AUX"
train_cleanup lucy_thin_inkwidth_aux_msgan "$LUCY_THIN_TRAIN_AUX" "$LUCY_THIN_EVAL_AUX"

echo "=== montage ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model old_dog=results/halo_mitigation_e2_dog_aux_msgan \
  --model clean_dog=results/badrough_retrain_e3_dog_aux_msgan \
  --model iw_dog=results/${TAG}_dog_inkwidth_aux_msgan \
  --model old_lucy_mild=results/lucy_mask_deep_e2_lucy_mild_aux_msgan \
  --model clean_lucy_mild=results/badrough_retrain_e3_lucy_mild_aux_msgan \
  --model iw_lucy_mild=results/${TAG}_lucy_mild_inkwidth_aux_msgan \
  --model old_lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan \
  --model clean_lucy_thin=results/badrough_retrain_e3_lucy_thin_aux_msgan \
  --model iw_lucy_thin=results/${TAG}_lucy_thin_inkwidth_aux_msgan \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    halo_mitigation_e2_dog_aux_msgan \
    badrough_retrain_e3_dog_aux_msgan \
    ${TAG}_dog_inkwidth_aux_msgan \
    lucy_mask_deep_e2_lucy_mild_aux_msgan \
    badrough_retrain_e3_lucy_mild_aux_msgan \
    ${TAG}_lucy_mild_inkwidth_aux_msgan \
    lucy_mask_deep_e2_lucy_thin_aux_msgan \
    badrough_retrain_e3_lucy_thin_aux_msgan \
    ${TAG}_lucy_thin_inkwidth_aux_msgan \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

echo "=== haze metrics ==="
"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --models \
    old_dog=results/halo_mitigation_e2_dog_aux_msgan \
    clean_dog=results/badrough_retrain_e3_dog_aux_msgan \
    iw_dog=results/${TAG}_dog_inkwidth_aux_msgan \
    old_lucy_mild=results/lucy_mask_deep_e2_lucy_mild_aux_msgan \
    clean_lucy_mild=results/badrough_retrain_e3_lucy_mild_aux_msgan \
    iw_lucy_mild=results/${TAG}_lucy_mild_inkwidth_aux_msgan \
    old_lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan \
    clean_lucy_thin=results/badrough_retrain_e3_lucy_thin_aux_msgan \
    iw_lucy_thin=results/${TAG}_lucy_thin_inkwidth_aux_msgan \
  --output-csv "$HAZE_METRICS"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "train_list=$TRAIN_LIST"
  echo "aux_source_tag=$AUX_SOURCE_TAG"
  echo "montage=$MONTAGE"
  echo "metrics=$METRICS"
  echo "haze_metrics=$HAZE_METRICS"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart badrough ink/width survey complete" \
  "Review $MONTAGE, $METRICS, and $HAZE_METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] badrough ink/width survey complete"
