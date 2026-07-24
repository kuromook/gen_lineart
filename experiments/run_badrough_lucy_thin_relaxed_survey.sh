#!/bin/bash
# Focused lucy_thin cleanup retrain with relaxed ink/width controls.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-./venv/bin/python}
TAG=${1:-badrough_lucy_thin_relaxed_e3}
EPOCHS=${EPOCHS:-3}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean_no_ako5_badrough.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
TRAIN_RAW_ROUGH=${TRAIN_RAW_ROUGH:-dataset/pairs_480/train/rough}
EVAL_RAW_ROUGH=${EVAL_RAW_ROUGH:-dataset/pairs_480/test/rough}
TRAIN_LINE_DIR=${TRAIN_LINE_DIR:-dataset/pairs_480/train/line}
AUX_SOURCE_TAG=${AUX_SOURCE_TAG:-badrough_retrain_e3}

LUCY_THIN_TRAIN_AUX=results/${AUX_SOURCE_TAG}_lucy_thin_train
LUCY_THIN_EVAL_AUX=results/${AUX_SOURCE_TAG}_lucy_thin_eval

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
HAZE_METRICS=results/haze_uncertainty_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] badrough lucy_thin relaxed survey start"
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
  echo "missing or incomplete aux dir: $dir" >&2
  return 1
}

ensure_dir_files "$LUCY_THIN_TRAIN_AUX" 728
ensure_dir_files "$LUCY_THIN_EVAL_AUX" 8

train_variant() {
  local label=$1
  local pos_weight=$2
  local ink_weight=$3
  local binary_weight=$4
  local width_weight=$5
  local aux_dropout=$6
  local aux_scale_min=$7
  local structure_weight=$8
  local adv_weight=$9
  local ckpt=checkpoints/${TAG}_${label}
  local out=results/${TAG}_${label}

  echo "=== train ${label} ==="
  "$PY" scripts/train_i2i_survey.py \
    --checkpoint-dir "$ckpt" \
    --file-list "$TRAIN_LIST" \
    --rough-dir "$TRAIN_RAW_ROUGH" \
    --line-dir "$TRAIN_LINE_DIR" \
    --aux-dir "$LUCY_THIN_TRAIN_AUX" \
    --aux-dropout "$aux_dropout" --aux-scale-min "$aux_scale_min" \
    --epochs "$EPOCHS" \
    --workers 0 \
    --model cleanup \
    --gan --multiscale-gan \
    --lr 7e-5 --lr-d 2e-5 --pos-weight "$pos_weight" \
    --bce-weight 0.72 --l1-weight 0.04 \
    --shape-weight 0.08 --ink-weight "$ink_weight" \
    --binary-weight "$binary_weight" --width-weight "$width_weight" \
    --structure-weight "$structure_weight" \
    --adv-weight "$adv_weight" --feature-match-weight 0.08

  echo "=== infer ${label} ==="
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "$ckpt/best.pth" \
    --file-list "$EVAL_LIST" \
    --rough-dir "$EVAL_RAW_ROUGH" \
    --aux-dir "$LUCY_THIN_EVAL_AUX" \
    --output-dir "$out" \
    --autocontrast
}

train_variant lucy_thin_relaxed_a 4.3 0.16 0.12 0.04 0.10 0.85 0.05 0.025
train_variant lucy_thin_relaxed_b 4.5 0.16 0.14 0.05 0.15 0.80 0.05 0.025

echo "=== montage ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model old_lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan \
  --model clean_lucy_thin=results/badrough_retrain_e3_lucy_thin_aux_msgan \
  --model inkwidth_lucy_thin=results/badrough_inkwidth_e3_lucy_thin_inkwidth_aux_msgan \
  --model relaxed_a=results/${TAG}_lucy_thin_relaxed_a \
  --model relaxed_b=results/${TAG}_lucy_thin_relaxed_b \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    lucy_mask_deep_e2_lucy_thin_aux_msgan \
    badrough_retrain_e3_lucy_thin_aux_msgan \
    badrough_inkwidth_e3_lucy_thin_inkwidth_aux_msgan \
    ${TAG}_lucy_thin_relaxed_a \
    ${TAG}_lucy_thin_relaxed_b \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

echo "=== haze metrics ==="
"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --models \
    old_lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan \
    clean_lucy_thin=results/badrough_retrain_e3_lucy_thin_aux_msgan \
    inkwidth_lucy_thin=results/badrough_inkwidth_e3_lucy_thin_inkwidth_aux_msgan \
    relaxed_a=results/${TAG}_lucy_thin_relaxed_a \
    relaxed_b=results/${TAG}_lucy_thin_relaxed_b \
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
  "Lineart badrough lucy_thin relaxed survey complete" \
  "Review $MONTAGE, $METRICS, and $HAZE_METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] badrough lucy_thin relaxed survey complete"
