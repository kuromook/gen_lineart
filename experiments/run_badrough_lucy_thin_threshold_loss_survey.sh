#!/bin/bash
# Focused lucy_thin retrain with threshold-aware differentiable losses.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-./venv/bin/python}
TAG=${1:-badrough_lucy_thin_threshold_e3}
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

echo "[$(date --iso-8601=seconds)] badrough lucy_thin threshold-loss survey start"
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
  local threshold_shape=$2
  local threshold_ink=$3
  local threshold_value=$4
  local threshold_sharpness=$5
  local binary_weight=$6
  local width_weight=$7
  local aux_dropout=$8
  local aux_scale_min=$9
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
    --lr 7e-5 --lr-d 2e-5 --pos-weight 4.5 \
    --bce-weight 0.72 --l1-weight 0.04 \
    --shape-weight 0.08 --ink-weight 0.16 \
    --binary-weight "$binary_weight" --width-weight "$width_weight" \
    --threshold-shape-weight "$threshold_shape" \
    --threshold-ink-weight "$threshold_ink" \
    --threshold-value "$threshold_value" \
    --threshold-sharpness "$threshold_sharpness" \
    --structure-weight 0.05 \
    --adv-weight 0.025 --feature-match-weight 0.08

  echo "=== infer ${label} ==="
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "$ckpt/best.pth" \
    --file-list "$EVAL_LIST" \
    --rough-dir "$EVAL_RAW_ROUGH" \
    --aux-dir "$LUCY_THIN_EVAL_AUX" \
    --output-dir "$out" \
    --autocontrast

  for mode in threshold50 threshold52 threshold54; do
    echo "=== postprocess ${label}_${mode} ==="
    "$PY" tools/compare/postprocess_line_outputs.py \
      --input-dir "$out" \
      --output-dir "results/${TAG}_${label}_post_${mode}" \
      --mode "$mode"
  done
}

train_variant lucy_thin_thresh_light 0.04 0.08 0.52 24.0 0.12 0.04 0.12 0.85
train_variant lucy_thin_thresh_mid 0.07 0.12 0.52 28.0 0.12 0.04 0.12 0.85

echo "=== montage ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model old_lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan \
  --model clean_lucy_thin=results/badrough_retrain_e3_lucy_thin_aux_msgan \
  --model relaxed_b=results/badrough_lucy_thin_relaxed_e3_lucy_thin_relaxed_b \
  --model relaxed_b_t52=results/badrough_lucy_thin_relaxed_e3_lucy_thin_relaxed_b_post_threshold52 \
  --model thresh_light=results/${TAG}_lucy_thin_thresh_light \
  --model light_t52=results/${TAG}_lucy_thin_thresh_light_post_threshold52 \
  --model thresh_mid=results/${TAG}_lucy_thin_thresh_mid \
  --model mid_t52=results/${TAG}_lucy_thin_thresh_mid_post_threshold52 \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    lucy_mask_deep_e2_lucy_thin_aux_msgan \
    badrough_retrain_e3_lucy_thin_aux_msgan \
    badrough_lucy_thin_relaxed_e3_lucy_thin_relaxed_b \
    badrough_lucy_thin_relaxed_e3_lucy_thin_relaxed_b_post_threshold52 \
    ${TAG}_lucy_thin_thresh_light \
    ${TAG}_lucy_thin_thresh_light_post_threshold50 \
    ${TAG}_lucy_thin_thresh_light_post_threshold52 \
    ${TAG}_lucy_thin_thresh_light_post_threshold54 \
    ${TAG}_lucy_thin_thresh_mid \
    ${TAG}_lucy_thin_thresh_mid_post_threshold50 \
    ${TAG}_lucy_thin_thresh_mid_post_threshold52 \
    ${TAG}_lucy_thin_thresh_mid_post_threshold54 \
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
    relaxed_b=results/badrough_lucy_thin_relaxed_e3_lucy_thin_relaxed_b \
    relaxed_b_t52=results/badrough_lucy_thin_relaxed_e3_lucy_thin_relaxed_b_post_threshold52 \
    thresh_light=results/${TAG}_lucy_thin_thresh_light \
    light_t52=results/${TAG}_lucy_thin_thresh_light_post_threshold52 \
    thresh_mid=results/${TAG}_lucy_thin_thresh_mid \
    mid_t52=results/${TAG}_lucy_thin_thresh_mid_post_threshold52 \
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
  "Lineart badrough lucy_thin threshold-loss survey complete" \
  "Review $MONTAGE, $METRICS, and $HAZE_METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] badrough lucy_thin threshold-loss survey complete"
