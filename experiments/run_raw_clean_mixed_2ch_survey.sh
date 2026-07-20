#!/bin/bash
# Test raw+haze data as white-collapse suppression for cleaned rough U-Net training.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-raw_clean_mixed_2ch_e3}
EPOCHS=${EPOCHS:-3}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
TRAIN_RAW_ROUGH=${TRAIN_RAW_ROUGH:-dataset/pairs_480/train/rough}
EVAL_RAW_ROUGH=${EVAL_RAW_ROUGH:-dataset/pairs_480/test/rough}
TRAIN_LINE_DIR=${TRAIN_LINE_DIR:-dataset/pairs_480/train/line}
LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
HAZE_METRICS=results/haze_uncertainty_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

MODES=(
  edge_preserve
  line_background_mild
)

mkdir -p logs results
: > "$LOG"

clean_split() {
  local mode=$1
  local list=$2
  local in_dir=$3
  local out_dir=$4
  echo "=== clean ${mode} ${out_dir} ===" | tee -a "$LOG"
  "$PY" tools/preprocess/clean_rough_input.py \
    --file-list "$list" \
    --input-dir "$in_dir" \
    --output-dir "$out_dir" \
    --mode "$mode" 2>&1 | tee -a "$LOG"
}

train_unet_1ch() {
  local label=$1
  local list=$2
  local rough_dir=$3
  local line_dir=$4
  local ckpt=checkpoints/${TAG}_${label}
  echo "=== train ${label} ===" | tee -a "$LOG"
  "$PY" scripts/train_i2i_survey.py \
    --checkpoint-dir "$ckpt" \
    --file-list "$list" \
    --rough-dir "$rough_dir" \
    --line-dir "$line_dir" \
    --epochs "$EPOCHS" \
    --workers 0 \
    --model unet \
    --lr 7e-5 --pos-weight 6.0 \
    --bce-weight 0.85 --l1-weight 0.04 \
    --shape-weight 0.08 --ink-weight 0.14 \
    --binary-weight 0.06 --adv-weight 0.0 \
    --no-autocontrast 2>&1 | tee -a "$LOG"
}

train_unet_2ch() {
  local label=$1
  local clean_dir=$2
  local ckpt=checkpoints/${TAG}_${label}
  echo "=== train ${label} ===" | tee -a "$LOG"
  "$PY" scripts/train_i2i_survey.py \
    --checkpoint-dir "$ckpt" \
    --file-list "$TRAIN_LIST" \
    --rough-dir "$TRAIN_RAW_ROUGH" \
    --line-dir "$TRAIN_LINE_DIR" \
    --aux-dir "$clean_dir" \
    --epochs "$EPOCHS" \
    --workers 0 \
    --model unet \
    --lr 7e-5 --pos-weight 6.0 \
    --bce-weight 0.85 --l1-weight 0.04 \
    --shape-weight 0.08 --ink-weight 0.14 \
    --binary-weight 0.06 --adv-weight 0.0 \
    --no-autocontrast 2>&1 | tee -a "$LOG"
}

infer_1ch() {
  local label=$1
  local rough_dir=$2
  echo "=== infer ${label} ===" | tee -a "$LOG"
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "checkpoints/${TAG}_${label}/best.pth" \
    --file-list "$EVAL_LIST" \
    --rough-dir "$rough_dir" \
    --output-dir "results/${TAG}_${label}" 2>&1 | tee -a "$LOG"
}

infer_2ch() {
  local label=$1
  local clean_dir=$2
  echo "=== infer ${label} ===" | tee -a "$LOG"
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "checkpoints/${TAG}_${label}/best.pth" \
    --file-list "$EVAL_LIST" \
    --rough-dir "$EVAL_RAW_ROUGH" \
    --aux-dir "$clean_dir" \
    --output-dir "results/${TAG}_${label}" 2>&1 | tee -a "$LOG"
}

MODEL_ARGS=(
  --model cleaned_edge_unet_e2=results/cleaned_rough_model_survey_e2_edge_preserve_unet
  --model cleaned_mild_unet_e2=results/cleaned_rough_model_survey_e2_line_background_mild_unet
)
METRIC_MODELS=(
  cleaned_rough_model_survey_e2_edge_preserve_unet
  cleaned_rough_model_survey_e2_line_background_mild_unet
)
HAZE_ARGS=(
  cleaned_edge_unet_e2=results/cleaned_rough_model_survey_e2_edge_preserve_unet
  cleaned_mild_unet_e2=results/cleaned_rough_model_survey_e2_line_background_mild_unet
)

for mode in "${MODES[@]}"; do
  train_clean=results/${TAG}_rough_train_${mode}
  eval_clean=results/${TAG}_rough_eval_${mode}
  clean_split "$mode" "$TRAIN_LIST" "$TRAIN_RAW_ROUGH" "$train_clean"
  clean_split "$mode" "$EVAL_LIST" "$EVAL_RAW_ROUGH" "$eval_clean"

  mixed_root=results/${TAG}_mixed_train_${mode}
  mixed_list=dataset/pairs_480/${TAG}_${mode}_mixed_train.txt
  "$PY" tools/preprocess/build_mixed_rough_dataset.py \
    --file-list "$TRAIN_LIST" \
    --raw-rough-dir "$TRAIN_RAW_ROUGH" \
    --cleaned-rough-dir "$train_clean" \
    --line-dir "$TRAIN_LINE_DIR" \
    --output-root "$mixed_root" \
    --output-list "$mixed_list" 2>&1 | tee -a "$LOG"

  mixed_label=${mode}_mixed_unet_e${EPOCHS}
  twoch_label=${mode}_raw_clean_2ch_unet_e${EPOCHS}

  train_unet_1ch "$mixed_label" "$mixed_list" "$mixed_root/rough" "$mixed_root/line"
  infer_1ch "$mixed_label" "$eval_clean"

  train_unet_2ch "$twoch_label" "$train_clean"
  infer_2ch "$twoch_label" "$eval_clean"

  MODEL_ARGS+=(--model "${mixed_label}=results/${TAG}_${mixed_label}")
  MODEL_ARGS+=(--model "${twoch_label}=results/${TAG}_${twoch_label}")
  METRIC_MODELS+=("${TAG}_${mixed_label}")
  METRIC_MODELS+=("${TAG}_${twoch_label}")
  HAZE_ARGS+=("${mixed_label}=results/${TAG}_${mixed_label}")
  HAZE_ARGS+=("${twoch_label}=results/${TAG}_${twoch_label}")
done

echo "=== montage ===" | tee -a "$LOG"
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  "${MODEL_ARGS[@]}" \
  --output "$MONTAGE" 2>&1 | tee -a "$LOG"

echo "=== fixed metrics ===" | tee -a "$LOG"
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models "${METRIC_MODELS[@]}" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS" 2>&1 | tee -a "$LOG"

echo "=== haze/uncertainty metrics ===" | tee -a "$LOG"
"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --models "${HAZE_ARGS[@]}" \
  --output-csv "$HAZE_METRICS" 2>&1 | tee -a "$LOG"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "modes=${MODES[*]}"
  echo "metrics=$METRICS"
  echo "haze_metrics=$HAZE_METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

echo "done marker: $DONE" | tee -a "$LOG"

./experiments/send_autoloop_notification.sh \
  "Lineart raw+clean mixed/2ch survey complete" \
  "Review $MONTAGE, $METRICS, and $HAZE_METRICS" || true
