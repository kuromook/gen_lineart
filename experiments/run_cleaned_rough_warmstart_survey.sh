#!/bin/bash
# Compare longer U-Net training with warm-started ResNet-GAN on cleaned rough inputs.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-cleaned_rough_warmstart_e5}
EPOCHS=${EPOCHS:-5}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
TRAIN_ROUGH_DIR=${TRAIN_ROUGH_DIR:-dataset/pairs_480/train/rough}
EVAL_ROUGH_DIR=${EVAL_ROUGH_DIR:-dataset/pairs_480/test/rough}
TRAIN_LINE_DIR=${TRAIN_LINE_DIR:-dataset/pairs_480/train/line}
WARM_RESNET_CKPT=${WARM_RESNET_CKPT:-checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth}
LOG=logs/${TAG}.log
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
HAZE_METRICS=results/haze_uncertainty_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png
DONE=logs/${TAG}.done

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

infer_batch() {
  local label=$1
  local ckpt=$2
  local rough_dir=$3
  local out_dir=results/${TAG}_${label}
  echo "=== infer ${label} ===" | tee -a "$LOG"
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "$ckpt" \
    --file-list "$EVAL_LIST" \
    --rough-dir "$rough_dir" \
    --output-dir "$out_dir" 2>&1 | tee -a "$LOG"
}

train_unet() {
  local mode=$1
  local rough_dir=$2
  local label=${mode}_unet_e${EPOCHS}
  local ckpt=checkpoints/${TAG}_${label}
  echo "=== train ${label} ===" | tee -a "$LOG"
  "$PY" scripts/train_i2i_survey.py \
    --checkpoint-dir "$ckpt" \
    --file-list "$TRAIN_LIST" \
    --rough-dir "$rough_dir" \
    --line-dir "$TRAIN_LINE_DIR" \
    --epochs "$EPOCHS" \
    --workers 0 \
    --model unet \
    --lr 1e-4 --pos-weight 5.0 \
    --bce-weight 0.85 --l1-weight 0.04 \
    --shape-weight 0.08 --ink-weight 0.10 \
    --binary-weight 0.08 --adv-weight 0.0 \
    --no-autocontrast 2>&1 | tee -a "$LOG"
}

train_resnet_gan_warm() {
  local mode=$1
  local rough_dir=$2
  local label=${mode}_resnet_gan_warm_e${EPOCHS}
  local ckpt=checkpoints/${TAG}_${label}
  echo "=== train ${label} ===" | tee -a "$LOG"
  "$PY" scripts/train_i2i_survey.py \
    --checkpoint-dir "$ckpt" \
    --file-list "$TRAIN_LIST" \
    --rough-dir "$rough_dir" \
    --line-dir "$TRAIN_LINE_DIR" \
    --epochs "$EPOCHS" \
    --workers 0 \
    --model resnet \
    --gan \
    --resume-generator "$WARM_RESNET_CKPT" \
    --strict-resume \
    --lr 5e-5 --lr-d 1e-5 --pos-weight 3.0 \
    --bce-weight 0.60 --l1-weight 0.18 \
    --shape-weight 0.05 --ink-weight 0.04 \
    --adv-weight 0.015 \
    --no-autocontrast 2>&1 | tee -a "$LOG"
}

MODEL_ARGS=(
  --model raw_resnet_gan=results/model_resnet_binft_e3_resnet_gan_advsharp_binft
  --model cleaned_edge_unet_e2=results/cleaned_rough_model_survey_e2_edge_preserve_unet
  --model cleaned_mild_unet_e2=results/cleaned_rough_model_survey_e2_line_background_mild_unet
)
METRIC_MODELS=(
  model_resnet_binft_e3_resnet_gan_advsharp_binft
  cleaned_rough_model_survey_e2_edge_preserve_unet
  cleaned_rough_model_survey_e2_line_background_mild_unet
)
HAZE_ARGS=(
  raw_resnet_gan=results/model_resnet_binft_e3_resnet_gan_advsharp_binft
  cleaned_edge_unet_e2=results/cleaned_rough_model_survey_e2_edge_preserve_unet
  cleaned_mild_unet_e2=results/cleaned_rough_model_survey_e2_line_background_mild_unet
)

for mode in "${MODES[@]}"; do
  train_clean=results/${TAG}_rough_train_${mode}
  eval_clean=results/${TAG}_rough_eval_${mode}
  clean_split "$mode" "$TRAIN_LIST" "$TRAIN_ROUGH_DIR" "$train_clean"
  clean_split "$mode" "$EVAL_LIST" "$EVAL_ROUGH_DIR" "$eval_clean"

  train_unet "$mode" "$train_clean"
  infer_batch "${mode}_unet_e${EPOCHS}" "checkpoints/${TAG}_${mode}_unet_e${EPOCHS}/best.pth" "$eval_clean"

  train_resnet_gan_warm "$mode" "$train_clean"
  infer_batch "${mode}_resnet_gan_warm_e${EPOCHS}" "checkpoints/${TAG}_${mode}_resnet_gan_warm_e${EPOCHS}/best.pth" "$eval_clean"

  MODEL_ARGS+=(--model "${mode}_unet_e${EPOCHS}=results/${TAG}_${mode}_unet_e${EPOCHS}")
  MODEL_ARGS+=(--model "${mode}_resnet_gan_warm_e${EPOCHS}=results/${TAG}_${mode}_resnet_gan_warm_e${EPOCHS}")
  METRIC_MODELS+=("${TAG}_${mode}_unet_e${EPOCHS}")
  METRIC_MODELS+=("${TAG}_${mode}_resnet_gan_warm_e${EPOCHS}")
  HAZE_ARGS+=("${mode}_unet_e${EPOCHS}=results/${TAG}_${mode}_unet_e${EPOCHS}")
  HAZE_ARGS+=("${mode}_resnet_gan_warm_e${EPOCHS}=results/${TAG}_${mode}_resnet_gan_warm_e${EPOCHS}")
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
  echo "warm_resnet_ckpt=$WARM_RESNET_CKPT"
  echo "train_list=$TRAIN_LIST"
  echo "eval_list=$EVAL_LIST"
  echo "modes=${MODES[*]}"
  echo "metrics=$METRICS"
  echo "haze_metrics=$HAZE_METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

echo "done marker: $DONE" | tee -a "$LOG"

./experiments/send_autoloop_notification.sh \
  "Lineart cleaned rough warm-start survey complete" \
  "Review $MONTAGE, $METRICS, and $HAZE_METRICS" || true
