#!/bin/bash
# Compare 2ch raw+clean controls for white-collapse and background haze.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-2ch_haze_control_e3}
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

train_2ch() {
  local label=$1
  local clean_dir=$2
  shift 2
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
    --no-autocontrast \
    "$@" 2>&1 | tee -a "$LOG"
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

postprocess_output() {
  local label=$1
  local mode=$2
  local out_label=${label}_post_${mode}
  echo "=== postprocess ${out_label} ===" | tee -a "$LOG"
  "$PY" tools/compare/postprocess_line_outputs.py \
    --input-dir "results/${TAG}_${label}" \
    --output-dir "results/${TAG}_${out_label}" \
    --mode "$mode" 2>&1 | tee -a "$LOG"
}

MODEL_ARGS=(
  --model clean-edge=results/cleaned_rough_model_survey_e2_edge_preserve_unet
  --model clean-mild=results/cleaned_rough_model_survey_e2_line_background_mild_unet
  --model prev-edge2=results/raw_clean_mixed_2ch_e3_edge_preserve_raw_clean_2ch_unet_e3
  --model prev-mild2=results/raw_clean_mixed_2ch_e3_line_background_mild_raw_clean_2ch_unet_e3
)
METRIC_MODELS=(
  cleaned_rough_model_survey_e2_edge_preserve_unet
  cleaned_rough_model_survey_e2_line_background_mild_unet
  raw_clean_mixed_2ch_e3_edge_preserve_raw_clean_2ch_unet_e3
  raw_clean_mixed_2ch_e3_line_background_mild_raw_clean_2ch_unet_e3
)
HAZE_ARGS=(
  cleaned_edge_unet_e2=results/cleaned_rough_model_survey_e2_edge_preserve_unet
  cleaned_mild_unet_e2=results/cleaned_rough_model_survey_e2_line_background_mild_unet
  prev_edge_2ch=results/raw_clean_mixed_2ch_e3_edge_preserve_raw_clean_2ch_unet_e3
  prev_mild_2ch=results/raw_clean_mixed_2ch_e3_line_background_mild_raw_clean_2ch_unet_e3
)

add_result() {
  local label=$1
  local display=${2:-$label}
  MODEL_ARGS+=(--model "${display}=results/${TAG}_${label}")
  METRIC_MODELS+=("${TAG}_${label}")
  HAZE_ARGS+=("${label}=results/${TAG}_${label}")
}

for mode in "${MODES[@]}"; do
  train_clean=results/${TAG}_rough_train_${mode}
  eval_clean=results/${TAG}_rough_eval_${mode}
  clean_split "$mode" "$TRAIN_LIST" "$TRAIN_RAW_ROUGH" "$train_clean"
  clean_split "$mode" "$EVAL_LIST" "$EVAL_RAW_ROUGH" "$eval_clean"

  auxdrop_label=${mode}_2ch_auxdrop50_scale35_e${EPOCHS}
  haze_label=${mode}_2ch_bghaze06_e${EPOCHS}
  both_label=${mode}_2ch_auxdrop_bghaze_e${EPOCHS}

  train_2ch "$auxdrop_label" "$train_clean" \
    --aux-dropout 0.50 --aux-scale-min 0.35
  infer_2ch "$auxdrop_label" "$eval_clean"
  add_result "$auxdrop_label" "${mode}-auxdrop"

  train_2ch "$haze_label" "$train_clean" \
    --background-haze-weight 0.06 --background-haze-radius 9
  infer_2ch "$haze_label" "$eval_clean"
  add_result "$haze_label" "${mode}-bghaze"

  train_2ch "$both_label" "$train_clean" \
    --aux-dropout 0.50 --aux-scale-min 0.35 \
    --background-haze-weight 0.06 --background-haze-radius 9
  infer_2ch "$both_label" "$eval_clean"
  add_result "$both_label" "${mode}-both"

  for label in "$haze_label" "$both_label"; do
    for pp_mode in threshold35 unsharp_curve; do
      postprocess_output "$label" "$pp_mode"
      add_result "${label}_post_${pp_mode}" "${mode}-${label#${mode}_2ch_}-${pp_mode}"
    done
  done
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
  echo "variants=auxdrop50_scale35,bghaze06,auxdrop_bghaze"
  echo "postprocess=threshold35,unsharp_curve on haze variants"
  echo "metrics=$METRICS"
  echo "haze_metrics=$HAZE_METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

echo "done marker: $DONE" | tee -a "$LOG"

./experiments/send_autoloop_notification.sh \
  "Lineart 2ch haze-control survey complete" \
  "Review $MONTAGE, $METRICS, and $HAZE_METRICS" || true
