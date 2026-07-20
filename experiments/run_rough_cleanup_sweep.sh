#!/bin/bash
# Sweep rough cleansing modes before committing to cleaned-rough atari training.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-rough_cleanup_sweep200}
COUNT=${COUNT:-200}
SEED=${SEED:-20260721}
POOL_LIST=${POOL_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
EXCLUDE_LIST=${EXCLUDE_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
AGREEMENT_CSV=${AGREEMENT_CSV:-results/agreement_halo_e2_agreement_scores.csv}
ROUGH_DIR=${ROUGH_DIR:-dataset/pairs_480/train/rough}

SAMPLE_LIST=dataset/pairs_480/${TAG}.txt
SAMPLE_META=results/${TAG}_samples.csv
LOG=logs/${TAG}.log
METRICS=results/haze_uncertainty_metrics_${TAG}.csv
MONTAGE=results/compare_${TAG}.png
DONE=logs/${TAG}.done

MODES=(
  identity
  background_mild
  background
  background_strong
  line_mild
  line
  line_strong
  line_background_mild
  line_background
  line_background_strong
  softcut_only
  edge_preserve
)

mkdir -p logs results
: > "$LOG"

echo "=== build sweep sample list ===" | tee -a "$LOG"
"$PY" tools/evaluation/build_atari_halo_diagnostic_set.py \
  --pool-list "$POOL_LIST" \
  --agreement-csv "$AGREEMENT_CSV" \
  --exclude-list "$EXCLUDE_LIST" \
  --count "$COUNT" \
  --seed "$SEED" \
  --output-list "$SAMPLE_LIST" \
  --output-csv "$SAMPLE_META" 2>&1 | tee -a "$LOG"

MODEL_ARGS=()
METRIC_ARGS=()

for mode in "${MODES[@]}"; do
  out_dir=results/${TAG}_${mode}
  echo "=== clean ${mode} ===" | tee -a "$LOG"
  "$PY" tools/preprocess/clean_rough_input.py \
    --file-list "$SAMPLE_LIST" \
    --input-dir "$ROUGH_DIR" \
    --output-dir "$out_dir" \
    --mode "$mode" 2>&1 | tee -a "$LOG"
  MODEL_ARGS+=(--model "${mode}=${out_dir}")
  METRIC_ARGS+=("${mode}=${out_dir}")
done

echo "=== split haze/uncertainty metrics ===" | tee -a "$LOG"
"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --sample-list "$SAMPLE_LIST" \
  --split train \
  --models "${METRIC_ARGS[@]}" \
  --output-csv "$METRICS" 2>&1 | tee -a "$LOG"

echo "=== montage ===" | tee -a "$LOG"
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$SAMPLE_LIST" \
  --split train \
  "${MODEL_ARGS[@]}" \
  --output "$MONTAGE" 2>&1 | tee -a "$LOG"

{
  echo "tag=$TAG"
  echo "sample_list=$SAMPLE_LIST"
  echo "sample_meta=$SAMPLE_META"
  echo "metrics=$METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

echo "done: $DONE" | tee -a "$LOG"
