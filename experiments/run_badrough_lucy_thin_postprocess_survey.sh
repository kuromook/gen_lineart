#!/bin/bash
# Postprocess calibration survey for the best relaxed lucy_thin candidate.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-./venv/bin/python}
TAG=${1:-badrough_lucy_thin_postprocess}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
SOURCE_LABEL=${SOURCE_LABEL:-badrough_lucy_thin_relaxed_e3_lucy_thin_relaxed_b}
SOURCE_DIR=${SOURCE_DIR:-results/${SOURCE_LABEL}}

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
HAZE_METRICS=results/haze_uncertainty_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

MODES=(threshold48 threshold50 threshold52 threshold54 threshold55 unsharp_curve)

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] badrough lucy_thin postprocess survey start"
echo "tag=$TAG source=$SOURCE_DIR eval_list=$EVAL_LIST"

MODEL_ARGS=(
  --model old_lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan
  --model clean_lucy_thin=results/badrough_retrain_e3_lucy_thin_aux_msgan
  --model relaxed_b="$SOURCE_DIR"
)
METRIC_MODELS=(
  lucy_mask_deep_e2_lucy_thin_aux_msgan
  badrough_retrain_e3_lucy_thin_aux_msgan
  "$SOURCE_LABEL"
)
HAZE_ARGS=(
  old_lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan
  clean_lucy_thin=results/badrough_retrain_e3_lucy_thin_aux_msgan
  relaxed_b="$SOURCE_DIR"
)

for mode in "${MODES[@]}"; do
  label=${SOURCE_LABEL}_post_${mode}
  out_dir=results/${label}
  echo "=== postprocess ${mode} -> ${out_dir} ==="
  "$PY" tools/compare/postprocess_line_outputs.py \
    --input-dir "$SOURCE_DIR" \
    --output-dir "$out_dir" \
    --mode "$mode"
  MODEL_ARGS+=(--model "${mode}=results/${label}")
  METRIC_MODELS+=("$label")
  HAZE_ARGS+=("${mode}=results/${label}")
done

echo "=== montage ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  "${MODEL_ARGS[@]}" \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models "${METRIC_MODELS[@]}" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

echo "=== haze metrics ==="
"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --models "${HAZE_ARGS[@]}" \
  --output-csv "$HAZE_METRICS"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "source_label=$SOURCE_LABEL"
  echo "source_dir=$SOURCE_DIR"
  echo "modes=${MODES[*]}"
  echo "montage=$MONTAGE"
  echo "metrics=$METRICS"
  echo "haze_metrics=$HAZE_METRICS"
} > "$DONE"

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] badrough lucy_thin postprocess survey complete"
