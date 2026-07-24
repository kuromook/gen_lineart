#!/bin/bash
# Materialize current expert outputs on feature-stratified train probes and run oracle analysis.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-./venv/bin/python}
TAG=${1:-router_feature_probe_current_no_ako5_badrough}
FEATURE_CSV=${FEATURE_CSV:-results/pair_feature_scan_current_no_ako5_badrough.csv}
COUNT_PER_GROUP=${COUNT_PER_GROUP:-12}

LIST_DIR=dataset/pairs_480/${TAG}_lists
COMBINED_LIST=${LIST_DIR}/${TAG}_combined.txt
LABELS_CSV=${LIST_DIR}/${TAG}_labels.csv
LOG=logs/${TAG}.log
DONE=logs/${TAG}.done

ROUGH_DIR=dataset/pairs_480/train/rough

ATARI_CKPT=checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth
ATARI_DIR=results/${TAG}_atari

EDGE_AUX_DIR=results/${TAG}_rough_edge_preserve
DOG_AUX_DIR=results/${TAG}_dog_aux
LUCY_MILD_AUX_DIR=results/${TAG}_lucy_mild_aux
LUCY_THIN_AUX_DIR=results/${TAG}_lucy_thin_aux
FLOWDOG_AUX_DIR=results/${TAG}_flowdog_aux

ORACLE_DIR=results/${TAG}_oracle
ORACLE_CSV=results/${TAG}_oracle_expert_metrics.csv
ORACLE_SUMMARY=results/${TAG}_oracle_summary.json
CHOICES_CSV=results/${TAG}_oracle_choices_with_features.csv
FEATURE_SUMMARY=results/${TAG}_oracle_feature_summary.csv
MONTAGE=results/compare_${TAG}_oracle.png

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] router feature probe start tag=${TAG}"

echo "=== build feature probe lists ==="
"$PY" tools/evaluation/build_router_feature_probe_lists.py \
  --feature-csv "$FEATURE_CSV" \
  --output-dir "$LIST_DIR" \
  --tag "$TAG" \
  --count-per-group "$COUNT_PER_GROUP"

echo "=== materialize base atari ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$ATARI_CKPT" \
  --file-list "$COMBINED_LIST" \
  --rough-dir "$ROUGH_DIR" \
  --output-dir "$ATARI_DIR" \
  --autocontrast

echo "=== materialize aux variants ==="
"$PY" tools/preprocess/clean_rough_input.py \
  --file-list "$COMBINED_LIST" \
  --input-dir "$ROUGH_DIR" \
  --output-dir "$EDGE_AUX_DIR" \
  --mode edge_preserve
"$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_DIR" --output-dir "$DOG_AUX_DIR" --mode dog
"$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_DIR" --output-dir "$LUCY_MILD_AUX_DIR" --mode lucy_mild
"$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_DIR" --output-dir "$LUCY_THIN_AUX_DIR" --mode lucy_thin
"$PY" scripts/preprocess_atari_aux.py --input-dir "$ATARI_DIR" --output-dir "$FLOWDOG_AUX_DIR" --mode flowdog

infer_expert() {
  local label=$1
  local ckpt=$2
  local aux_dir=${3:-}
  local autocontrast_flag=${4:-}
  local out_dir=results/${TAG}_${label}
  echo "=== infer ${label} ==="
  if [[ -n "$aux_dir" ]]; then
    "$PY" scripts/inference_i2i_batch.py \
      --checkpoint "$ckpt" \
      --file-list "$COMBINED_LIST" \
      --rough-dir "$ROUGH_DIR" \
      --aux-dir "$aux_dir" \
      --output-dir "$out_dir" \
      $autocontrast_flag
  else
    "$PY" scripts/inference_i2i_batch.py \
      --checkpoint "$ckpt" \
      --file-list "$COMBINED_LIST" \
      --rough-dir "$ROUGH_DIR" \
      --output-dir "$out_dir" \
      $autocontrast_flag
  fi
}

infer_expert edge_bghaze checkpoints/2ch_haze_control_e3_edge_preserve_2ch_bghaze06_e3/best.pth "$EDGE_AUX_DIR"
infer_expert bin12 checkpoints/line_refiner_tight_e2_refiner_unet_bin12_ink14/best.pth "$ATARI_DIR" --autocontrast
infer_expert bin20 checkpoints/line_refiner_tight_e2_refiner_unet_bin20_ink16/best.pth "$ATARI_DIR" --autocontrast
infer_expert dog checkpoints/halo_mitigation_e2_dog_aux_msgan/best.pth "$DOG_AUX_DIR" --autocontrast
infer_expert lucy_mild checkpoints/lucy_mask_deep_e2_lucy_mild_aux_msgan/best.pth "$LUCY_MILD_AUX_DIR" --autocontrast
infer_expert lucy_thin checkpoints/lucy_mask_deep_e2_lucy_thin_aux_msgan/best.pth "$LUCY_THIN_AUX_DIR" --autocontrast
infer_expert flowdog checkpoints/halo_filter_flowmask_e2_flowdog_aux_msgan/best.pth "$FLOWDOG_AUX_DIR" --autocontrast

echo "=== oracle ==="
"$PY" scripts/evaluate_oracle_moe.py \
  --sample-list "$COMBINED_LIST" \
  --split train \
  --oracle-dir "$ORACLE_DIR" \
  --output-csv "$ORACLE_CSV" \
  --summary-json "$ORACLE_SUMMARY" \
  --score-mode balanced \
  --model edge_bghaze=results/${TAG}_edge_bghaze \
  --model bin12=results/${TAG}_bin12 \
  --model bin20=results/${TAG}_bin20 \
  --model dog=results/${TAG}_dog \
  --model lucy_mild=results/${TAG}_lucy_mild \
  --model lucy_thin=results/${TAG}_lucy_thin \
  --model flowdog=results/${TAG}_flowdog

"$PY" tools/evaluation/analyze_router_oracle_features.py \
  --labels-csv "$LABELS_CSV" \
  --oracle-summary "$ORACLE_SUMMARY" \
  --expert-metrics-csv "$ORACLE_CSV" \
  --choice-csv "$CHOICES_CSV" \
  --output-csv "$FEATURE_SUMMARY"

"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$COMBINED_LIST" \
  --split train \
  --model rough_atari="$ATARI_DIR" \
  --model edge_bghaze=results/${TAG}_edge_bghaze \
  --model bin12=results/${TAG}_bin12 \
  --model dog=results/${TAG}_dog \
  --model lucy_mild=results/${TAG}_lucy_mild \
  --model lucy_thin=results/${TAG}_lucy_thin \
  --model oracle="$ORACLE_DIR" \
  --output "$MONTAGE"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "count_per_group=$COUNT_PER_GROUP"
  echo "combined_list=$COMBINED_LIST"
  echo "labels_csv=$LABELS_CSV"
  echo "oracle_summary=$ORACLE_SUMMARY"
  echo "oracle_feature_summary=$FEATURE_SUMMARY"
  echo "montage=$MONTAGE"
} > "$DONE"

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] router feature probe complete"
