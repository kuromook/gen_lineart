#!/bin/bash
# Diagnose whether halo originates in atari generation or final refinement.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-atari_halo_diag_shuffle60}
COUNT=${COUNT:-60}
SEED=${SEED:-20260720}
POOL_LIST=${POOL_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
EXCLUDE_LIST=${EXCLUDE_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
AGREEMENT_CSV=${AGREEMENT_CSV:-results/agreement_halo_e2_agreement_scores.csv}

SAMPLE_LIST=dataset/pairs_480/${TAG}.txt
SAMPLE_META=results/${TAG}_samples.csv
LOG=logs/${TAG}.log
HALO_CSV=results/halo_metrics_${TAG}.csv
AMPLIFICATION_CSV=results/halo_amplification_${TAG}.csv
AMPLIFICATION_SUMMARY=results/halo_amplification_${TAG}_summary.csv
MONTAGE=results/compare_${TAG}.png
DONE=logs/${TAG}.done

RAW_ATARI=results/line_refiner_e2_atari_train
DOG_ATARI=results/halo_mitigation_e2_dog_train
LUCY_MILD_ATARI=results/lucy_mask_deep_e2_lucy_mild_train
LUCY_THIN_ATARI=results/lucy_mask_deep_e2_lucy_thin_train

CLEANUP_CKPT=checkpoints/line_refiner_msgan_e2_cleanup_msgan_fm/best.pth
DOG_CKPT=checkpoints/halo_mitigation_e2_dog_aux_msgan/best.pth
LUCY_MILD_CKPT=checkpoints/lucy_mask_deep_e2_lucy_mild_aux_msgan/best.pth
LUCY_THIN_CKPT=checkpoints/lucy_mask_deep_e2_lucy_thin_aux_msgan/best.pth

CLEANUP_OUT=results/${TAG}_cleanup_msgan
DOG_OUT=results/${TAG}_dog_final
LUCY_MILD_OUT=results/${TAG}_lucy_mild_final
LUCY_THIN_OUT=results/${TAG}_lucy_thin_final

mkdir -p logs results "$CLEANUP_OUT" "$DOG_OUT" "$LUCY_MILD_OUT" "$LUCY_THIN_OUT"
: > "$LOG"

echo "=== build shuffled diagnostic list ===" | tee -a "$LOG"
"$PY" tools/evaluation/build_atari_halo_diagnostic_set.py \
  --pool-list "$POOL_LIST" \
  --agreement-csv "$AGREEMENT_CSV" \
  --exclude-list "$EXCLUDE_LIST" \
  --count "$COUNT" \
  --seed "$SEED" \
  --output-list "$SAMPLE_LIST" \
  --output-csv "$SAMPLE_META" 2>&1 | tee -a "$LOG"

infer_pair() {
  local label=$1
  local ckpt=$2
  local aux_dir=$3
  local out_dir=$4
  echo "=== infer ${label} ===" | tee -a "$LOG"
  while read -r sample; do
    [[ -z "$sample" ]] && continue
    name="${sample%.jpg}"
    "$PY" scripts/inference_i2i.py \
      --checkpoint "$ckpt" \
      --input "dataset/pairs_480/train/rough/${name}.jpg" \
      --aux-input "$aux_dir/${name}_out.png" \
      --output "$out_dir/${name}_out.png" \
      --autocontrast 2>&1 | tee -a "$LOG"
  done < "$SAMPLE_LIST"
}

infer_pair cleanup_msgan "$CLEANUP_CKPT" "$RAW_ATARI" "$CLEANUP_OUT"
infer_pair dog_final "$DOG_CKPT" "$DOG_ATARI" "$DOG_OUT"
infer_pair lucy_mild_final "$LUCY_MILD_CKPT" "$LUCY_MILD_ATARI" "$LUCY_MILD_OUT"
infer_pair lucy_thin_final "$LUCY_THIN_CKPT" "$LUCY_THIN_ATARI" "$LUCY_THIN_OUT"

echo "=== halo metrics ===" | tee -a "$LOG"
"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --sample-list "$SAMPLE_LIST" \
  --split train \
  --models \
    raw_atari="$RAW_ATARI" \
    dog_atari="$DOG_ATARI" \
    lucy_mild_atari="$LUCY_MILD_ATARI" \
    lucy_thin_atari="$LUCY_THIN_ATARI" \
    cleanup_msgan="$CLEANUP_OUT" \
    dog_final="$DOG_OUT" \
    lucy_mild_final="$LUCY_MILD_OUT" \
    lucy_thin_final="$LUCY_THIN_OUT" \
  --output-csv "$HALO_CSV" 2>&1 | tee -a "$LOG"

echo "=== amplification metrics ===" | tee -a "$LOG"
"$PY" tools/evaluation/compare_halo_amplification.py \
  --halo-csv "$HALO_CSV" \
  --output-csv "$AMPLIFICATION_CSV" \
  --summary-csv "$AMPLIFICATION_SUMMARY" 2>&1 | tee -a "$LOG"

echo "=== montage ===" | tee -a "$LOG"
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$SAMPLE_LIST" \
  --split train \
  --model raw_atari="$RAW_ATARI" \
  --model cleanup_msgan="$CLEANUP_OUT" \
  --model dog_atari="$DOG_ATARI" \
  --model dog_final="$DOG_OUT" \
  --model lucy_mild_atari="$LUCY_MILD_ATARI" \
  --model lucy_mild_final="$LUCY_MILD_OUT" \
  --model lucy_thin_atari="$LUCY_THIN_ATARI" \
  --model lucy_thin_final="$LUCY_THIN_OUT" \
  --output "$MONTAGE" 2>&1 | tee -a "$LOG"

{
  echo "tag=$TAG"
  echo "sample_list=$SAMPLE_LIST"
  echo "sample_meta=$SAMPLE_META"
  echo "halo_metrics=$HALO_CSV"
  echo "amplification=$AMPLIFICATION_CSV"
  echo "amplification_summary=$AMPLIFICATION_SUMMARY"
  echo "montage=$MONTAGE"
} > "$DONE"

echo "done: $DONE" | tee -a "$LOG"
