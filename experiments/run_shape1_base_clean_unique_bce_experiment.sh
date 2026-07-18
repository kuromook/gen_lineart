#!/bin/bash
# Base-only leak-free and duplicate-free BCE/L1 baseline for MoE.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

PY=./venv/bin/python
CKPT=checkpoints/shape1_base_clean_unique_bce
RESULTS=results/shape1_base_clean_unique_bce
TRAIN_LIST=dataset/pairs_480/valid_train_base_clean_unique.txt
EVAL_LIST=dataset/pairs_480/eval_fixed_clean_lineart004.txt

if [[ ! -f "$TRAIN_LIST" ]]; then
  "$PY" tools/pair_extraction/build_clean_unique_lists.py
fi

"$PY" tools/evaluation/audit_pair_dataset_integrity.py \
  --train-lists "$TRAIN_LIST" \
  --eval-lists dataset/pairs_480/valid_test.txt "$EVAL_LIST" \
  --output-summary results/pair_dataset_integrity_summary_base_clean_unique.csv \
  --output-findings results/pair_dataset_integrity_findings_base_clean_unique.csv

"$PY" scripts/train.py \
  --file-list "$TRAIN_LIST" \
  --checkpoint-dir "$CKPT" \
  --epochs 200 \
  --lr 1e-4 \
  --pos-weight 5.0 \
  --edge-weight 0.0 \
  --shape-weight 0.0 \
  --ink-weight 0.0 \
  --autocontrast

mkdir -p "$RESULTS"
while read -r SAMPLE; do
  [[ -z "$SAMPLE" ]] && continue
  NAME="${SAMPLE%.jpg}"
  "$PY" scripts/inference.py \
    --checkpoint "$CKPT/best.pth" \
    --input "dataset/pairs_480/test/rough/${NAME}.jpg" \
    --output "$RESULTS/${NAME}_out.png" \
    --autocontrast
done < "$EVAL_LIST"

"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models shape1_base_clean_unique_bce \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv results/fixed_output_metrics_base_clean_unique_lineart004.csv

"$PY" tools/compare/make_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --model-dir "$RESULTS" \
  --model-label shape1_base_clean_unique_bce \
  --split test \
  --output results/compare_shape1_base_clean_unique_lineart004.png
