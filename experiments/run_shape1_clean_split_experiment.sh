#!/bin/bash
set -e
cd "$(dirname "$0")/.."

PY=./venv/bin/python
CKPT=checkpoints/shape1_clean_split
RESULTS=results/shape1_clean_split
TRAIN_LIST=dataset/pairs_480/valid_train_warm_regions_clean_split.txt
EVAL_LIST=dataset/pairs_480/eval_fixed_clean.txt

if [[ ! -f "$TRAIN_LIST" || ! -f "$EVAL_LIST" ]]; then
    "$PY" tools/pair_extraction/build_clean_split_lists.py
fi

"$PY" tools/evaluation/audit_split_leakage.py \
    --train-list "$TRAIN_LIST" \
    --test-list dataset/pairs_480/valid_test.txt

"$PY" scripts/train.py \
    --file-list "$TRAIN_LIST" \
    --checkpoint-dir "$CKPT" \
    --epochs 200 \
    --lr 1e-4 \
    --pos-weight 2.0 \
    --edge-weight 0.0 \
    --shape-weight 1.0 \
    --ink-weight 2.0 \
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
    --models shape1_clean_split \
    --sample-list "$EVAL_LIST" \
    --split test \
    --output-csv results/fixed_output_metrics_clean_split.csv

"$PY" tools/compare/make_model_eval_compare.py \
    --sample-list "$EVAL_LIST" \
    --model-dir "$RESULTS" \
    --model-label shape1_clean_split \
    --split test \
    --output results/compare_shape1_clean_split.png
