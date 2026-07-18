#!/bin/bash
# Base-only clean split baseline: excludes ako5r region pairs from warm_regions.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

PY=./venv/bin/python
CKPT=checkpoints/shape1_base_clean_split_bce
RESULTS=results/shape1_base_clean_split_bce
TRAIN_LIST=dataset/pairs_480/valid_train_base_clean_split.txt
EVAL_LIST=dataset/pairs_480/eval_fixed_clean_lineart004.txt

if [[ ! -f "$TRAIN_LIST" ]]; then
  "$PY" - <<'PY'
from pathlib import Path
src = Path("dataset/pairs_480/valid_train_warm_regions_clean_split.txt")
out = Path("dataset/pairs_480/valid_train_base_clean_split.txt")
names = [line.strip() for line in src.read_text().splitlines() if line.strip()]
base = [name for name in names if not name.startswith("ako5r_")]
out.write_text("".join(f"{name}\n" for name in base))
PY
fi

"$PY" tools/evaluation/audit_split_leakage.py \
    --train-list "$TRAIN_LIST" \
    --test-list dataset/pairs_480/valid_test.txt

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
    --models shape1_base_clean_split_bce \
    --sample-list "$EVAL_LIST" \
    --split test \
    --output-csv results/fixed_output_metrics_base_clean_lineart004.csv

"$PY" tools/compare/make_model_eval_compare.py \
    --sample-list "$EVAL_LIST" \
    --model-dir "$RESULTS" \
    --model-label shape1_base_clean_split_bce \
    --split test \
    --output results/compare_shape1_base_clean_lineart004.png
