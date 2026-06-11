#!/bin/bash
set -e
cd "$(dirname "$0")"

PY=./venv/bin/python
CKPT=checkpoints_shape1
RESULTS=results/shape1
SAMPLES="lineart_004_002 lineart_004_004 lineart_004_006 lineart_004_008 lineart_004_010 housei_002_19_12 housei_002_07_12 housei_002_06_15"

$PY train.py \
    --file-list dataset_480/valid_train_warm_regions.txt \
    --checkpoint-dir "$CKPT" \
    --resume checkpoints_warm_regions/best.pth \
    --lr 1e-5 \
    --epochs 20 \
    --pos-weight 2.0 \
    --edge-weight 0.0 \
    --shape-weight 1.0 \
    --ink-weight 2.0 \
    --autocontrast

mkdir -p "$RESULTS"
for NAME in $SAMPLES; do
    if [[ $NAME == housei* ]]; then
        INPUT="dataset_480/train/rough/${NAME}.jpg"
    else
        INPUT="dataset_480/test/rough/${NAME}.jpg"
    fi
    $PY inference.py \
        --checkpoint "$CKPT/best.pth" \
        --input "$INPUT" \
        --output "$RESULTS/${NAME}_out.png" \
        --autocontrast
done
