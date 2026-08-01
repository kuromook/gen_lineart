#!/bin/bash
# Warmstart control run for the native-resolution stroke-scale-filtered
# ako5ver2 keep281 tiles (see doc/preprocess/raw_dataset_extraction_knowledge.md and
# doc/preprocess/region_dataset_extraction_policy.md for the native re-materialization
# and tile-score filter design).
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

PY=./venv/bin/python
FILE_LIST=dataset/pairs_480/valid_train_ako5ver2_native_strict_cut25_20260725.txt
ROUGH_DIR=dataset/pairs_480/train/rough
LINE_DIR=dataset/pairs_480/train/line_ako5ver2_native_strict_cut25_20260725
BASE_CKPT=checkpoints/shape1_clean_split_bce/best.pth
CKPT=checkpoints/ako5ver2_native_strict_cut25_480_warm_clean_bce_e10
LOG=logs/train_ako5ver2_native_strict_cut25_480_warm_clean_bce_e10.log
OUT_DIR=results/ako5ver2_native_strict_cut25_480_warm_clean_bce_e10_outputs

$PY scripts/train_i2i_survey.py \
  --model unet \
  --file-list "$FILE_LIST" \
  --rough-dir "$ROUGH_DIR" \
  --line-dir "$LINE_DIR" \
  --checkpoint-dir "$CKPT" \
  --resume-generator "$BASE_CKPT" \
  --strict-resume \
  --image-size 480 \
  --epochs 10 \
  --batch-size 2 \
  --workers 0 \
  --lr 1e-5 \
  --pos-weight 3.0 \
  --bce-weight 0.6 \
  --l1-weight 0.2 \
  --shape-weight 0.05 \
  --ink-weight 0.02 \
  --save-every 10 \
  --require-cuda \
  2>&1 | tee "$LOG"

$PY scripts/inference_i2i_batch.py \
  --checkpoint "$CKPT/best.pth" \
  --file-list "$FILE_LIST" \
  --rough-dir "$ROUGH_DIR" \
  --output-dir "$OUT_DIR" \
  --autocontrast

echo "saved outputs: $OUT_DIR"
