#!/bin/bash
# Fine-tune with scaled 960px source crops resized to 480px model inputs.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

PY=./venv/bin/python
FILE_LIST=dataset/pairs_480/valid_train_warm_regions_kurip_vlm_accept_refined_s960_123.txt
LINE_DIR=dataset/pairs_480/train/line_kurip_vlm_accept_refined_s960_123_mix_clean_t192_cc8
CKPT=checkpoints/kurip_vlm_refined_s960_123_noac_ink2
LOG=logs/train_kurip_vlm_refined_s960_123_noac_ink2.log

$PY tools/pair_extraction/make_kurip_vlm_accept_mix.py \
  --kurip-list dataset/pairs_480/valid_train_kurip_vlm_accept_refined_s960.txt \
  --kurip-line-dir dataset/pairs_480/train/line_kurip_vlm_accept_refined_s960_clean_t192_cc8 \
  --output-list "$FILE_LIST" \
  --output-kurip-list dataset/pairs_480/valid_train_kurip_vlm_accept_refined_s960_123.txt \
  --output-line-dir "$LINE_DIR" \
  --kurip-count 123

$PY scripts/train.py \
  --file-list "$FILE_LIST" \
  --rough-dir dataset/pairs_480/train/rough \
  --line-dir "$LINE_DIR" \
  --checkpoint-dir "$CKPT" \
  --resume checkpoints/shape1/best.pth \
  --epochs 10 \
  --lr 5e-5 \
  --pos-weight 2.0 \
  --edge-weight 0.0 \
  --shape-weight 1.0 \
  --ink-weight 2.0 \
  --no-autocontrast \
  2>&1 | tee "$LOG"
