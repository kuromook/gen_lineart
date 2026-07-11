#!/bin/bash
# Conservative kurip VLM-accept fine-tune before moving on to MoE.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

PY=./venv/bin/python
FILE_LIST=dataset/pairs_480/valid_train_warm_regions_kurip_vlm_accept190.txt
LINE_DIR=dataset/pairs_480/train/line_kurip_vlm_accept190_mix_clean_t192_cc8
CKPT=checkpoints/kurip_vlm_accept190_noac_ink2
LOG=logs/train_kurip_vlm_accept190_noac_ink2.log

$PY tools/pair_extraction/make_kurip_vlm_accept_mix.py

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
