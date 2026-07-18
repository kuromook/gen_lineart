#!/bin/bash
# Fine-tune the s960 kurip refined specialist from the clean split baseline.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

PY=./venv/bin/python
BASE_CKPT=checkpoints/shape1_clean_split/best.pth
BASE_LIST=dataset/pairs_480/valid_train_warm_regions_clean_split.txt
FILE_LIST=dataset/pairs_480/valid_train_warm_regions_clean_split_kurip_vlm_accept_refined_s960_123.txt
LINE_DIR=dataset/pairs_480/train/line_kurip_vlm_accept_refined_s960_123_cleanbase_mix_clean_t192_cc8
CKPT=checkpoints/kurip_vlm_refined_s960_123_cleanbase_noac_ink2
LOG=logs/train_kurip_vlm_refined_s960_123_cleanbase_noac_ink2.log

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "missing clean baseline checkpoint: $BASE_CKPT" >&2
  exit 1
fi

if [[ ! -f "$BASE_LIST" ]]; then
  "$PY" tools/pair_extraction/build_clean_split_lists.py
fi

"$PY" tools/pair_extraction/make_kurip_vlm_accept_mix.py \
  --base-list "$BASE_LIST" \
  --kurip-list dataset/pairs_480/valid_train_kurip_vlm_accept_refined_s960.txt \
  --kurip-line-dir dataset/pairs_480/train/line_kurip_vlm_accept_refined_s960_clean_t192_cc8 \
  --output-list "$FILE_LIST" \
  --output-kurip-list dataset/pairs_480/valid_train_kurip_vlm_accept_refined_s960_123_cleanbase.txt \
  --output-line-dir "$LINE_DIR" \
  --kurip-count 123

"$PY" scripts/train.py \
  --file-list "$FILE_LIST" \
  --rough-dir dataset/pairs_480/train/rough \
  --line-dir "$LINE_DIR" \
  --checkpoint-dir "$CKPT" \
  --resume "$BASE_CKPT" \
  --epochs 10 \
  --lr 5e-5 \
  --pos-weight 2.0 \
  --edge-weight 0.0 \
  --shape-weight 1.0 \
  --ink-weight 2.0 \
  --no-autocontrast \
  2>&1 | tee "$LOG"
