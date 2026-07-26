#!/bin/bash
# Warmstart ako5ver2 masked keep281 from the clean-split BCE U-Net baseline.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

PY=./venv/bin/python
MANIFEST=dataset/regions_ako5ver2_varregion_20260725_postalign12_masked_line_conservative/manifest_user_review_keep281.csv
BASE_CKPT=checkpoints/shape1_clean_split_bce/best.pth
CKPT=checkpoints/ako5ver2_varregion_keep281_masked_unet480_warm_clean_bce_e10
LOG=logs/train_ako5ver2_varregion_keep281_masked_unet480_warm_clean_bce_e10.log
OUT_DIR=results/ako5ver2_varregion_keep281_masked_unet480_warm_clean_bce_e10_outputs
MONTAGE=results/ako5ver2_varregion_keep281_masked_unet480_warm_clean_bce_e10_compare.png

$PY scripts/train_i2i_survey.py \
  --model unet \
  --region-manifest "$MANIFEST" \
  --region-fit-mode square_pad \
  --region-rough-key masked_rough_path \
  --region-line-key masked_line_path \
  --region-mask-key valid_mask_path \
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

$PY tools/compare/make_region_manifest_compare.py \
  --checkpoint "$CKPT/best.pth" \
  --manifest "$MANIFEST" \
  --output-dir "$OUT_DIR" \
  --montage "$MONTAGE" \
  --image-size 480 \
  --fit-mode square_pad \
  --rough-key masked_rough_path \
  --line-key masked_line_path \
  --limit 12 \
  --autocontrast \
  --require-cuda

echo "saved comparison: $MONTAGE"
