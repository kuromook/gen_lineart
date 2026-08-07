#!/bin/bash
# Rare-trigger-token retry, keeping domain-framing words this time
# (2026-08-07 user direction, after the first sks-trigger attempt
# regressed badly).
#
# domain_lora_line_sd15base_sks_20260807 replaced the ENTIRE style suffix
# ("lineartstyle, monochrome line art, clean linework, manga panel, black
# and white") with just "sks style, monochrome, black and white". Result:
# at scale1.0 the LoRA's weak influence let the base SD1.5 checkpoint's
# own strong prior for "1girl ... monochrome, black and white" take over
# -- which turned out to mean photorealistic B&W PORTRAIT PHOTOGRAPHY, not
# line art. At scale1.3/1.4 photo and manga-linework elements collided
# within the same image, worse than every prior variant. Deleted (see
# doc/work_log.md) -- this proved "manga panel, monochrome line art" were
# not just baggage from the base model's own habits, they were necessary
# DOMAIN framing keeping generation in line-art territory at all.
#
# This run keeps that domain framing and only swaps the specific
# style-execution descriptor ("lineartstyle, clean linework" -- the part
# most likely carrying pre-existing "generic clean vector-ish linework"
# bias) for the rare token: "sks style, monochrome line art, manga panel,
# black and white". Same rank16/attn-only/SD1.5-base/1489-image/per-image-
# WD14-tag setup as every line-domain run since 20260806.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=domain_lora_line_sd15base_sksv2_20260807
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
CAPTION="sks style, monochrome line art, manga panel, black and white"
CAPTION_CSV="results/domain_lora_line_captiontags_20260807/tags_sksv2.csv"
IMAGE_DIRS="dataset/pairs_480/train/line_combined_koma_20260729"
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] domain_lora line-domain sd15-base sksv2-trigger start (epochs=$EPOCHS rank=$LORA_RANK base=$BASE_CKPT captions=$CAPTION_CSV)"

"$PY" scripts/train_domain_lora.py \
  --image-dirs $IMAGE_DIRS \
  --output-dir "$CKPT" \
  --caption "$CAPTION" \
  --caption-csv "$CAPTION_CSV" \
  --lora-rank "$LORA_RANK" \
  --epochs "$EPOCHS" \
  --base-ckpt "$BASE_CKPT"

MOTIF_CAPTION="1girl, solo, close-up, white_background, simple_background, $CAPTION"

echo "=== sample + contact sheet (scale 1.0 baseline) ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "$MOTIF_CAPTION" \
  --base-ckpt "$BASE_CKPT" \
  --tag "$TAG" \
  --num-samples 16

echo "=== sample + contact sheet (scale 1.3) ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "$MOTIF_CAPTION" \
  --base-ckpt "$BASE_CKPT" \
  --lora-scale 1.3 \
  --tag "${TAG}_scale13" \
  --num-samples 16

echo "=== sample + contact sheet (scale 1.4) ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "$MOTIF_CAPTION" \
  --base-ckpt "$BASE_CKPT" \
  --lora-scale 1.4 \
  --tag "${TAG}_scale14" \
  --num-samples 16

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "lora_rank=$LORA_RANK"
  echo "caption_csv=$CAPTION_CSV"
  echo "base_ckpt=$BASE_CKPT"
  echo "checkpoint=$CKPT/final"
  echo "contact_sheet_scale10=results/${TAG}/contact_sheet_${TAG}.png"
  echo "contact_sheet_scale13=results/${TAG}_scale13/contact_sheet_${TAG}_scale13.png"
  echo "contact_sheet_scale14=results/${TAG}_scale14/contact_sheet_${TAG}_scale14.png"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart domain LoRA (line domain, SD1.5 base, sksv2 trigger token) training complete" \
  "Review results/${TAG}*/contact_sheet_*.png" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] domain_lora line-domain sd15-base sksv2-trigger complete"
