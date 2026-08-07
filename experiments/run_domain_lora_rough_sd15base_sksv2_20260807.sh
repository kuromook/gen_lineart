#!/bin/bash
# Apply the full line-domain isolation-chain outcome to the rough domain
# in one shot (2026-08-07 user direction) -- base checkpoint and capacity
# were already resolved generically (SD1.5 rank16 attn-only beat AOM3A1B
# and rank32+conv on the line domain; not domain-specific, carried over
# rather than re-isolated), so this run combines the remaining four
# techniques directly: per-image WD14 captions, motif-prompt sampling,
# LoRA inference-time scale sweep, and a domain-word-preserving rare
# trigger token ("sks style").
#
# Data: the 2,014-image CLEAN rough pool (dataset/unpaired_rough_
# candidates/*/rough, balanced across all 5 sources, no legacy/dense
# duplication) -- this was already confirmed the better data composition
# for the rough domain (domain_lora_roughclean_20260805 beat the full
# 21,468-image legacy-heavy pool on 7/8 stroke-level profile axes, see
# doc/work_log.md 2026-08-06 entry), carried forward rather than
# re-isolated here.
#
# Caption: per-image WD14 tags (results/domain_lora_rough_
# captiontags_20260807/tags.csv) + "sks style, pencil rough sketch,
# monochrome" -- domain-framing words ("pencil rough sketch, monochrome")
# kept per the line-domain lesson that stripping them entirely caused a
# photorealistic-photo regression; only the style-execution descriptor
# ("roughsketchstyle, messy sketchy construction lines") is replaced by
# the rare token.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=domain_lora_rough_sd15base_sksv2_20260807
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
CAPTION="sks style, pencil rough sketch, monochrome"
CAPTION_CSV="results/domain_lora_rough_captiontags_20260807/tags.csv"
IMAGE_DIRS="dataset/unpaired_rough_candidates/ako5ver2/rough dataset/unpaired_rough_candidates/fitness/rough dataset/unpaired_rough_candidates/gakuen/rough dataset/unpaired_rough_candidates/hamlabi/rough dataset/unpaired_rough_candidates/housei/rough"
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] domain_lora rough-domain sd15-base sksv2-trigger start (epochs=$EPOCHS rank=$LORA_RANK base=$BASE_CKPT captions=$CAPTION_CSV)"

"$PY" scripts/train_domain_lora.py \
  --image-dirs $IMAGE_DIRS \
  --output-dir "$CKPT" \
  --caption "$CAPTION" \
  --caption-csv "$CAPTION_CSV" \
  --lora-rank "$LORA_RANK" \
  --epochs "$EPOCHS" \
  --base-ckpt "$BASE_CKPT"

MOTIF_CAPTION="1girl, solo, close-up, sketch, $CAPTION"

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
  "Lineart domain LoRA (rough domain, SD1.5 base, sksv2 trigger token) training complete" \
  "Review results/${TAG}*/contact_sheet_*.png" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] domain_lora rough-domain sd15-base sksv2-trigger complete"
