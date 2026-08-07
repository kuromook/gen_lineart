#!/bin/bash
# Rare-trigger-token isolation test, keeping rank16/attn-only capacity and
# per-image WD14 content tags fixed (2026-08-07 user direction: dig into
# the LoRA weights themselves next -- either narrow the training data or
# retrain with a "meaningless" trigger token, after the LoRA-scale sweep
# showed style-fidelity vs. shape-coherence is a hard trade-off along a
# single knob, not something inference-time scale alone can resolve).
#
# Same setup as domain_lora_line_sd15base_captiontags_20260807 (SD1.5
# base, rank16, attention-only LoRA, 1489-image line_combined_koma_20260729
# pool, 10 epochs, per-image WD14 content tags) except the style suffix
# changes from ordinary English words with pre-existing CLIP associations
# ("lineartstyle, monochrome line art, clean linework, manga panel, black
# and white") to a DreamBooth-style rare token ("sks style, monochrome,
# black and white") -- "sks" is a real but visually-uncommon token (not
# decomposed into common meaningful subwords by the CLIP tokenizer) with
# no strong pre-existing stylistic association, so the LoRA has to build
# its meaning purely from our tiles instead of blending with whatever
# "clean linework" or "manga panel" already evoke in the base model.
# "monochrome"/"black and white" are kept since they're objective
# color-space facts, not stylistic descriptors carrying baked-in bias.
#
# Captions rebuilt from the already-tagged
# results/domain_lora_line_captiontags_20260807/tags.csv (no need to
# rerun the WD14 tagger) -- see tags_sks.csv.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=domain_lora_line_sd15base_sks_20260807
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
CAPTION="sks style, monochrome, black and white"
CAPTION_CSV="results/domain_lora_line_captiontags_20260807/tags_sks.csv"
IMAGE_DIRS="dataset/pairs_480/train/line_combined_koma_20260729"
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] domain_lora line-domain sd15-base sks-trigger start (epochs=$EPOCHS rank=$LORA_RANK base=$BASE_CKPT captions=$CAPTION_CSV)"

"$PY" scripts/train_domain_lora.py \
  --image-dirs $IMAGE_DIRS \
  --output-dir "$CKPT" \
  --caption "$CAPTION" \
  --caption-csv "$CAPTION_CSV" \
  --lora-rank "$LORA_RANK" \
  --epochs "$EPOCHS" \
  --base-ckpt "$BASE_CKPT"

echo "=== sample + contact sheet (scale 1.0 baseline) ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "1girl, solo, close-up, white_background, simple_background, $CAPTION" \
  --base-ckpt "$BASE_CKPT" \
  --tag "$TAG" \
  --num-samples 16

echo "=== sample + contact sheet (scale 1.3) ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "1girl, solo, close-up, white_background, simple_background, $CAPTION" \
  --base-ckpt "$BASE_CKPT" \
  --lora-scale 1.3 \
  --tag "${TAG}_scale13" \
  --num-samples 16

echo "=== sample + contact sheet (scale 1.4) ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "1girl, solo, close-up, white_background, simple_background, $CAPTION" \
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
  "Lineart domain LoRA (line domain, SD1.5 base, sks trigger token) training complete" \
  "Review results/${TAG}*/contact_sheet_*.png" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] domain_lora line-domain sd15-base sks-trigger complete"
