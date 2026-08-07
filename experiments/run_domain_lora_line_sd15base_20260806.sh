#!/bin/bash
# Base-checkpoint isolation test (2026-08-06 user direction).
#
# Both domain-only LoRAs so far (line_20260804, rough variants on
# 20260805) were trained on top of AOM3A1B_orangemixs, a heavily-merged
# anime-illustration checkpoint whose own strong prior (finished, shaded,
# colored art) plausibly fights a rank-16 LoRA's ability to actually pull
# the model into our specific rough/line domain -- user's visual read
# across all runs so far: output resembles the base model's own drawing
# habits with only a faint style nudge from our tiles, not the tiles
# themselves (style or line quality). Two candidate root causes were
# discussed (LoRA capacity too low vs. base checkpoint fighting the
# target domain); this run isolates the second, cheaper to test first.
#
# Identical to run_domain_lora_line_20260804.sh (same curated 1489-image
# line_combined_koma_20260729 pool, same caption, same rank, same epoch
# count) except --base-ckpt points at vanilla SD1.5 (v1-5-pruned-emaonly)
# instead of AOM3A1B -- a genuinely neutral base with no anime-illustration
# merge bias, at the cost of weaker built-in anime/manga concept
# knowledge. Only this one variable changes, so any visual/profile-tool
# difference from domain_lora_line_20260804 is attributable to the base
# checkpoint choice.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=domain_lora_line_sd15base_20260806
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
CAPTION="lineartstyle, monochrome line art, clean linework, manga panel, black and white"
IMAGE_DIRS="dataset/pairs_480/train/line_combined_koma_20260729"
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] domain_lora line-domain sd15-base start (epochs=$EPOCHS rank=$LORA_RANK base=$BASE_CKPT)"

"$PY" scripts/train_domain_lora.py \
  --image-dirs $IMAGE_DIRS \
  --output-dir "$CKPT" \
  --caption "$CAPTION" \
  --lora-rank "$LORA_RANK" \
  --epochs "$EPOCHS" \
  --base-ckpt "$BASE_CKPT"

echo "=== sample + contact sheet ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "$CAPTION" \
  --tag "$TAG" \
  --base-ckpt "$BASE_CKPT" \
  --num-samples 16

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "lora_rank=$LORA_RANK"
  echo "base_ckpt=$BASE_CKPT"
  echo "checkpoint=$CKPT/final"
  echo "contact_sheet=results/${TAG}/contact_sheet_${TAG}.png"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart domain LoRA (line domain, SD1.5 base) training complete" \
  "Review results/${TAG}/contact_sheet_${TAG}.png" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] domain_lora line-domain sd15-base complete"
