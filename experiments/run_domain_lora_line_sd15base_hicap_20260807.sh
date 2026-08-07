#!/bin/bash
# Capacity isolation test, layered on top of the base-checkpoint fix
# (2026-08-06/07 user direction).
#
# domain_lora_line_sd15base_20260806 (SD1.5 base, still rank16 attn-only
# LoRA) showed real progress on structural axes (blank_cell_fraction,
# grid_ink_cv, background_ratio all moved toward the real line_ref
# distribution) versus the AOM3A1B-base run, but visually settled into a
# DIFFERENT strong prior -- bold manga/comic finished-inking with heavy
# solid-black fill blocks (deep_black_ratio ~55x real, width_consistency
# actually worse than the AOM3A1B run) instead of the real tiles' thin,
# uniform single-pass linework. Read as: SD1.5 has a weaker
# illustration-finish prior than AOM3A1B, but rank-16 attention-only LoRA
# still doesn't have enough capacity to fully pull it into this project's
# specific line style.
#
# This run keeps the SD1.5 base and adds capacity on top: lora-rank 16 ->
# 32, and target modules extended from attention-only (to_k/to_q/to_v/
# to_out.0) to also include the UNet's conv/projection layers (conv1,
# conv2, conv_shortcut, proj_in, proj_out) -- attention-only LoRA can only
# reweight *what* the model attends to, not the actual conv-level stroke
# rendering, which is plausibly why width_consistency/deep_black_ratio
# didn't budge. ~7.8x more trainable params (24.7M vs 3.19M), verified via
# a 6-step smoke test (train + sample) before this launch. Same data
# (line_combined_koma_20260729, 1489 images), same caption, same epoch
# count (10) as every prior line-domain run for a clean comparison.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=domain_lora_line_sd15base_hicap_20260807
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-32}
LORA_TARGETS="to_k to_q to_v to_out.0 conv1 conv2 conv_shortcut proj_in proj_out"
CAPTION="lineartstyle, monochrome line art, clean linework, manga panel, black and white"
IMAGE_DIRS="dataset/pairs_480/train/line_combined_koma_20260729"
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] domain_lora line-domain sd15-base hicap start (epochs=$EPOCHS rank=$LORA_RANK targets=$LORA_TARGETS base=$BASE_CKPT)"

"$PY" scripts/train_domain_lora.py \
  --image-dirs $IMAGE_DIRS \
  --output-dir "$CKPT" \
  --caption "$CAPTION" \
  --lora-rank "$LORA_RANK" \
  --lora-target-modules $LORA_TARGETS \
  --epochs "$EPOCHS" \
  --base-ckpt "$BASE_CKPT"

echo "=== sample + contact sheet ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "$CAPTION" \
  --base-ckpt "$BASE_CKPT" \
  --tag "$TAG" \
  --num-samples 16

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "lora_rank=$LORA_RANK"
  echo "lora_target_modules=$LORA_TARGETS"
  echo "base_ckpt=$BASE_CKPT"
  echo "checkpoint=$CKPT/final"
  echo "contact_sheet=results/${TAG}/contact_sheet_${TAG}.png"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart domain LoRA (line domain, SD1.5 base, hi-cap rank32+conv) training complete" \
  "Review results/${TAG}/contact_sheet_${TAG}.png" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] domain_lora line-domain sd15-base hicap complete"
