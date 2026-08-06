#!/bin/bash
# Direction 4 follow-up, data-composition isolation variant.
#
# The first rough-domain LoRA (run_domain_lora_rough_20260804.sh, tag
# domain_lora_rough_20260804) trained on the full ~21.5k-image cross-source
# rough pool for only 4 epochs and produced samples that read as generic
# "line-practice" hatching scribbles unrelated to the actual tile content,
# not genuine rough-sketch composition -- see doc/work_log.md ("diffusion"
# branch, 2026-08-05 entries). Two confounded variables in that run: (a)
# only 4 epochs (2.5x fewer per-image passes than the line domain's 10),
# and (b) the training pool itself is dominated (74%) by legacy
# ako/housei-prefixed tiles plus duplicate-heavy "*komadense" tiles (the
# same kind of redundant-crop composition this project already found hurt
# a different training in the 2026-08-01 direct-unet dense-28ep result).
#
# This run isolates (b): same rank, same caption, same target epoch count
# as the line-domain run (10), but trained ONLY on the 2026-08-05-confirmed
# -clean dataset/unpaired_rough_candidates/*/rough pool (2,014 images,
# balanced across all 5 sources, no legacy/dense duplication) instead of
# the full 21.5k legacy-dominated pool. Much cheaper than fixing (a) alone
# on the full pool (~2,520 steps here vs ~16,000 additional steps to reach
# 10 epochs on the full 21.5k pool).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=domain_lora_roughclean_20260805
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
CAPTION="roughsketchstyle, pencil rough sketch, messy sketchy construction lines, monochrome"
IMAGE_DIRS="dataset/unpaired_rough_candidates/ako5ver2/rough dataset/unpaired_rough_candidates/fitness/rough dataset/unpaired_rough_candidates/gakuen/rough dataset/unpaired_rough_candidates/hamlabi/rough dataset/unpaired_rough_candidates/housei/rough"

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] domain_lora roughclean start (epochs=$EPOCHS rank=$LORA_RANK)"

"$PY" scripts/train_domain_lora.py \
  --image-dirs $IMAGE_DIRS \
  --output-dir "$CKPT" \
  --caption "$CAPTION" \
  --lora-rank "$LORA_RANK" \
  --epochs "$EPOCHS"

echo "=== sample + contact sheet ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "$CAPTION" \
  --tag "$TAG" \
  --num-samples 16

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "lora_rank=$LORA_RANK"
  echo "checkpoint=$CKPT/final"
  echo "contact_sheet=results/${TAG}/contact_sheet_${TAG}.png"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart domain LoRA (rough-clean subset) training complete" \
  "Review results/${TAG}/contact_sheet_${TAG}.png" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] domain_lora roughclean complete"
