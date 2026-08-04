#!/bin/bash
# Direction 4 follow-up: unconditional LoRA fine-tune on the rough-sketch
# domain alone. Second of two runs (see run_domain_lora_line_20260804.sh
# for the line-domain counterpart and full rationale).
#
# Rough pool is much larger than the line pool (~21.5k vs ~1.5k images --
# dataset/pairs_480/train/rough is the shared cross-source rough pool,
# not scoped to combined_koma, plus the 2026-08-05-confirmed-clean
# dataset/unpaired_rough_candidates/*/rough), so a lower epoch count is
# used to keep total training time comparable in spirit -- this is a
# first-look domain-quality check, not a final convergence run.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=domain_lora_rough_20260804
EPOCHS=${EPOCHS:-4}
LORA_RANK=${LORA_RANK:-16}
CAPTION="roughsketchstyle, pencil rough sketch, messy sketchy construction lines, monochrome"
IMAGE_DIRS="dataset/pairs_480/train/rough dataset/unpaired_rough_candidates/ako5ver2/rough dataset/unpaired_rough_candidates/fitness/rough dataset/unpaired_rough_candidates/gakuen/rough dataset/unpaired_rough_candidates/hamlabi/rough dataset/unpaired_rough_candidates/housei/rough"

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] domain_lora rough-domain start (epochs=$EPOCHS rank=$LORA_RANK)"

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
  "Lineart domain LoRA (rough domain) training complete" \
  "Review results/${TAG}/contact_sheet_${TAG}.png" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] domain_lora rough-domain complete"
