#!/bin/bash
# Direction 4 follow-up: unconditional LoRA fine-tune on the line-art
# domain alone (no rough conditioning, no pairing). First of two runs
# (line here, rough in run_domain_lora_rough_20260804.sh) checking
# whether the base SD checkpoint can be adapted to genuinely represent
# each domain individually, before any cross-domain translation attempt
# resumes. See doc/work_log.md ("diffusion" branch, 2026-08-04/05).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=domain_lora_line_20260804
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
CAPTION="lineartstyle, monochrome line art, clean linework, manga panel, black and white"
IMAGE_DIRS="dataset/pairs_480/train/line_combined_koma_20260729"

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] domain_lora line-domain start (epochs=$EPOCHS rank=$LORA_RANK)"

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
  "Lineart domain LoRA (line domain) training complete" \
  "Review results/${TAG}/contact_sheet_${TAG}.png" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] domain_lora line-domain complete"
