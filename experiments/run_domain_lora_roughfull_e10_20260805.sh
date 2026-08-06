#!/bin/bash
# Direction 4 follow-up, epoch-count isolation variant (the "option 1"
# counterpart to run_domain_lora_roughclean_20260805.sh's data-composition
# isolation).
#
# Resumes the original 4-epoch full-pool run (domain_lora_rough_20260804,
# 21,468 images, 10,732 steps) and continues it up to the line domain's
# per-image exposure (10 epochs -> 26,830 steps total on this pool size,
# i.e. ~16,098 additional steps from here). Writes to a NEW output dir so
# the original 4-epoch checkpoint/contact-sheet (already reviewed,
# documented in doc/work_log.md) is left untouched for comparison.
#
# Run this only after reviewing run_domain_lora_roughclean_20260805.sh's
# result -- the two runs isolate different variables (data composition vs.
# epoch count) behind the same "rough LoRA doesn't resemble real tiles"
# symptom; do not launch both conclusions blind.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=domain_lora_roughfull_e10_20260805
EPOCHS_TOTAL_EQUIV=10          # target total epochs over the full pool
STEPS_PER_EPOCH=2683           # from the 20260804 run's own log line
MAX_STEPS=$((STEPS_PER_EPOCH * EPOCHS_TOTAL_EQUIV))
LORA_RANK=${LORA_RANK:-16}
CAPTION="roughsketchstyle, pencil rough sketch, messy sketchy construction lines, monochrome"
IMAGE_DIRS="dataset/pairs_480/train/rough dataset/unpaired_rough_candidates/ako5ver2/rough dataset/unpaired_rough_candidates/fitness/rough dataset/unpaired_rough_candidates/gakuen/rough dataset/unpaired_rough_candidates/hamlabi/rough dataset/unpaired_rough_candidates/housei/rough"
RESUME_FROM=checkpoints/domain_lora_rough_20260804

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] domain_lora roughfull_e10 start (resume_from=$RESUME_FROM target_max_steps=$MAX_STEPS rank=$LORA_RANK)"

"$PY" scripts/train_domain_lora.py \
  --image-dirs $IMAGE_DIRS \
  --output-dir "$CKPT" \
  --caption "$CAPTION" \
  --lora-rank "$LORA_RANK" \
  --max-train-steps "$MAX_STEPS" \
  --resume-from-checkpoint "$RESUME_FROM"

echo "=== sample + contact sheet ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "$CAPTION" \
  --tag "$TAG" \
  --num-samples 16

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "resumed_from=$RESUME_FROM"
  echo "max_train_steps=$MAX_STEPS"
  echo "lora_rank=$LORA_RANK"
  echo "checkpoint=$CKPT/final"
  echo "contact_sheet=results/${TAG}/contact_sheet_${TAG}.png"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart domain LoRA (rough full-pool, 10ep) training complete" \
  "Review results/${TAG}/contact_sheet_${TAG}.png" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] domain_lora roughfull_e10 complete"
