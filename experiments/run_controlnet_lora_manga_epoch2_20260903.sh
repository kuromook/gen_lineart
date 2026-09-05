#!/bin/bash
# Hypothesis #3 (inbox/initial_notice.md, 2026-09-02/03): after hypothesis
# #1 (data-pool bias) and #2 (LoRA rank capacity, rank16 vs rank32) were
# both measured and rejected, the remaining cheap-to-test explanation is
# epoch count. The rank16 (2026-08-27) and rank32 (2026-09-02) manga_line
# runs both show per-step diffusion loss already at its noisy converged
# level by step ~250 (well under 1 epoch = 1058 steps), suggesting the
# model reaches its steady-state behavior (including the hatch fallback)
# very early -- NOT that it is undertrained. This variant tests that by
# cutting epochs 10->2 (same rank16/lr/data/conditioning as the 2026-08-27
# manga_trained baseline, only epoch count changed) to see whether
# stopping before the model has had as much exposure changes the
# hallucination character, at the cost of positional accuracy.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
TAG=controlnet_lora_manga_epoch2_20260903
EPOCHS=${EPOCHS:-2}
LORA_RANK=${LORA_RANK:-16}
LR=${LR:-1e-4}
FILE_LIST=data/train_list.txt
ROUGH_DIR=data/rough_manga_line
LINE_DIR=data/line
CAPTION_CSV=data/captions.csv
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors
CONTROLNET_INIT=$HOME/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints

WRITTEN=$(find "$ROUGH_DIR" -type f 2>/dev/null | wc -l)
EXPECTED=$(wc -l < "$FILE_LIST")
if [ "$WRITTEN" -ne "$EXPECTED" ]; then
  echo "ERROR: $ROUGH_DIR has $WRITTEN files, expected $EXPECTED -- preprocessing incomplete, refusing to train" >&2
  exit 1
fi

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] controlnet LoRA (manga_line preprocessing, epoch2) fine-tune start (epochs=$EPOCHS rank=$LORA_RANK lr=$LR)"
echo "file count: $(wc -l < "$FILE_LIST")"

echo "=== smoke test (6 steps, scratch dir) ==="
SMOKE_CKPT=checkpoints/${TAG}_smoke
rm -rf "$SMOKE_CKPT"
"$PY" /home/sh1/deepl/lineart/scripts/train_controlnet.py \
  --file-list "$FILE_LIST" \
  --rough-dir "$ROUGH_DIR" \
  --line-dir "$LINE_DIR" \
  --base-ckpt "$BASE_CKPT" \
  --controlnet-init "$CONTROLNET_INIT" \
  --controlnet-lora-rank "$LORA_RANK" \
  --caption-csv "$CAPTION_CSV" \
  --output-dir "$SMOKE_CKPT" \
  --max-train-steps 6 \
  --log-steps 1
echo "smoke test passed, cleaning up scratch checkpoint"
rm -rf "$SMOKE_CKPT"

echo "=== full fine-tune ($EPOCHS epochs) ==="
"$PY" /home/sh1/deepl/lineart/scripts/train_controlnet.py \
  --file-list "$FILE_LIST" \
  --rough-dir "$ROUGH_DIR" \
  --line-dir "$LINE_DIR" \
  --base-ckpt "$BASE_CKPT" \
  --controlnet-init "$CONTROLNET_INIT" \
  --controlnet-lora-rank "$LORA_RANK" \
  --caption-csv "$CAPTION_CSV" \
  --output-dir "$CKPT" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --save-steps 500 \
  --log-steps 50

echo "=== eval (diag5, condition matching training format) ==="
"$PY" /home/sh1/deepl/lineart/scripts/infer_controlnet.py \
  --sample-list data/diag_valid5.txt \
  --rough-dir data/diag_rough_manga_line \
  --controlnet-dir "$CONTROLNET_INIT" \
  --controlnet-lora-dir "$CKPT/final" \
  --base-ckpt "$BASE_CKPT" \
  --caption "monochrome line art, manga panel, black and white" \
  --tag "${TAG}_eval" \
  --output-dir "results/${TAG}_eval"

rm -rf "$CKPT/resume_state"
echo "removed resume_state (no longer needed after successful completion)"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "lora_rank=$LORA_RANK"
  echo "lr=$LR"
  echo "file_list=$FILE_LIST"
  echo "rough_dir=$ROUGH_DIR"
  echo "line_dir=$LINE_DIR"
  echo "caption_csv=$CAPTION_CSV"
  echo "controlnet_init=$CONTROLNET_INIT"
  echo "checkpoint=$CKPT/final"
  echo "eval_dir=results/${TAG}_eval"
} > "$DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "ControlNet LoRA (manga_line preprocessing, epoch2) fine-tune complete" \
  "Review results/${TAG}_eval/" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] controlnet LoRA (manga_line preprocessing, epoch2) fine-tune complete"
