#!/bin/bash
# Untested hypothesis #2 (inbox/initial_notice.md, 2026-09-02): after
# hypothesis #1 (clip_pairs data-pool structural bias) was measured and
# rejected (results/data_bias_check_20260902/), the leading remaining
# explanation for the persistent cross-hatch hallucination is that
# LoRA rank16 lacks the capacity to represent thin/clean linework on top
# of the frozen public ControlNet checkpoint, and the model falls back to
# a "safe" dense-hatch texture instead. This isolates rank as the single
# variable: same base_ckpt/ControlNet-init/conditioning/epochs/lr as the
# 2026-08-27 manga_line run (candidate #3, which independently scored best
# on orientation_entropy among the SD1.5 candidates), only rank16->rank32.
# lora_alpha == rank in train_controlnet.py, so effective LoRA scale stays
# ~1x and no other hyperparameter needs retuning for this comparison.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
TAG=controlnet_lora_manga_rank32_20260902
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-32}
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

echo "[$(date --iso-8601=seconds)] controlnet LoRA (manga_line preprocessing, rank32) fine-tune start (epochs=$EPOCHS rank=$LORA_RANK lr=$LR)"
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
  "ControlNet LoRA (manga_line preprocessing, rank32) fine-tune complete" \
  "Review results/${TAG}_eval/" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] controlnet LoRA (manga_line preprocessing, rank32) fine-tune complete"
