#!/bin/bash
# Hypothesis #6 (inbox/initial_notice.md, 2026-09-05): hypotheses #1-#5
# all kept the base UNet frozen -- LoRA was only ever added to the
# ControlNet -- so the UNet's own text-conditioned denoising prior, where
# SD1.5's hatching associations live, was never trainable. This adds a
# second LoRA adapter on the UNet, trained jointly with the ControlNet
# LoRA on the same paired data (scripts/train_controlnet_unet_lora.py).
#
# Everything else is held identical to manga_trained
# (controlnet_lora_manga_20260827): same base_ckpt, ControlNet init,
# manga_line conditioning, rank16 ControlNet LoRA, lr, captions -- so
# --unet-lora-rank is the single isolated variable. Verified 2026-09-05:
# --unet-lora-rank 0 reproduces plain train_controlnet.py's loss values
# bit-for-bit over 6 steps (0.0092/0.0271/0.0378/0.0564/0.0190/0.0501).
#
# EPOCHS stays at manga_trained's 10 (user's call, 2026-09-05) so this run
# is strictly comparable to the ten_model_comparison table rather than
# introducing a second changed variable. The added UNet backward pass
# measured 5.2s/step vs the plain script's 3.1s/step, so 10 epochs
# (10,580 steps) is ~15.3h: a Monday-morning start finishes around
# midnight and gets reviewed Tuesday. Hypothesis #3 (2026-09-03) found
# 2 epochs statistically indistinguishable from 10 for the ControlNet
# LoRA, so EPOCHS=2 (~3.1h) is a defensible fallback if the schedule
# needs to compress -- but the UNet LoRA is a newly trainable component
# and may not converge on the same timescale, so it is not assumed here.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
TAG=controlnet_lora_manga_unetlora_20260905
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
UNET_LORA_RANK=${UNET_LORA_RANK:-16}
LR=${LR:-1e-4}
FILE_LIST=data/train_list.txt
ROUGH_DIR=data/rough_manga_line
LINE_DIR=data/line
CAPTION_CSV=data/captions.csv
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors
CONTROLNET_INIT=$HOME/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime

# 2026-09-05 breakthrough: cs=1.0 (infer_controlnet.py's default, used for
# every eval in this track to date) sits in the region where the hatch
# hallucination dominates and model differences wash out. The optimum is
# cs2.5-3.5, so this evaluates at both the historical 1.0 (comparable to
# the ten_model_comparison table) and 3.0 (the useful operating point).
EVAL_COND_SCALES=${EVAL_COND_SCALES:-"1.0 3.0"}

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

echo "[$(date --iso-8601=seconds)] controlnet LoRA + UNet LoRA fine-tune start (epochs=$EPOCHS controlnet_rank=$LORA_RANK unet_rank=$UNET_LORA_RANK lr=$LR)"
echo "file count: $(wc -l < "$FILE_LIST")"

echo "=== smoke test (6 steps, scratch dir) ==="
SMOKE_CKPT=checkpoints/${TAG}_smoke
rm -rf "$SMOKE_CKPT"
"$PY" scripts/train_controlnet_unet_lora.py \
  --file-list "$FILE_LIST" \
  --rough-dir "$ROUGH_DIR" \
  --line-dir "$LINE_DIR" \
  --base-ckpt "$BASE_CKPT" \
  --controlnet-init "$CONTROLNET_INIT" \
  --controlnet-lora-rank "$LORA_RANK" \
  --unet-lora-rank "$UNET_LORA_RANK" \
  --caption-csv "$CAPTION_CSV" \
  --output-dir "$SMOKE_CKPT" \
  --max-train-steps 6 \
  --log-steps 1
echo "smoke test passed, cleaning up scratch checkpoint"
rm -rf "$SMOKE_CKPT"

echo "=== full fine-tune ($EPOCHS epochs) ==="
"$PY" scripts/train_controlnet_unet_lora.py \
  --file-list "$FILE_LIST" \
  --rough-dir "$ROUGH_DIR" \
  --line-dir "$LINE_DIR" \
  --base-ckpt "$BASE_CKPT" \
  --controlnet-init "$CONTROLNET_INIT" \
  --controlnet-lora-rank "$LORA_RANK" \
  --unet-lora-rank "$UNET_LORA_RANK" \
  --caption-csv "$CAPTION_CSV" \
  --output-dir "$CKPT" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --save-steps 500 \
  --log-steps 50

for CS in $EVAL_COND_SCALES; do
  echo "=== eval (diag5, controlnet_conditioning_scale=$CS) ==="
  "$PY" /home/sh1/deepl/lineart/scripts/infer_controlnet.py \
    --sample-list data/diag_valid5.txt \
    --rough-dir data/diag_rough_manga_line \
    --controlnet-dir "$CONTROLNET_INIT" \
    --controlnet-lora-dir "$CKPT/final" \
    --lora-dir "$CKPT/final/unet_lora" \
    --base-ckpt "$BASE_CKPT" \
    --caption "monochrome line art, manga panel, black and white" \
    --controlnet-conditioning-scale "$CS" \
    --tag "${TAG}_eval_cs${CS}" \
    --output-dir "results/${TAG}_eval_cs${CS}"
done

rm -rf "$CKPT/resume_state"
echo "removed resume_state (no longer needed after successful completion)"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "lora_rank=$LORA_RANK"
  echo "unet_lora_rank=$UNET_LORA_RANK"
  echo "lr=$LR"
  echo "file_list=$FILE_LIST"
  echo "rough_dir=$ROUGH_DIR"
  echo "line_dir=$LINE_DIR"
  echo "caption_csv=$CAPTION_CSV"
  echo "controlnet_init=$CONTROLNET_INIT"
  echo "checkpoint=$CKPT/final"
  echo "unet_lora=$CKPT/final/unet_lora"
  echo "eval_cond_scales=$EVAL_COND_SCALES"
} > "$DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "ControlNet LoRA + UNet LoRA fine-tune complete" \
  "Review results/${TAG}_eval_cs*/" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] controlnet LoRA + UNet LoRA fine-tune complete"
