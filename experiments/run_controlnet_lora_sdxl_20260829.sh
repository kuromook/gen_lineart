#!/bin/bash
# Candidate #1 from the 2026-08-27 alternative-model survey
# (inbox/initial_notice.md): SDXL migration. Base: cagliostrolab/animagine-xl-3.1
# (anime-tuned SDXL, diffusers format). ControlNet init:
# Eugeoter/noob-sdxl-controlnet-lineart_anime (diffusers ControlNetModel
# format; trained against a different base -- Laxhar/sdxl_noob, a messy
# non-diffusers training-checkpoint dump not worth chasing -- but SDXL
# ControlNets are architecturally portable across same-family anime
# checkpoints, same tradeoff already validated for the SD1.5 candidate #2
# swap). Conditioning: data/rough_lineart_coarse (reused, same as our
# current best-practice preprocessing).
#
# 2 epochs (not the usual 10): SDXL costs ~3x/step and needs batch_size=1
# (vs SD1.5's 2) to fit 12GB VRAM even with the ControlNet base kept in fp16
# and only the LoRA delta upcast to fp32 -- 10 epochs would be ~85h. 2
# epochs (~17h) roughly matches the sample-throughput of the SD1.5
# candidates' 10-epoch/~10-14h runs, for a comparable-effort first look.
# User is open to a full 10-epoch/~85h run over a multi-day away period
# (from 2026-09-01 Monday onward) if this 2-epoch pass looks promising.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
TAG=controlnet_lora_sdxl_20260829
EPOCHS=${EPOCHS:-2}
LORA_RANK=${LORA_RANK:-16}
LR=${LR:-1e-4}
FILE_LIST=data/train_list.txt
ROUGH_DIR=data/rough_lineart_coarse
LINE_DIR=data/line
CAPTION_CSV=data/captions.csv
BASE_CKPT=/home/sh1/.cache/huggingface/hub/models--cagliostrolab--animagine-xl-3.1/snapshots/483f0c322568ed13697ed01dd0be07204746d12b
CONTROLNET_INIT=/home/sh1/.cache/huggingface/hub/models--Eugeoter--noob-sdxl-controlnet-lineart_anime/snapshots/61ed2d40710b32a5a1c9873f7dec89ff0af9f2a4

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] controlnet LoRA SDXL fine-tune start (epochs=$EPOCHS rank=$LORA_RANK lr=$LR)"
echo "file count: $(wc -l < "$FILE_LIST")"

echo "=== smoke test (6 steps) already verified interactively 2026-08-29, skipping re-run ==="

echo "=== full fine-tune ($EPOCHS epochs) ==="
"$PY" scripts/train_controlnet_sdxl.py \
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
  --batch-size 1 \
  --grad-accum 4 \
  --save-steps 500 \
  --log-steps 50

echo "=== eval (diag5, coarse-preprocessed conditioning) ==="
"$PY" scripts/infer_controlnet_sdxl.py \
  --sample-list data/diag_valid5.txt \
  --rough-dir data/diag_rough_lineart_coarse \
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
  echo "base_ckpt=$BASE_CKPT"
  echo "controlnet_init=$CONTROLNET_INIT"
  echo "checkpoint=$CKPT/final"
  echo "eval_dir=results/${TAG}_eval"
} > "$DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "ControlNet LoRA SDXL fine-tune complete" \
  "Review results/${TAG}_eval/" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] controlnet LoRA SDXL fine-tune complete"
