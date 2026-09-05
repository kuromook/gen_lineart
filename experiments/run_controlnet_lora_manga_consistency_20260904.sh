#!/bin/bash
# Hypothesis #4 (inbox/initial_notice.md, 2026-09-03/04): after hypotheses
# #1 (data-pool bias), #2 (LoRA rank capacity), and #3 (epoch count) were
# all measured and rejected, this tests a loss-function-side intervention:
# scripts/train_controlnet_consistency.py adds an auxiliary Sobel
# edge-magnitude L1 loss between the model's one-step x0 estimate (decoded
# via the frozen VAE) and the GT target image, masked to low timesteps
# (< --consistency-max-timestep). See that script's docstring for the full
# design rationale (in particular why the literal ControlNet++-style
# roundtrip through the manga_line condition extractor was ruled out --
# venv incompatibility). Same base_ckpt/ControlNet-init/data/conditioning/
# rank/epochs/lr as manga_trained (2026-08-27, the best orientation_entropy
# candidate so far) so this isolates the one new variable: the added loss
# term. --consistency-weight 0 reproduces manga_trained's plain behavior
# exactly (verified via smoke test 2026-09-03), so this script doubles as
# the A/B harness.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
TAG=controlnet_lora_manga_consistency_20260904
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
LR=${LR:-1e-4}
CONSISTENCY_WEIGHT=${CONSISTENCY_WEIGHT:-0.1}
CONSISTENCY_MAX_TIMESTEP=${CONSISTENCY_MAX_TIMESTEP:-200}
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

echo "[$(date --iso-8601=seconds)] controlnet LoRA (manga_line preprocessing, x0-consistency loss) fine-tune start (epochs=$EPOCHS rank=$LORA_RANK lr=$LR consistency_weight=$CONSISTENCY_WEIGHT consistency_max_timestep=$CONSISTENCY_MAX_TIMESTEP)"
echo "file count: $(wc -l < "$FILE_LIST")"

echo "=== smoke test (6 steps, scratch dir) ==="
SMOKE_CKPT=checkpoints/${TAG}_smoke
rm -rf "$SMOKE_CKPT"
"$PY" scripts/train_controlnet_consistency.py \
  --file-list "$FILE_LIST" \
  --rough-dir "$ROUGH_DIR" \
  --line-dir "$LINE_DIR" \
  --base-ckpt "$BASE_CKPT" \
  --controlnet-init "$CONTROLNET_INIT" \
  --controlnet-lora-rank "$LORA_RANK" \
  --caption-csv "$CAPTION_CSV" \
  --output-dir "$SMOKE_CKPT" \
  --max-train-steps 6 \
  --log-steps 1 \
  --consistency-weight "$CONSISTENCY_WEIGHT" \
  --consistency-max-timestep "$CONSISTENCY_MAX_TIMESTEP"
echo "smoke test passed, cleaning up scratch checkpoint"
rm -rf "$SMOKE_CKPT"

echo "=== full fine-tune ($EPOCHS epochs) ==="
"$PY" scripts/train_controlnet_consistency.py \
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
  --log-steps 50 \
  --consistency-weight "$CONSISTENCY_WEIGHT" \
  --consistency-max-timestep "$CONSISTENCY_MAX_TIMESTEP"

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
  echo "consistency_weight=$CONSISTENCY_WEIGHT"
  echo "consistency_max_timestep=$CONSISTENCY_MAX_TIMESTEP"
  echo "file_list=$FILE_LIST"
  echo "rough_dir=$ROUGH_DIR"
  echo "line_dir=$LINE_DIR"
  echo "caption_csv=$CAPTION_CSV"
  echo "controlnet_init=$CONTROLNET_INIT"
  echo "checkpoint=$CKPT/final"
  echo "eval_dir=results/${TAG}_eval"
} > "$DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "ControlNet LoRA (manga_line preprocessing, x0-consistency loss) fine-tune complete" \
  "Review results/${TAG}_eval/" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] controlnet LoRA (manga_line preprocessing, x0-consistency loss) fine-tune complete"
