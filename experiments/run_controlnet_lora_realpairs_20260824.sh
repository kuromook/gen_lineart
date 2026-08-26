#!/bin/bash
# The actual real-paired-data fine-tune this project has been waiting for
# since the 2026-08-08 decision (doc/work_log.md, "ControlNet Hallucination
# Re-Diagnosed"): LoRA fine-tune of the public `control_v11p_sd15s2_
# lineart_anime` checkpoint (not a from-scratch ControlNetModel.from_unet
# copy) on real rough/line pairs, LoRA rather than full fine-tune (confirmed
# safer -- the zero-conv safety net only protects cold-start training, see
# the 2026-08-08 pseudo-pair bootstrap entry), per-tile WD14 captions from
# the start, and the `lineart_anime` preprocessor applied to rough tiles
# before conditioning (so the model sees the same clean edge-map format its
# own pretraining used, not a raw noisy pencil scan).
#
# Data: combined_koma_20260729 (1489, existing) + clip_pairs_koma_20260823
# (6978, new, this session's koma-panel pipeline run on the extraction
# tool's QC'd/koma-layer-enhanced clip_pairs delivery) = 8467 pairs,
# dataset/pairs_480/valid_train_combined_all_20260824.txt. Conditioning
# images pre-processed with tools/pair_extraction/
# preprocess_lineart_anime_condition.py into
# dataset/pairs_480/train/rough_lineart_anime_20260824/. Captions merged
# from both sources' WD14 tagging passes,
# dataset/pairs_480/captions_combined_all_20260824_wd14.csv.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=controlnet_lora_realpairs_20260824
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
LR=${LR:-1e-4}
FILE_LIST=dataset/pairs_480/valid_train_combined_all_20260824.txt
ROUGH_DIR=dataset/pairs_480/train/rough_lineart_anime_20260824
LINE_DIR=dataset/pairs_480/train/line_combined_all_20260824
CAPTION_CSV=dataset/pairs_480/captions_combined_all_20260824_wd14.csv
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors
CONTROLNET_INIT=$HOME/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] controlnet LoRA real-pairs fine-tune start (epochs=$EPOCHS rank=$LORA_RANK lr=$LR)"
echo "file count: $(wc -l < "$FILE_LIST")"

echo "=== smoke test (6 steps, scratch dir) ==="
SMOKE_CKPT=checkpoints/${TAG}_smoke
rm -rf "$SMOKE_CKPT"
"$PY" scripts/train_controlnet.py \
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
"$PY" scripts/train_controlnet.py \
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

echo "=== eval (raw diagnostic tiles, generic caption) ==="
"$PY" scripts/infer_controlnet.py \
  --sample-list dataset/pairs_480/diag_controlnet_same_coordinate_10.txt \
  --rough-dir dataset/pairs_480/train/rough \
  --controlnet-dir "$CONTROLNET_INIT" \
  --controlnet-lora-dir "$CKPT/final" \
  --base-ckpt "$BASE_CKPT" \
  --caption "monochrome line art, manga panel, black and white" \
  --tag "${TAG}_eval"

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

experiments/send_autoloop_notification.sh \
  "ControlNet LoRA real-pairs fine-tune complete" \
  "Review results/${TAG}_eval/" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] controlnet LoRA real-pairs fine-tune complete"
