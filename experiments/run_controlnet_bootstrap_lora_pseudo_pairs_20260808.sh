#!/bin/bash
# ControlNet bootstrap fine-tune, LoRA variant (2026-08-08, hypothesis 1
# from the day's research: full-parameter warm-start fine-tuning of an
# already-trained ControlNet has no zero-convolution safety net -- that
# protection only applies to cold-start (ControlNetModel.from_unet)
# training, per lllyasviel's own training docs, which is why both the
# full-fine-tune bootstrap attempts (1860 steps and a 300-step retry)
# destabilized the pretrained public ControlNet's clean output into
# hatch-texture/soft-rendering failure modes despite retaining some
# structural correspondence (the recurring "sword tile" example). A LoRA
# adapter keeps the base frozen and bounds the learned delta, which should
# behave more like a safe, small perturbation -- the same reasoning that
# makes LoRA fine-tuning of SD checkpoints in general more stable than
# full fine-tuning on small data.
#
# Same pseudo-pair data and per-tile captions as the (rejected) full
# fine-tune attempts: dataset/pairs_480/train/rough_pseudo_roughified_
# 20260808/ (deterministic roughification of real clean line art, see
# tools/pair_extraction/roughify_line.py) paired with dataset/pairs_480/
# train/line_combined_koma_20260729/, WD14 per-tile captions from
# results/domain_lora_line_captiontags_20260807/tags.csv. Only the
# fine-tuning mechanism changes (--controlnet-lora-rank 16 instead of
# full-parameter fine-tune), isolating that one variable for a fair
# comparison against the full-fine-tune bootstrap results.
#
# LR raised back to 1e-4 (vs. 2e-6 for the full fine-tune attempts) --
# standard LoRA practice, since a small adapter needs proportionally
# larger updates to have any effect, unlike full fine-tune where a tiny
# LR is what protects the pretrained weights.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=controlnet_bootstrap_lora_pseudo_pairs_20260808
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
LR=${LR:-1e-4}
FILE_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
ROUGH_DIR=dataset/pairs_480/train/rough_pseudo_roughified_20260808
LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
CAPTION_CSV=results/domain_lora_line_captiontags_20260807/tags.csv
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors
CONTROLNET_INIT=$HOME/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] controlnet bootstrap LoRA (pseudo-pairs) start (epochs=$EPOCHS rank=$LORA_RANK lr=$LR init=$CONTROLNET_INIT)"

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
  --save-steps 300 \
  --log-steps 50

echo "=== eval (raw input + generic caption, same as the full-finetune baseline) ==="
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
  echo "controlnet_init=$CONTROLNET_INIT"
  echo "checkpoint=$CKPT/final"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "ControlNet LoRA pseudo-pair bootstrap fine-tune complete" \
  "Review results/${TAG}_eval/" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] controlnet bootstrap LoRA (pseudo-pairs) complete"
