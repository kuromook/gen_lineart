#!/bin/bash
# ControlNet bootstrap fine-tune on synthetic (pseudo-paired) data, while
# waiting for the real paired rough/line extraction batch (user: ready
# "next week" at the earliest, possibly later depending on how much is
# needed).
#
# Motivation (doc/work_log.md, `diffusion` branch, 2026-08-08): today's
# diagnostic found the public `control_v11p_sd15s2_lineart_anime`
# ControlNet (massively pretrained, downloaded via diffusers) produces far
# better rough->line structural correspondence than training our own
# ControlNet from scratch on ~1,500-2,500 tiles ever did -- decision was to
# fine-tune that public checkpoint on our own pairs once they exist,
# rather than train from scratch again. This run is the "warm-up" while
# waiting: rather than idle, generate synthetic pseudo-pairs from data we
# already have (deterministic roughification of real clean line art via
# tools/pair_extraction/roughify_line.py -- NOT SDEdit, which showed
# unreliable structure preservation in the same day's testing) and use
# them for an initial fine-tune pass. Correspondence is guaranteed by
# construction (same source line art, just degraded), unlike a generative
# pseudo-pair approach. This is the same bootstrapping idea used to train
# public canny/scribble ControlNets from millions of real photos via
# deterministic edge extraction, applied in the opposite direction here.
#
# Data: dataset/pairs_480/train/rough_pseudo_roughified_20260808/ (1,489
# synthetic-rough tiles, generated 2026-08-08) paired by filename with the
# real dataset/pairs_480/train/line_combined_koma_20260729/ tiles -- same
# file list as every other line-domain experiment
# (valid_train_combined_koma_20260729.txt). Per-tile WD14 captions reused
# directly from the domain-LoRA work
# (results/domain_lora_line_captiontags_20260807/tags.csv, same 1,489
# tiles) rather than the old single fixed caption that isolation chain
# already showed starves the model of content-disambiguation signal.
#
# Fine-tuned FROM the public pretrained ControlNet (--controlnet-init),
# not ControlNetModel.from_unet(unet) (training from scratch). Lower LR
# than the from-scratch default (2e-6 vs 1e-5) since fine-tuning an
# already-competent pretrained model, not training one from nothing --
# standard transfer-learning practice to avoid wrecking existing
# capability while adapting toward our style.
#
# This checkpoint is explicitly a bootstrap/placeholder -- plan is to
# continue fine-tuning it further (or restart similarly) once real paired
# data arrives, not to treat this pseudo-pair result as final.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=controlnet_bootstrap_pseudo_pairs_20260808
EPOCHS=${EPOCHS:-10}
LR=${LR:-2e-6}
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

echo "[$(date --iso-8601=seconds)] controlnet bootstrap (pseudo-pairs) start (epochs=$EPOCHS lr=$LR init=$CONTROLNET_INIT)"

"$PY" scripts/train_controlnet.py \
  --file-list "$FILE_LIST" \
  --rough-dir "$ROUGH_DIR" \
  --line-dir "$LINE_DIR" \
  --base-ckpt "$BASE_CKPT" \
  --controlnet-init "$CONTROLNET_INIT" \
  --caption-csv "$CAPTION_CSV" \
  --output-dir "$CKPT" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --save-steps 300 \
  --log-steps 50

echo "=== sample + eval (same 10-tile diagnostic subset as today's public-ControlNet tests) ==="
"$PY" scripts/infer_controlnet.py \
  --sample-list dataset/pairs_480/diag_controlnet_same_coordinate_10.txt \
  --rough-dir dataset/pairs_480/train/rough \
  --controlnet-dir "$CKPT/final" \
  --base-ckpt "$BASE_CKPT" \
  --caption "monochrome line art, manga panel, black and white" \
  --tag "${TAG}_eval"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "lr=$LR"
  echo "controlnet_init=$CONTROLNET_INIT"
  echo "checkpoint=$CKPT/final"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "ControlNet pseudo-pair bootstrap fine-tune complete" \
  "Review checkpoint at $CKPT/final" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] controlnet bootstrap (pseudo-pairs) complete"
