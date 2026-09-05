#!/bin/bash
# Follow-up to candidate #1 (controlnet_lora_sdxl_20260829): the first SDXL
# attempt paired animagine-xl-3.1 + Eugeoter/noob-sdxl-controlnet-lineart_anime
# with data/rough_lineart_coarse conditioning, and scored worst of all 5
# models on both axes (gt_bsds_f1 0.0945, orientation_entropy 0.6366),
# with a qualitatively different failure mode: mostly blank/flat output plus
# sparse unrelated strokes, one sample generating content unrelated to the
# input entirely -- conditioning essentially not being followed.
#
# User's hypothesis (2026-08-30): Eugeoter's noob-sdxl-controlnet-* series is
# named after controlnet_aux preprocessor outputs (canny, lineart_anime,
# manga_line, softedge_hed, scribble_pidinet, ...), the same convention as
# the SD1.5 ControlNet v1.1 family -- so "lineart_anime" almost certainly
# expects LineartAnimeDetector-style conditioning, not the lineart_coarse
# conditioning we fed it. This track's own prior finding (2026-08-26) is
# that lineart_anime and lineart_coarse preprocessing produce very different
# ink statistics (lineart_anime discards ~50% of rough content; lineart_coarse
# is denser than GT) -- a real conditioning-format mismatch, on top of the
# already-flagged base-model lineage risk (this ControlNet's README lists
# base_model: Laxhar/sdxl_noob, not cagliostrolab/animagine-xl-3.1).
#
# This run swaps to Eugeoter/noob-sdxl-controlnet-manga_line +
# data/rough_manga_line (already preprocessed for the SD1.5 candidate #3
# manga run, which independently scored best on orientation_entropy/hatch
# -escape) so the ControlNet's expected format and the conditioning we
# supply are name-matched. base_ckpt (animagine-xl-3.1) and all other
# hyperparameters unchanged from the 2026-08-29 SDXL run so this isolates
# the ControlNet-init + conditioning-format variable.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
TAG=controlnet_lora_sdxl_manga_20260830
EPOCHS=${EPOCHS:-2}
LORA_RANK=${LORA_RANK:-16}
LR=${LR:-1e-4}
FILE_LIST=data/train_list.txt
ROUGH_DIR=data/rough_manga_line
LINE_DIR=data/line
CAPTION_CSV=data/captions.csv
BASE_CKPT=/home/sh1/.cache/huggingface/hub/models--cagliostrolab--animagine-xl-3.1/snapshots/483f0c322568ed13697ed01dd0be07204746d12b
CONTROLNET_INIT=/home/sh1/.cache/huggingface/hub/models--Eugeoter--noob-sdxl-controlnet-manga_line/snapshots/bc7619904de6489ba7e171cf27a80f0e12822943

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] controlnet LoRA SDXL(manga_line) fine-tune start (epochs=$EPOCHS rank=$LORA_RANK lr=$LR)"
echo "file count: $(wc -l < "$FILE_LIST")"

echo "=== smoke test (6 steps) ==="
"$PY" scripts/train_controlnet_sdxl.py \
  --file-list "$FILE_LIST" \
  --rough-dir "$ROUGH_DIR" \
  --line-dir "$LINE_DIR" \
  --base-ckpt "$BASE_CKPT" \
  --controlnet-init "$CONTROLNET_INIT" \
  --controlnet-lora-rank "$LORA_RANK" \
  --caption-csv "$CAPTION_CSV" \
  --output-dir "checkpoints/${TAG}_smoke" \
  --epochs 1 \
  --lr "$LR" \
  --batch-size 1 \
  --grad-accum 4 \
  --save-steps 1000 \
  --log-steps 1 \
  --max-train-steps 6
rm -rf "checkpoints/${TAG}_smoke"
echo "smoke test passed, removed checkpoints/${TAG}_smoke"

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

echo "=== eval (diag5, manga_line-preprocessed conditioning) ==="
"$PY" scripts/infer_controlnet_sdxl.py \
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
  echo "base_ckpt=$BASE_CKPT"
  echo "controlnet_init=$CONTROLNET_INIT"
  echo "checkpoint=$CKPT/final"
  echo "eval_dir=results/${TAG}_eval"
} > "$DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "ControlNet LoRA SDXL(manga_line) fine-tune complete" \
  "Review results/${TAG}_eval/" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] controlnet LoRA SDXL(manga_line) fine-tune complete"
