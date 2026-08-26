#!/bin/bash
# Expand the adopted line-domain LoRA's training pool with the
# already-extracted but previously-unused psd_line tiles
# (dataset/psd_line_koma_extraction_20260809/line, 2022 tiles, saved
# 2026-08-09 but never fed into any domain-LoRA training run -- see
# doc/work_log.md 2026-08-21 "unpaired" follow-up discussion).
#
# Pool: line_combined_koma_20260729 (1489) + psd_line_koma_extraction_20260809
# (2022) = 3511 images, 2.36x the original line-domain pool. Same recipe as
# the adopted domain_lora_line_sd15base_sksv2_20260807 (rank16 attn-only,
# SD1.5 base, per-image WD14 tags + "sks style, monochrome line art, manga
# panel, black and white" suffix, 10 epochs) -- only the data pool changes,
# isolating "does more/varied line-domain data improve fidelity" per this
# project's isolation-experiment methodology.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=domain_lora_line_sd15base_sksv2_expanded_psdline_20260821
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
CAPTION="sks style, monochrome line art, manga panel, black and white"

TAGDIR=results/domain_lora_line_captiontags_20260807
PSDLINE_DIR=dataset/psd_line_koma_extraction_20260809/line
PSDLINE_FILELIST=$TAGDIR/psdline_filelist_20260821.txt
PSDLINE_CAPTIONS=$TAGDIR/tags_sksv2_psdline_20260821.csv
MERGED_CAPTIONS=$TAGDIR/tags_sksv2_expanded_psdline_20260821.csv

IMAGE_DIRS="dataset/pairs_480/train/line_combined_koma_20260729 $PSDLINE_DIR"
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints "$TAGDIR"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] domain_lora line-domain expanded-with-psdline start"

echo "=== step 1: WD14-tag the new psd_line tiles (2022 images, not yet tagged) ==="
if [ ! -f "$PSDLINE_CAPTIONS" ]; then
  ls -1 "$PSDLINE_DIR" > "$PSDLINE_FILELIST"
  "$PY" scripts/tag_wd14.py \
    --file-list "$PSDLINE_FILELIST" \
    --image-dir "$PSDLINE_DIR" \
    --output-csv "$PSDLINE_CAPTIONS" \
    --caption-suffix "$CAPTION"
else
  echo "already tagged, reusing $PSDLINE_CAPTIONS"
fi

echo "=== step 2: merge with the existing adopted-line captions (tags_sksv2.csv) ==="
head -n1 "$TAGDIR/tags_sksv2.csv" > "$MERGED_CAPTIONS"
tail -n +2 "$TAGDIR/tags_sksv2.csv" >> "$MERGED_CAPTIONS"
tail -n +2 "$PSDLINE_CAPTIONS" >> "$MERGED_CAPTIONS"
echo "merged caption rows: $(($(wc -l < "$MERGED_CAPTIONS") - 1))"

echo "=== step 3: smoke test (6 steps, scratch dir) ==="
SMOKE_CKPT=checkpoints/${TAG}_smoke
rm -rf "$SMOKE_CKPT"
"$PY" scripts/train_domain_lora.py \
  --image-dirs $IMAGE_DIRS \
  --output-dir "$SMOKE_CKPT" \
  --caption "$CAPTION" \
  --caption-csv "$MERGED_CAPTIONS" \
  --lora-rank "$LORA_RANK" \
  --max-train-steps 6 \
  --base-ckpt "$BASE_CKPT"
echo "smoke test passed, cleaning up scratch checkpoint"
rm -rf "$SMOKE_CKPT"

echo "=== step 4: full training ($EPOCHS epochs, expanded pool) ==="
"$PY" scripts/train_domain_lora.py \
  --image-dirs $IMAGE_DIRS \
  --output-dir "$CKPT" \
  --caption "$CAPTION" \
  --caption-csv "$MERGED_CAPTIONS" \
  --lora-rank "$LORA_RANK" \
  --epochs "$EPOCHS" \
  --base-ckpt "$BASE_CKPT"

MOTIF_CAPTION="1girl, solo, close-up, white_background, simple_background, $CAPTION"

echo "=== sample + contact sheet (scale 1.0 baseline) ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "$MOTIF_CAPTION" \
  --base-ckpt "$BASE_CKPT" \
  --tag "$TAG" \
  --num-samples 16

echo "=== sample + contact sheet (scale 1.3) ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "$MOTIF_CAPTION" \
  --base-ckpt "$BASE_CKPT" \
  --lora-scale 1.3 \
  --tag "${TAG}_scale13" \
  --num-samples 16

echo "=== sample + contact sheet (scale 1.4, matches the adopted config's scale) ==="
"$PY" scripts/sample_domain_lora.py \
  --lora-dir "$CKPT/final" \
  --caption "$MOTIF_CAPTION" \
  --base-ckpt "$BASE_CKPT" \
  --lora-scale 1.4 \
  --tag "${TAG}_scale14" \
  --num-samples 16

echo "=== step 5: lineart-profile comparison vs koma_ref (same reference used throughout this project) ==="
"$PY" tools/evaluation/measure_lineart_profile.py \
  --source "koma_ref=dataset/pairs_480/train/line_combined_koma_20260729" \
  --source "lora_line_expanded=results/${TAG}_scale14" \
  --sample-size 300 \
  --output-csv "results/lineart_profile_koma_ref_vs_${TAG}_scale14_20260821.csv"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "lora_rank=$LORA_RANK"
  echo "image_dirs=$IMAGE_DIRS"
  echo "caption_csv=$MERGED_CAPTIONS"
  echo "base_ckpt=$BASE_CKPT"
  echo "checkpoint=$CKPT/final"
  echo "contact_sheet_scale10=results/${TAG}/contact_sheet_${TAG}.png"
  echo "contact_sheet_scale13=results/${TAG}_scale13/contact_sheet_${TAG}_scale13.png"
  echo "contact_sheet_scale14=results/${TAG}_scale14/contact_sheet_${TAG}_scale14.png"
  echo "profile_csv=results/lineart_profile_koma_ref_vs_${TAG}_scale14_20260821.csv"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart domain LoRA (line domain, expanded with psd_line tiles) training complete" \
  "Review results/${TAG}*/contact_sheet_*.png and the profile CSV" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] domain_lora line-domain expanded-with-psdline complete"
