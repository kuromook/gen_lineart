#!/bin/bash
# Retry of the psd_line pool-expansion experiment
# (run_domain_lora_line_sd15base_sksv2_expanded_psdline_20260821.sh,
# rejected -- severe vertical-hatch texture collapse, see doc/work_log.md
# 2026-08-21 entry), this time filtering psd_line down to the ~25% of
# tiles that fall within the original combined_koma pool's own p90/p10
# structural bands (long_line_ratio, long_component_ratio,
# components_per_1k_ink_px -- measured directly, see
# results/lineart_profile_orig_vs_psdline_raw_pools_20260821.csv and
# results/lineart_profile_psdline_full_20260821.csv). 510 of 2022 psd_line
# tiles passed, symlinked into
# dataset/psd_line_koma_extraction_20260809/line_filtered_20260821/.
#
# Pool: line_combined_koma_20260729 (1489) + filtered psd_line (510) =
# 1999 images, a much more conservative +34% vs the rejected attempt's
# +136%. Same recipe as the adopted domain_lora_line_sd15base_sksv2_20260807
# otherwise (rank16 attn-only, SD1.5 base, 10 epochs) -- isolating
# "filtered/smaller addition" as the one changed variable from the
# rejected attempt.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=domain_lora_line_sd15base_sksv2_filtered_psdline_20260821
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
CAPTION="sks style, monochrome line art, manga panel, black and white"

TAGDIR=results/domain_lora_line_captiontags_20260807
FILTERED_PSDLINE_DIR=dataset/psd_line_koma_extraction_20260809/line_filtered_20260821
MERGED_CAPTIONS=$TAGDIR/tags_sksv2_filtered_psdline_20260821.csv

IMAGE_DIRS="dataset/pairs_480/train/line_combined_koma_20260729 $FILTERED_PSDLINE_DIR"
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] domain_lora line-domain filtered-psdline start"
echo "image count check:"
echo "  line_combined_koma_20260729: $(ls dataset/pairs_480/train/line_combined_koma_20260729 | wc -l)"
echo "  filtered psd_line: $(ls $FILTERED_PSDLINE_DIR | wc -l)"
echo "  merged caption rows: $(($(wc -l < "$MERGED_CAPTIONS") - 1))"

echo "=== smoke test (6 steps, scratch dir) ==="
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

echo "=== full training ($EPOCHS epochs, filtered-expanded pool) ==="
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

echo "=== lineart-profile comparison: koma_ref vs adopted-original vs this filtered-expanded attempt ==="
"$PY" tools/evaluation/measure_lineart_profile.py \
  --source "koma_ref=dataset/pairs_480/train/line_combined_koma_20260729" \
  --source "adopted_sksv2_original=results/domain_lora_line_sd15base_sksv2_20260807_scale14" \
  --source "filtered_psdline=results/${TAG}_scale14" \
  --sample-size 300 \
  --output-csv "results/lineart_profile_koma_ref_vs_original_vs_filtered_20260821.csv"

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
  echo "profile_csv=results/lineart_profile_koma_ref_vs_original_vs_filtered_20260821.csv"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart domain LoRA (line domain, filtered psd_line retry) training complete" \
  "Review results/${TAG}*/contact_sheet_*.png and the profile CSV" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] domain_lora line-domain filtered-psdline complete"
