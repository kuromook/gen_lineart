#!/bin/bash
# Weekend job (user decision 2026-09-08: another track holds the GPU until
# Friday, so this is prepared now and launched then).
#
# WHY: the track's headline result -- the bare lineart_anime ControlNet at
# 1024/cs2.5 scores gt_bsds_f1 0.2582, beating every model in the
# eleven-model table and beating our own 50h fine-tune by 0.098 -- rests on
# **five tiles**. Every comparison this project has made on that diag set
# rests on the same five. Before any strategy is decided on "off-the-shelf
# beats everything we trained", that number needs a wider measurement.
#
# Inference only. Nothing is trained here.
#
# TWO GROUPS, SCORED SEPARATELY, because scale and domain must not be
# confounded (the test split is 1,596 housei + 24 lineart_004, so a plain
# random sample would change the source at the same time as the count):
#
#   A. dataset/pairs_480/holdout_lineart_family.txt -- 192 tiles, the same
#      source family as the five diag tiles (168 from the shared train split
#      + 24 from the test split). Verified to have ZERO overlap with our
#      8,467-row train_list.txt, so the fine-tune has seen none of them.
#      This is the primary question: does the five-tile result hold at 192?
#      The five diag tiles are inside this group, so their numbers are also
#      re-measured here as an anchor against the historical values.
#
#   B. dataset/pairs_480/holdout_housei_100.txt -- 100 tiles from a
#      different source (fixed seed 20260912). Secondary question: does it
#      generalise off this family at all?
#
# COST: 292 tiles x 4 configs x ~25s ~= 8.1h, plus ~10 min preprocessing.
# Every cell carries a .complete marker, so an interrupted run resumes by
# re-invoking this script.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
SHARED=/home/sh1/deepl/lineart/dataset/pairs_480

SDXL_BASE=/home/sh1/.cache/huggingface/hub/models--cagliostrolab--animagine-xl-3.1/snapshots/483f0c322568ed13697ed01dd0be07204746d12b
CN_ANIME=/home/sh1/.cache/huggingface/hub/models--Eugeoter--noob-sdxl-controlnet-lineart_anime/snapshots/61ed2d40710b32a5a1c9873f7dec89ff0af9f2a4
FT_LORA=checkpoints/controlnet_lora_sdxl_anime_1024_20260907/final

CAPTION="monochrome line art, manga panel, black and white"
RESOLUTION=1024
OUT_ROOT=results/holdout_validation_20260912
COND_DIR=$OUT_ROOT/conditioning
LOG=logs/holdout_validation_20260912.log
DONE=logs/holdout_validation_20260912.done
TAG=holdout_validation_20260912

mkdir -p logs "$OUT_ROOT" "$COND_DIR"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

on_failure() {
  local code=$?
  echo "[$(date --iso-8601=seconds)] FAILED at line $1 (exit $code)"
  echo "failed_at=$(date --iso-8601=seconds) line=$1 exit=$code" > "logs/${TAG}.failed"
  /home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
    "holdout validation FAILED (exit $code)" "line $1 -- see $LOG" || true
}
trap 'on_failure $LINENO' ERR
rm -f "logs/${TAG}.failed"

echo "[$(date --iso-8601=seconds)] holdout validation start"

# --- 1. gather the roughs -------------------------------------------------
# Group A spans both shared splits, so resolve each tile to whichever split
# holds it rather than assuming one.
echo "=== staging roughs from the shared pool ==="
RAW_DIR=$OUT_ROOT/rough_raw
mkdir -p "$RAW_DIR"
cat dataset/pairs_480/holdout_lineart_family.txt dataset/pairs_480/holdout_housei_100.txt \
  > "$OUT_ROOT/all_tiles.txt"
missing=0
while read -r t; do
  [ -z "$t" ] && continue
  [ -f "$RAW_DIR/$t" ] && continue
  if   [ -f "$SHARED/train/rough/$t" ]; then cp "$SHARED/train/rough/$t" "$RAW_DIR/$t"
  elif [ -f "$SHARED/test/rough/$t"  ]; then cp "$SHARED/test/rough/$t"  "$RAW_DIR/$t"
  else echo "MISSING rough: $t" >&2; missing=$((missing+1)); fi
done < "$OUT_ROOT/all_tiles.txt"
echo "staged $(ls "$RAW_DIR" | wc -l) roughs, $missing missing"
[ "$missing" -eq 0 ] || { echo "ERROR: some roughs are missing" >&2; exit 1; }

# --- 2. conditioning ------------------------------------------------------
# Same preprocessor the whole track used: LineartDetector(coarse=True).
echo "=== preprocessing to lineart_coarse conditioning ==="
"$PY" tools/preprocess_lineart_coarse_condition.py \
  --file-list "$OUT_ROOT/all_tiles.txt" \
  --rough-dir "$RAW_DIR" \
  --output-dir "$COND_DIR"

# --- 3. inference ---------------------------------------------------------
# bare at the plateau (2.0/2.5/3.0) plus the fine-tune at its own best cs
# (2.0), which is the pairing the five-tile verdict was written from.
# label | lora ("-" for bare) | cs
CELLS=(
  "bare_cs2.0|-|2.0"
  "bare_cs2.5|-|2.5"
  "bare_cs3.0|-|3.0"
  "ft_cs2.0|$FT_LORA|2.0"
)
for entry in "${CELLS[@]}"; do
  IFS='|' read -r LABEL LORA CS <<< "$entry"
  OUT="$OUT_ROOT/outputs/$LABEL"
  if [ -f "$OUT/.complete" ]; then echo "--- $LABEL done, skipping"; continue; fi
  echo "--- $LABEL (cs=$CS) -> $OUT"
  mkdir -p "$OUT"
  args=(
    --sample-list "$OUT_ROOT/all_tiles.txt" --rough-dir "$COND_DIR"
    --controlnet-dir "$CN_ANIME" --base-ckpt "$SDXL_BASE"
    --caption "$CAPTION" --resolution "$RESOLUTION"
    --controlnet-conditioning-scale "$CS" --cpu-offload
    --tag "${TAG}_${LABEL}" --output-dir "$OUT"
  )
  [ "$LORA" != "-" ] && args+=(--controlnet-lora-dir "$LORA")
  "$PY" scripts/infer_controlnet_sdxl.py "${args[@]}" 2>&1 | grep -vE "^Loading|it/s\]$"
  touch "$OUT/.complete"
done

echo "=== scoring (groups reported separately, plus the five-tile anchor) ==="
"$PY" experiments/score_holdout_validation_20260912.py

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tiles=$(wc -l < "$OUT_ROOT/all_tiles.txt")"
  echo "output_dir=$OUT_ROOT"
} > "$DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "holdout validation complete" "Review $OUT_ROOT/scores.csv and montage" || true

echo "[$(date --iso-8601=seconds)] holdout validation complete"
