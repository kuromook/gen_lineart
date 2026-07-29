#!/bin/bash
# Panel -> tile pipeline driver for a single koma-panel source.
# Runs materialize_koma_panels.py -> split_koma_panel_subregions.py ->
# build_region_valid_masks.py -> tile_region_manifest_480.py end to end.
#
# Usage: run_koma_tile_pipeline.sh <source> <panels_csv> <zip> <zip_root> <date> [max_soft_ink_ratio]
set -euo pipefail
cd /home/sh1/deepl/lineart

SOURCE="$1"
PANELS_CSV="$2"
ZIP="$3"
ZIP_ROOT="$4"
DATE="$5"
MAX_SOFT_INK_RATIO="${6:-1.0}"
PY=venv/bin/python

PANEL_DIR="dataset/regions_${SOURCE}_koma_panels_${DATE}"
SUBREGION_DIR="dataset/regions_${SOURCE}_koma_subregions_${DATE}"
MASKED_DIR="${SUBREGION_DIR}_masked"

echo "[$(date +%H:%M:%S)] [$SOURCE] === materialize panels (max-chamfer 45) ==="
$PY tools/pair_extraction/materialize_koma_panels.py \
  --panels-csv "$PANELS_CSV" \
  --zip "$ZIP" --zip-root "$ZIP_ROOT" \
  --max-chamfer 45 \
  --out-dir "$PANEL_DIR" \
  --qc-out "results/${SOURCE}_koma_panels_${DATE}_materialized_qc.png" \
  --save

echo "[$(date +%H:%M:%S)] [$SOURCE] === sub-region split + per-sub-region alignment refinement ==="
$PY tools/pair_extraction/split_koma_panel_subregions.py \
  --panel-manifest "${PANEL_DIR}/manifest.csv" \
  --max-chamfer 999 \
  --out-dir "$SUBREGION_DIR" \
  --qc-out "results/${SOURCE}_koma_subregions_${DATE}_qc.png" \
  --save

echo "[$(date +%H:%M:%S)] [$SOURCE] === valid masks (native settings) ==="
$PY tools/pair_extraction/build_region_valid_masks.py \
  --manifest "${SUBREGION_DIR}/manifest.csv" \
  --out-base "$MASKED_DIR" \
  --image-size 0 --reuse-source-images \
  --support-px 20 --window 61 --expand-ignore 16 --close-ignore 16

echo "[$(date +%H:%M:%S)] [$SOURCE] === tile extraction (native-strict gates, max-soft-ink-ratio ${MAX_SOFT_INK_RATIO}) ==="
$PY tools/pair_extraction/tile_region_manifest_480.py \
  --manifest "${MASKED_DIR}/manifest.csv" \
  --ink-min 0.012 --ink-max 0.08 \
  --max-black-component-ratio 0.025 --max-thick-ink-ratio 0.015 \
  --max-line-width-p50 6.0 --max-long-line-ratio 0.25 \
  --max-soft-ink-ratio "$MAX_SOFT_INK_RATIO" \
  --min-support 0.90 --score-mode strict \
  --duplicate-overlap 0.50 --max-per-region 4 --min-tile-score 2.5 \
  --dedup-scope page \
  --name-prefix "${SOURCE}koma" \
  --csv-out "results/${SOURCE}_koma_tiles_480_${DATE}.csv" \
  --qc-out "results/${SOURCE}_koma_tiles_480_${DATE}_qc.png" \
  --qc-tail-out "results/${SOURCE}_koma_tiles_480_${DATE}_qc_tail.png" \
  --qc-sample-out "results/${SOURCE}_koma_tiles_480_${DATE}_qc_sample.png" \
  --rough-out dataset/pairs_480/train/rough \
  --line-out "dataset/pairs_480/train/line_${SOURCE}_koma_${DATE}" \
  --list-out "dataset/pairs_480/valid_train_${SOURCE}_koma_${DATE}.txt" \
  --save

echo "[$(date +%H:%M:%S)] [$SOURCE] === DONE ==="
