#!/bin/bash
# Enumerate the full alignment-passing tile-candidate pool for the
# remaining koma sources (fitness/gakuen/hamlabi/housei -- ako5ver2 already
# done), using each source's already-materialized masked subregion
# manifest, with style gates loosened to maximal permissiveness but
# alignment gates, dedup, and score-mode kept identical to production
# (tile_region_manifest_480.py's fixed ALIGNMENT_* constants, strict-mode
# scoring, duplicate-overlap 0.50, max-per-region 4, dedup-scope page).
#
# Purpose: gather rough-side crops that were dropped from the paired
# training set purely by style-gate/tile-score cutoffs (not alignment
# failures) as additional unpaired-rough domain-adaptation material for
# Direction 4, alongside the existing skima pool -- per 2026-08-03/04
# session discussion. Filtering for finished-ink contamination and visual
# QC are deliberately NOT automated here (needs human/model-verified
# review before adoption) -- this script only gathers candidates.
set -euo pipefail
cd /home/sh1/deepl/lineart

PY=./venv/bin/python
OUT_ROOT=dataset/unpaired_rough_candidates
mkdir -p "$OUT_ROOT"

for SOURCE in fitness gakuen hamlabi housei; do
  echo "[$(date +%H:%M:%S)] === $SOURCE: permissive measurement pass ==="
  SRC_OUT="$OUT_ROOT/$SOURCE"
  mkdir -p "$SRC_OUT"
  "$PY" tools/pair_extraction/tile_region_manifest_480.py \
    --manifest "dataset/regions_${SOURCE}_koma_subregions_20260729_masked/manifest.csv" \
    --ink-min 0.0 --ink-max 1.0 \
    --max-black-component-ratio 1.0 --max-thick-ink-ratio 1.0 \
    --max-line-width-p50 0.0 --max-long-line-ratio 1.0 \
    --max-soft-ink-ratio 1.0 \
    --min-support 0.90 --score-mode strict \
    --duplicate-overlap 0.50 --max-per-region 4 --min-tile-score 0.0 \
    --dedup-scope page \
    --name-prefix "${SOURCE}koma" \
    --csv-out "$SRC_OUT/measure_all.csv" \
    --qc-out "$SRC_OUT/measure_qc.png" \
    --qc-tail-out "$SRC_OUT/measure_qc_tail.png" \
    --qc-sample-out "$SRC_OUT/measure_qc_sample.png" \
    --rough-out "$SRC_OUT/rough" \
    --line-out "$SRC_OUT/line" \
    --list-out "$SRC_OUT/measure_all.txt" \
    --save
  echo "[$(date +%H:%M:%S)] === $SOURCE: done ==="
done

echo "[$(date +%H:%M:%S)] all sources measured" > "$OUT_ROOT/measure_all.done"
experiments/send_autoloop_notification.sh \
  "Lineart unpaired-rough drop-candidate measurement complete" \
  "fitness/gakuen/hamlabi/housei measurement passes done under $OUT_ROOT. Still need ink-contamination filter + visual QC before adoption (not automated)." || true
