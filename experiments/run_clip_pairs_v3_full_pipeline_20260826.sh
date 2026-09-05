#!/bin/bash
# Full clip_pairs v3 (2026-08-26 full-rerun delivery) koma pipeline, chained
# end to end: panel-alignment search -> materialize -> sub-region split ->
# masks -> tile extraction. v3 replaces v2 (see doc/preprocess/
# raw_dataset_storage_policy.md and doc/work_log.md 2026-08-26 entries):
# the extraction tool withdrew the unreliable sketch_fragment flag and
# raised the line_fragment coverage threshold 0.10->0.30, reclassifying 124
# pages that v2 had marked ok as incomplete. Filtered pool (pair_quality=ok
# + is_primary + koma present) grew 1272 -> 1276, but the *content* changed
# (107 new pairs in, ~103 pages that were bad now excluded net).
#
# Each stage uses the same recipe/gates validated on v2's run (2026-08-21
# through 2026-08-24, see doc/work_log.md), chunked the same way so a crash
# partway through doesn't lose completed work. Smoke-tested against the v3
# zip directly (5 pairs / 3m2s, ~36s/pair, matches v2's per-pair cost) before
# this full run was launched.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
DATE=20260826
ZIP=dataset/raw_zips/dataset_clip_pairs_v3.zip
LOG=logs/clip_pairs_v3_full_pipeline_${DATE}.log
DONE=logs/clip_pairs_v3_full_pipeline_${DATE}.done

mkdir -p results logs dataset/pairs_480/train
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] clip_pairs v3 full pipeline start"

# ---------------------------------------------------------------------------
# Stage 1: panel-alignment search (chunked, ~1276 pairs, ~13h @ ~36s/pair)
# ---------------------------------------------------------------------------
PANELS_CSV=results/clip_pairs_koma_panels_v3_${DATE}.csv
PANELS_JSON=results/clip_pairs_koma_panels_v3_${DATE}.json
PANELS_QC=results/clip_pairs_koma_panels_v3_${DATE}_qc.png
OVERLAY_DIR=results/clip_pairs_koma_panels_v3_${DATE}_overlays

if [ ! -f "$PANELS_CSV" ]; then
  echo "=== stage 1: panel-alignment search ==="
  CHUNK=100
  TOTAL=1276
  start=0
  first=1
  while [ "$start" -lt "$TOTAL" ]; do
    end=$((start + CHUNK))
    if [ "$end" -gt "$TOTAL" ]; then end=$TOTAL; fi
    echo "--- stage1 chunk [$start:$end) ---"
    APPEND_FLAG=""
    if [ "$first" -eq 0 ]; then APPEND_FLAG="--append"; fi
    "$PY" tools/pair_extraction/match_clip_pairs_koma_panels.py \
      --zip "$ZIP" \
      --start "$start" --end "$end" $APPEND_FLAG \
      --csv-out "$PANELS_CSV" --json-out "$PANELS_JSON" \
      --qc-out "$PANELS_QC" --overlay-dir "$OVERLAY_DIR"
    first=0
    start=$end
  done
else
  echo "=== stage 1: skip (already exists: $PANELS_CSV) ==="
fi

# ---------------------------------------------------------------------------
# Stage 2: materialize accepted panels
# ---------------------------------------------------------------------------
PANEL_DIR=dataset/regions_clip_pairs_v3_koma_panels_${DATE}
if [ ! -f "${PANEL_DIR}/manifest.csv" ]; then
  echo "=== stage 2: materialize panels (max-chamfer 45) ==="
  "$PY" tools/pair_extraction/materialize_clip_pairs_koma_panels.py \
    --panels-csv "$PANELS_CSV" \
    --zip "$ZIP" \
    --max-chamfer 45 \
    --out-dir "$PANEL_DIR" \
    --qc-out "results/clip_pairs_koma_panels_v3_${DATE}_materialized_qc.png" \
    --save
else
  echo "=== stage 2: skip (already exists: ${PANEL_DIR}/manifest.csv) ==="
fi

PANEL_TOTAL=$(($(wc -l < "${PANEL_DIR}/manifest.csv") - 1))
echo "materialized panels: $PANEL_TOTAL"

# ---------------------------------------------------------------------------
# Stage 3: sub-region split (chunked, scales with PANEL_TOTAL, ~16s/panel)
# ---------------------------------------------------------------------------
SUBREGION_DIR=dataset/regions_clip_pairs_v3_koma_subregions_${DATE}
if [ ! -f "${SUBREGION_DIR}/manifest.csv" ]; then
  echo "=== stage 3: sub-region split ==="
  CHUNK=200
  offset=0
  first=1
  while [ "$offset" -lt "$PANEL_TOTAL" ]; do
    limit=$CHUNK
    remaining=$((PANEL_TOTAL - offset))
    if [ "$remaining" -lt "$CHUNK" ]; then limit=$remaining; fi
    echo "--- stage3 chunk offset=$offset limit=$limit ---"
    APPEND_FLAG=""
    if [ "$first" -eq 0 ]; then APPEND_FLAG="--append"; fi
    "$PY" tools/pair_extraction/split_koma_panel_subregions.py \
      --panel-manifest "${PANEL_DIR}/manifest.csv" \
      --max-chamfer 999 \
      --offset-panels "$offset" --limit-panels "$limit" $APPEND_FLAG \
      --out-dir "$SUBREGION_DIR" \
      --qc-out "results/clip_pairs_koma_subregions_v3_${DATE}_qc.png" \
      --save
    first=0
    offset=$((offset + limit))
  done
else
  echo "=== stage 3: skip (already exists: ${SUBREGION_DIR}/manifest.csv) ==="
fi

# ---------------------------------------------------------------------------
# Stage 4: valid masks (native settings, matches all 5 existing sources +
# the v2 clip_pairs run)
# ---------------------------------------------------------------------------
MASKED_DIR=${SUBREGION_DIR}_masked
if [ ! -f "${MASKED_DIR}/manifest.csv" ]; then
  echo "=== stage 4: valid masks ==="
  "$PY" tools/pair_extraction/build_region_valid_masks.py \
    --manifest "${SUBREGION_DIR}/manifest.csv" \
    --out-base "$MASKED_DIR" \
    --image-size 0 --reuse-source-images \
    --support-px 20 --window 61 --expand-ignore 16 --close-ignore 16
else
  echo "=== stage 4: skip (already exists: ${MASKED_DIR}/manifest.csv) ==="
fi

# ---------------------------------------------------------------------------
# Stage 5: tile extraction (ako5ver2-derived native-strict gates, same
# recipe as v2's clip_pairs_koma_20260823 run; --max-soft-ink-ratio 0.40,
# NOT housei's 0.50 relaxation, since clip_pairs spans many artists/styles)
# ---------------------------------------------------------------------------
TILES_CSV=results/clip_pairs_v3_koma_tiles_480_${DATE}.csv
LIST_OUT=dataset/pairs_480/valid_train_clip_pairs_v3_koma_${DATE}.txt
if [ ! -f "$TILES_CSV" ]; then
  echo "=== stage 5: tile extraction ==="
  "$PY" tools/pair_extraction/tile_region_manifest_480.py \
    --manifest "${MASKED_DIR}/manifest.csv" \
    --ink-min 0.012 --ink-max 0.08 \
    --max-black-component-ratio 0.025 --max-thick-ink-ratio 0.015 \
    --max-line-width-p50 6.0 --max-long-line-ratio 0.25 \
    --max-soft-ink-ratio 0.40 \
    --min-support 0.90 --score-mode strict \
    --duplicate-overlap 0.50 --max-per-region 4 --min-tile-score 2.5 \
    --dedup-scope page \
    --name-prefix "clipv3koma" \
    --csv-out "$TILES_CSV" \
    --qc-out "results/clip_pairs_v3_koma_tiles_480_${DATE}_qc.png" \
    --qc-tail-out "results/clip_pairs_v3_koma_tiles_480_${DATE}_qc_tail.png" \
    --qc-sample-out "results/clip_pairs_v3_koma_tiles_480_${DATE}_qc_sample.png" \
    --rough-out dataset/pairs_480/train/rough \
    --line-out "dataset/pairs_480/train/line_clip_pairs_v3_koma_${DATE}" \
    --list-out "$LIST_OUT" \
    --save
else
  echo "=== stage 5: skip (already exists: $TILES_CSV) ==="
fi

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "panels_csv=$PANELS_CSV"
  echo "panel_dir=$PANEL_DIR"
  echo "subregion_dir=$SUBREGION_DIR"
  echo "masked_dir=$MASKED_DIR"
  echo "tiles_csv=$TILES_CSV"
  echo "list_out=$LIST_OUT"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "clip_pairs v3 full pipeline complete" \
  "Review $TILES_CSV / ${LIST_OUT} -- integrity audit + dedup still needed before combining into the training pool" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] clip_pairs v3 full pipeline complete"
