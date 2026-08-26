#!/bin/bash
# Chunked driver for split_koma_panel_subregions.py across all 4750
# materialized clip_pairs koma panels (dataset/regions_clip_pairs_koma_panels_20260822).
# Smoke-tested at 20 panels / 5m18s (~16s/panel, dominated by per-sub-region
# ink-connected-component splitting + local alignment refinement) -- full
# run is ~21h, chunked with the script's own native --offset-panels/
# --limit-panels/--append flags so a crash partway through doesn't lose
# completed work, same convention as the koma alignment stage.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
CHUNK=${CHUNK:-200}
TOTAL=4750
PANEL_MANIFEST=dataset/regions_clip_pairs_koma_panels_20260822/manifest.csv
OUT_DIR=dataset/regions_clip_pairs_koma_subregions_20260822
QC_OUT=results/clip_pairs_koma_subregions_20260822_qc.png
LOG=logs/clip_pairs_koma_subregions_20260822.log
DONE=logs/clip_pairs_koma_subregions_20260822.done

mkdir -p results logs
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] clip_pairs sub-region split start (chunk=$CHUNK total=$TOTAL)"

offset=0
first=1
while [ "$offset" -lt "$TOTAL" ]; do
  limit=$CHUNK
  remaining=$((TOTAL - offset))
  if [ "$remaining" -lt "$CHUNK" ]; then limit=$remaining; fi
  echo "=== chunk offset=$offset limit=$limit ==="
  APPEND_FLAG=""
  if [ "$first" -eq 0 ]; then APPEND_FLAG="--append"; fi
  "$PY" tools/pair_extraction/split_koma_panel_subregions.py \
    --panel-manifest "$PANEL_MANIFEST" \
    --max-chamfer 999 \
    --offset-panels "$offset" --limit-panels "$limit" $APPEND_FLAG \
    --out-dir "$OUT_DIR" \
    --qc-out "$QC_OUT" \
    --save
  first=0
  offset=$((offset + limit))
done

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "out_dir=$OUT_DIR"
  echo "qc_out=$QC_OUT"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "clip_pairs koma sub-region split complete" \
  "Review $QC_OUT and $OUT_DIR/manifest.csv" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] clip_pairs sub-region split complete"
