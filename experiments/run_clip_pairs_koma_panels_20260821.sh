#!/bin/bash
# Chunked driver for match_clip_pairs_koma_panels.py across the full
# 1272-pair filtered pool (pair_quality=ok + is_primary + koma present,
# per the extraction tool's own recommendation -- see
# doc/work_log.md 2026-08-21 "Extraction Tool Reply" entries). Smoke-tested
# at 10 pages / 6m30s (~39s/page, dominated by the per-panel coarse+fine
# alignment search, same cost profile as the 5 existing koma-pipeline
# sources) -- full run is ~14h, chunked with --append so a crash partway
# through doesn't lose completed work, matching this project's established
# chunking convention for long extraction runs (fitness/housei 2026-07-26,
# ~250-row chunks).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
CHUNK=${CHUNK:-100}
TOTAL=1272
CSV_OUT=results/clip_pairs_koma_panels_20260821.csv
JSON_OUT=results/clip_pairs_koma_panels_20260821.json
QC_OUT=results/clip_pairs_koma_panels_20260821_qc.png
OVERLAY_DIR=results/clip_pairs_koma_panels_20260821_overlays
LOG=logs/clip_pairs_koma_panels_20260821.log
DONE=logs/clip_pairs_koma_panels_20260821.done

mkdir -p results logs
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] clip_pairs koma panel-alignment start (chunk=$CHUNK total=$TOTAL)"

start=0
first=1
while [ "$start" -lt "$TOTAL" ]; do
  end=$((start + CHUNK))
  if [ "$end" -gt "$TOTAL" ]; then end=$TOTAL; fi
  echo "=== chunk [$start:$end) ==="
  APPEND_FLAG=""
  if [ "$first" -eq 0 ]; then APPEND_FLAG="--append"; fi
  "$PY" tools/pair_extraction/match_clip_pairs_koma_panels.py \
    --start "$start" --end "$end" $APPEND_FLAG \
    --csv-out "$CSV_OUT" --json-out "$JSON_OUT" \
    --qc-out "$QC_OUT" --overlay-dir "$OVERLAY_DIR"
  first=0
  start=$end
done

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "csv_out=$CSV_OUT"
  echo "json_out=$JSON_OUT"
  echo "qc_out=$QC_OUT"
  echo "overlay_dir=$OVERLAY_DIR"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "clip_pairs koma panel-alignment search complete" \
  "Review $QC_OUT and $CSV_OUT (chamfer/edge_f1 per panel)" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] clip_pairs koma panel-alignment complete"
