#!/bin/bash
# Re-tile all 5 already-extracted koma sources with loosened dedup/overlap
# settings, to test whether more training tiles from the SAME underlying
# source pages (denser, more overlapping crops -- not genuinely new
# content) reduces the "wobble" instability seen in the 100-epoch
# single-stage direct-regression run (combined_koma_direct_unet_100ep_20260801).
#
# Only reruns the final tiling stage (tile_region_manifest_480.py) against
# the already-materialized masked subregion manifests -- panel detection
# and alignment refinement are NOT rerun, so this is cheap.
#
# Loosened vs production settings (run_koma_tile_pipeline.sh):
#   --duplicate-overlap 0.50 -> 0.80  (allow much more overlap before dropping)
#   --max-per-region 4 -> 10          (more than double the per-region cap)
#   --min-tile-score 2.5 -> 2.0       (slightly more permissive quality gate)
# All other quality gates (ink-min/max, black-component, line-width, etc.)
# are unchanged, so this should not meaningfully degrade tile quality --
# it only accepts more of the same-quality candidates that dedup previously
# discarded as "too similar to an already-kept tile."
#
# New tiles use a "komadense" name-prefix suffix so they cannot collide
# with the existing combined_koma_20260729 filenames; the caller is
# expected to take the union of the two file lists for training.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
DATE=20260801
LOG=logs/koma_dense_retile_${DATE}.log

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] koma dense retile start"

retile() {
  local source=$1
  local masked_dir="dataset/regions_${source}_koma_subregions_20260729_masked"
  echo "=== [$source] dense retile ==="
  "$PY" tools/pair_extraction/tile_region_manifest_480.py \
    --manifest "${masked_dir}/manifest.csv" \
    --ink-min 0.012 --ink-max 0.08 \
    --max-black-component-ratio 0.025 --max-thick-ink-ratio 0.015 \
    --max-line-width-p50 6.0 --max-long-line-ratio 0.25 \
    --max-soft-ink-ratio 1.0 \
    --min-support 0.90 --score-mode strict \
    --duplicate-overlap 0.80 --max-per-region 10 --min-tile-score 2.0 \
    --dedup-scope page \
    --name-prefix "${source}komadense" \
    --csv-out "results/${source}_koma_tiles_480_dense_${DATE}.csv" \
    --qc-out "results/${source}_koma_tiles_480_dense_${DATE}_qc.png" \
    --qc-tail-out "results/${source}_koma_tiles_480_dense_${DATE}_qc_tail.png" \
    --qc-sample-out "results/${source}_koma_tiles_480_dense_${DATE}_qc_sample.png" \
    --rough-out dataset/pairs_480/train/rough \
    --line-out "dataset/pairs_480/train/line_${source}_koma_dense_${DATE}" \
    --list-out "dataset/pairs_480/valid_train_${source}_koma_dense_${DATE}.txt" \
    --save
}

for source in ako5ver2 housei hamlabi fitness gakuen; do
  retile "$source"
done

echo "=== building combined dense line dir + file list ==="
COMBINED_LINE_DIR="dataset/pairs_480/train/line_combined_koma_dense_${DATE}"
mkdir -p "$COMBINED_LINE_DIR"
for source in ako5ver2 housei hamlabi fitness gakuen; do
  cp "dataset/pairs_480/train/line_${source}_koma_dense_${DATE}"/*.jpg "$COMBINED_LINE_DIR/" 2>/dev/null || true
done
cat dataset/pairs_480/valid_train_*_koma_dense_${DATE}.txt > dataset/pairs_480/valid_train_combined_koma_dense_${DATE}_newonly.txt

echo "=== union with the existing production combined_koma_20260729 list ==="
# Also copy the production line files into the same combined dense line
# dir so a single --line-dir covers both old and new tiles.
cp dataset/pairs_480/train/line_combined_koma_20260729/*.jpg "$COMBINED_LINE_DIR/" 2>/dev/null || true
cat dataset/pairs_480/valid_train_combined_koma_20260729.txt dataset/pairs_480/valid_train_combined_koma_dense_${DATE}_newonly.txt \
  | sort -u > dataset/pairs_480/valid_train_combined_koma_dense_${DATE}.txt

orig_count=$(wc -l < dataset/pairs_480/valid_train_combined_koma_20260729.txt)
new_count=$(wc -l < dataset/pairs_480/valid_train_combined_koma_dense_${DATE}_newonly.txt)
union_count=$(wc -l < dataset/pairs_480/valid_train_combined_koma_dense_${DATE}.txt)

echo "original tiles: $orig_count"
echo "new dense-retile tiles: $new_count"
echo "union (final densified training list): $union_count"
echo "combined line dir: $COMBINED_LINE_DIR"
echo "combined list: dataset/pairs_480/valid_train_combined_koma_dense_${DATE}.txt"

echo "[$(date --iso-8601=seconds)] koma dense retile complete"
