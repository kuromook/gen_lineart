#!/bin/bash
# 192-tile holdout validation, lineart_family group only (housei explicitly
# excluded per user decision 2026-09-13 -- see doc/initial_notice.md
# "共通基盤からの申し送り: 5枚問題は放置しないこと"). Adapted from Track B's
# ../lineart-controlnet-sdxl-fidelity/experiments/run_holdout_validation_20260912.sh,
# scoped to this track's own preprocessor (manga_line, not lineart_coarse)
# and checkpoints.
#
# WHY: every result this track has produced (round1/round2 consistency_weight
# sweep, the grey-source cross, the style-fidelity diagnostics) rests on the
# same 5 diagnostic tiles (data/diag_valid5.txt). Before trusting the
# round-2 verdict (w=0.4, cs=2.5, near_white_frac 0.813) enough to spend more
# GPU-days on it (consistency_max_timestep sweep, InnerControl), check
# whether it holds at 192 tiles from the same source family. The 5 diag
# tiles are a subset of this 192, so their own numbers get re-measured here
# as an anchor against the historical values.
#
# Inference only. Nothing is trained. Two checkpoints compared:
# round1's leader (w=0.2, cs2.5) and round2's leader (w=0.4, cs2.5) -- the
# two candidates the 5-tile result could not fully separate.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
MLE_PY=/tmp/claude-1000/-home-sh1-deepl-lineart-controlnet-realpairs/bc420faf-b7d1-49e0-8c69-c5d74a386d6d/scratchpad/mle_venv/bin/python
MLE_SCRIPT=/home/sh1/deepl/lineart-controlnet-realpairs/tools/preprocess_manga_line_extraction_condition.py
SHARED=/home/sh1/deepl/lineart/dataset/pairs_480

BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors
CONTROLNET_INIT=$HOME/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime
CAPTION="monochrome line art, manga panel, black and white"

OUT_ROOT=results/holdout_lineart_family_20260913
LOG=logs/holdout_lineart_family_20260913.log
DONE=logs/holdout_lineart_family_20260913.done
TAG=holdout_lineart_family_20260913

mkdir -p logs "$OUT_ROOT"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

on_failure() {
  local code=$?
  echo "[$(date --iso-8601=seconds)] FAILED at line $1 (exit $code)"
  echo "failed_at=$(date --iso-8601=seconds) line=$1 exit=$code" > "logs/${TAG}.failed"
}
trap 'on_failure $LINENO' ERR
rm -f "logs/${TAG}.failed"

echo "[$(date --iso-8601=seconds)] holdout validation (lineart_family, 192 tiles) start"

# --- 1. stage raw roughs (resolve each tile to whichever shared split holds it) ---
echo "=== staging raw roughs ==="
RAW_DIR=data/holdout_lineart_family_rough_raw
mkdir -p "$RAW_DIR"
TILE_LIST=data/holdout_lineart_family.txt
cp "$SHARED/holdout_lineart_family.txt" "$TILE_LIST"
missing=0
while read -r t; do
  [ -z "$t" ] && continue
  [ -f "$RAW_DIR/$t" ] && continue
  if   [ -f "$SHARED/train/rough/$t" ]; then cp "$SHARED/train/rough/$t" "$RAW_DIR/$t"
  elif [ -f "$SHARED/test/rough/$t"  ]; then cp "$SHARED/test/rough/$t"  "$RAW_DIR/$t"
  else echo "MISSING rough: $t" >&2; missing=$((missing+1)); fi
done < "$TILE_LIST"
echo "staged $(ls "$RAW_DIR" | wc -l) roughs, $missing missing"
[ "$missing" -eq 0 ] || { echo "ERROR: some roughs are missing" >&2; exit 1; }

# --- 2. stage GT line (same resolution logic) ---
echo "=== staging GT line ==="
GT_DIR=data/holdout_lineart_family_gt_line
mkdir -p "$GT_DIR"
missing_gt=0
while read -r t; do
  [ -z "$t" ] && continue
  [ -f "$GT_DIR/$t" ] && continue
  if   [ -f "$SHARED/train/line/$t" ]; then cp "$SHARED/train/line/$t" "$GT_DIR/$t"
  elif [ -f "$SHARED/test/line/$t"  ]; then cp "$SHARED/test/line/$t"  "$GT_DIR/$t"
  else echo "MISSING GT: $t" >&2; missing_gt=$((missing_gt+1)); fi
done < "$TILE_LIST"
echo "staged $(ls "$GT_DIR" | wc -l) GT tiles, $missing_gt missing"
[ "$missing_gt" -eq 0 ] || { echo "ERROR: some GT tiles are missing" >&2; exit 1; }

# --- 3. preprocess to manga_line conditioning (isolated venv, see
#        tools/preprocess_manga_line_extraction_condition.py docstring for why) ---
COND_DIR=data/holdout_lineart_family_rough_manga_line
if [ "$(ls "$COND_DIR" 2>/dev/null | wc -l)" -eq "$(wc -l < "$TILE_LIST")" ]; then
  echo "=== manga_line conditioning already complete, skipping ==="
else
  echo "=== preprocessing to manga_line conditioning (isolated venv) ==="
  "$MLE_PY" "$MLE_SCRIPT" \
    --file-list "$TILE_LIST" \
    --rough-dir "$RAW_DIR" \
    --output-dir "$COND_DIR"
fi
WRITTEN=$(ls "$COND_DIR" | wc -l)
EXPECTED=$(wc -l < "$TILE_LIST")
[ "$WRITTEN" -eq "$EXPECTED" ] || { echo "ERROR: conditioning count mismatch ($WRITTEN/$EXPECTED)" >&2; exit 1; }

# --- 4. inference: round1 leader (w=0.2) and round2 leader (w=0.4), both at cs=2.5 ---
# label | checkpoint
CELLS=(
  "w0.2_cs2.5|checkpoints/controlnet_lora_manga_consistency_w0.2_20260908/final"
  "w0.4_cs2.5|checkpoints/controlnet_lora_manga_consistency_w0.4_20260914/final"
)
for entry in "${CELLS[@]}"; do
  IFS='|' read -r LABEL CKPT <<< "$entry"
  OUT="$OUT_ROOT/outputs/$LABEL"
  if [ -f "$OUT/.complete" ]; then echo "--- $LABEL done, skipping"; continue; fi
  echo "--- $LABEL -> $OUT"
  mkdir -p "$OUT"
  "$PY" scripts/infer_controlnet.py \
    --sample-list "$TILE_LIST" --rough-dir "$COND_DIR" \
    --controlnet-dir "$CONTROLNET_INIT" --controlnet-lora-dir "$CKPT" \
    --base-ckpt "$BASE_CKPT" --caption "$CAPTION" \
    --controlnet-conditioning-scale 2.5 \
    --tag "${TAG}_${LABEL}" --output-dir "$OUT" 2>&1 | grep -vE "^Loading|it/s\]$"
  touch "$OUT/.complete"
done

echo "=== scoring (192-tile aggregate, condition_only baseline, oracle, 5-tile anchor) ==="
"$PY" experiments/score_holdout_lineart_family_20260913.py

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tiles=$(wc -l < "$TILE_LIST")"
  echo "output_dir=$OUT_ROOT"
} > "$DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "192-tile holdout validation (lineart_family) complete" \
  "Review $OUT_ROOT/scores_summary.csv and montage" || true

echo "[$(date --iso-8601=seconds)] holdout validation complete"
