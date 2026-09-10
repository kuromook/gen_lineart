#!/bin/bash
# SD1.5 adaptation of Track B's (lineart-controlnet-sdxl-fidelity) grey-source
# cross: is the grey-wash background in our best checkpoint's output intrinsic
# to the ControlNet checkpoint, or to the conditioning preprocessor? See
# doc/initial_notice.md, "Track Bからの申し送り(2026-09-06)" -- Track B's
# LoRA-free ControlNet x preprocessor 2x2 on SDXL isolated grey to the
# `lineart_anime` ControlNet checkpoint itself (unaffected by conditioning
# image or resolution). Template: ../lineart-controlnet-sdxl-fidelity/
# experiments/run_grey_source_cross_20260906.sh + score_grey_source_cross_20260906.py.
#
# Difference from the SDXL template (constraints specific to SD1.5, see
# doc/initial_notice.md and this script's companion scorer for detail):
#   - No SD1.5 `manga_line` ControlNet exists (SDXL-only), so the ControlNet
#     axis here is the two SD1.5 checkpoints actually on disk
#     (control_v11p_sd15_lineart, control_v11p_sd15s2_lineart_anime) crossed
#     against three already-materialized preprocessors (lineart_anime,
#     lineart_coarse, manga_line) -- a 2x3 grid, not a literal 2x2.
#   - SD1.5 has no established >512 resolution path, so instead of sweeping
#     resolution (as the SDXL template did) this sweeps
#     controlnet_conditioning_scale (1.0/2.5/3.5), per this track's own
#     operating rule: never judge on cs=1.0 alone.
#
# No LoRA is applied anywhere in this script (bare pretrained ControlNet
# only), matching Track B's isolation intent: this checks whether the grey
# is a property of the ControlNet checkpoint itself, independent of any
# fine-tune this track has since done on top of it.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
INFER=scripts/infer_controlnet.py

BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors
CN_LINEART=$HOME/disk/checkpoint/ControlNet/control_v11p_sd15_lineart
CN_ANIME=$HOME/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime

CAPTION="monochrome line art, manga panel, black and white"
SAMPLES=data/diag_valid5.txt
CS_VALUES=${CS_VALUES:-"1.0 2.5 3.5"}
OUT_ROOT=results/grey_source_cross_sd15_20260908
LOG=logs/grey_source_cross_sd15_20260908.log
DONE=logs/grey_source_cross_sd15_20260908.done

# label | controlnet | conditioning dir
CELLS=(
  "cnLineart_condAnime|$CN_LINEART|data/diag_rough_lineart_anime"
  "cnLineart_condCoarse|$CN_LINEART|data/diag_rough_lineart_coarse"
  "cnLineart_condManga|$CN_LINEART|data/diag_rough_manga_line"
  "cnAnime_condAnime|$CN_ANIME|data/diag_rough_lineart_anime"
  "cnAnime_condCoarse|$CN_ANIME|data/diag_rough_lineart_coarse"
  "cnAnime_condManga|$CN_ANIME|data/diag_rough_manga_line"
)

mkdir -p logs "$OUT_ROOT"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] grey-source 2x3 (SD1.5) start: cs [$CS_VALUES]"

for entry in "${CELLS[@]}"; do
  IFS='|' read -r LABEL CN COND <<< "$entry"
  for CS in $CS_VALUES; do
    OUT="$OUT_ROOT/outputs/${LABEL}/cs${CS}"
    if [ -f "$OUT/.complete" ]; then echo "--- $OUT done, skipping"; continue; fi
    echo "--- $LABEL cs=$CS"
    mkdir -p "$OUT"
    "$PY" "$INFER" \
      --sample-list "$SAMPLES" --rough-dir "$COND" \
      --controlnet-dir "$CN" --base-ckpt "$BASE_CKPT" \
      --caption "$CAPTION" \
      --controlnet-conditioning-scale "$CS" \
      --tag "${LABEL}_cs${CS}" --output-dir "$OUT" 2>&1 | grep -vE "^Loading|it/s\]$"
    touch "$OUT/.complete"
  done
done

echo "=== scoring ==="
"$PY" experiments/score_grey_source_cross_sd15_20260908.py

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "cs_values=$CS_VALUES"
  echo "output_dir=$OUT_ROOT"
} > "$DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "SD1.5 grey-source cross (2x3) complete" \
  "Review $OUT_ROOT/scores.csv and $OUT_ROOT/montage_grey_source_sd15.png" || true

echo "[$(date --iso-8601=seconds)] grey-source 2x3 (SD1.5) complete"
