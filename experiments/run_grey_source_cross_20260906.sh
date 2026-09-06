#!/bin/bash
# Where does the grey wash come from -- the ControlNet, or the conditioning
# preprocessor? (user observation, 2026-09-06: "anime側には一貫して
# グラデーションがかったグレーが配置されており、これはanimeを指定する限り
# ついてくるようにみえる")
#
# results/resolution_sweep_20260906 cannot answer that, because its labels
# change two things at once, following the previous track's pairing:
#   anime_* = noob-sdxl-controlnet-lineart_anime + rough_lineart_coarse
#   manga_* = noob-sdxl-controlnet-manga_line   + rough_manga_line
# and the measured gap is large -- at 512/cs1.0 anime_base has background
# mode 164 with 3.1% of pixels near white, manga_base has mode 255 with
# 76.5% (GT: 255 / 94.8%).
#
# So run the full 2x2 with no LoRA: each ControlNet against each
# conditioning map. If the grey follows the ControlNet row, it is intrinsic
# to lineart_anime and picking that ControlNet always costs paper white. If
# it follows the conditioning column, it is the coarse preprocessor and is
# fixable without changing ControlNet.
#
# The two matched cells duplicate sweep cells deliberately: rerunning them
# here under the same code path keeps the comparison self-contained and
# doubles as a repeat check.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
INFER=scripts/infer_controlnet_sdxl.py

SDXL_BASE=/home/sh1/.cache/huggingface/hub/models--cagliostrolab--animagine-xl-3.1/snapshots/483f0c322568ed13697ed01dd0be07204746d12b
CN_ANIME=/home/sh1/.cache/huggingface/hub/models--Eugeoter--noob-sdxl-controlnet-lineart_anime/snapshots/61ed2d40710b32a5a1c9873f7dec89ff0af9f2a4
CN_MANGA=/home/sh1/.cache/huggingface/hub/models--Eugeoter--noob-sdxl-controlnet-manga_line/snapshots/bc7619904de6489ba7e171cf27a80f0e12822943

CAPTION="monochrome line art, manga panel, black and white"
SAMPLES=data/diag_valid5.txt
RESOLUTIONS=${RESOLUTIONS:-"512 1024"}
CS=${CS:-1.0}
OUT_ROOT=results/grey_source_cross_20260906
LOG=logs/grey_source_cross_20260906.log
DONE=logs/grey_source_cross_20260906.done

# label | controlnet | conditioning dir
CELLS=(
  "cnAnime_condCoarse|$CN_ANIME|data/diag_rough_lineart_coarse"
  "cnAnime_condManga|$CN_ANIME|data/diag_rough_manga_line"
  "cnManga_condCoarse|$CN_MANGA|data/diag_rough_lineart_coarse"
  "cnManga_condManga|$CN_MANGA|data/diag_rough_manga_line"
)

mkdir -p logs "$OUT_ROOT"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] grey-source 2x2 start: res [$RESOLUTIONS] cs=$CS"

for entry in "${CELLS[@]}"; do
  IFS='|' read -r LABEL CN COND <<< "$entry"
  for RES in $RESOLUTIONS; do
    OUT="$OUT_ROOT/outputs/${LABEL}/res${RES}_cs${CS}"
    if [ -f "$OUT/.complete" ]; then echo "--- $OUT done, skipping"; continue; fi
    echo "--- $LABEL res=$RES"
    mkdir -p "$OUT"
    "$PY" "$INFER" \
      --sample-list "$SAMPLES" --rough-dir "$COND" \
      --controlnet-dir "$CN" --base-ckpt "$SDXL_BASE" \
      --caption "$CAPTION" --resolution "$RES" \
      --controlnet-conditioning-scale "$CS" --cpu-offload \
      --tag "${LABEL}_res${RES}" --output-dir "$OUT" 2>&1 | grep -vE "^Loading|it/s\]$"
    touch "$OUT/.complete"
  done
done

echo "=== scoring ==="
"$PY" experiments/score_grey_source_cross_20260906.py

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "resolutions=$RESOLUTIONS"
  echo "cs=$CS"
  echo "output_dir=$OUT_ROOT"
} > "$DONE"

echo "[$(date --iso-8601=seconds)] grey-source 2x2 complete"
