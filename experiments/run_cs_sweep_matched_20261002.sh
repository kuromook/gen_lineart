#!/bin/bash
# cs sweep with the ControlNet that is actually PAIRED with the preprocessor
# that produced the stored conditioning maps.
#
# WHY (2026-10-02): the maps on disk are `LineartDetector(coarse=True)` output.
# The ControlNet trained on that detector's output is
# `control_v11p_sd15_lineart`. Every run in this project so far fed those maps
# to `control_v11p_sd15s2_lineart_anime`, which is trained on a DIFFERENT
# preprocessor (`LineartAnimeDetector`). An 8-tile probe showed the two behave
# completely differently: with the anime model vs_condition_f1 stayed flat
# (0.31 -> 0.36) while the image only got darker, and swapping in a deliberately
# WRONG conditioning map changed gt_f1 by 0.007 -- i.e. the control channel
# carried nothing. With the matched model vs_condition_f1 rose monotonically
# 0.34 -> 0.86. This run establishes that on all 192 tiles.
#
# GRID: 0.0/0.25/0.5/0.75/1.0 is the real question -- 1.0 is the residual
# magnitude the ControlNet was trained to emit, and the project has never
# measured below it. 1.5 and 2.5 are carried as the upper context the track's
# operating rules require (never judge a ControlNet at a single cs).
#
# cs=0.0 is the most valuable single arm: ControlNet residuals multiplied by
# zero = the base model alone. That is the "unrelated dense drawing" FLOOR that
# every gt_bsds_f1 in this project has to be read against. The 8-tile estimate
# was gt_f1 0.177 / vs_condition 0.307.
#
# Everything else is identical to the 2026-09-30 probe: same base checkpoint,
# caption, scheduler, steps, guidance, resolution and per-tile seed. No
# IP-Adapter is loaded.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/sh1/deepl/lineart/venv/bin/python
CN="$HOME/disk/checkpoint/ControlNet/control_v11p_sd15_lineart"
ROOT=results/cs_sweep_matched_20261002
LOG=logs/cs_sweep_matched_20261002.log
mkdir -p logs

"$PY" -u experiments/cs_sweep_20261002.py \
  --controlnet-dir "$CN" \
  --cs-values 0.0,0.25,0.5,0.75,1.0,1.5,2.5 \
  --out-root "$ROOT" >> "$LOG" 2>&1
echo "[chain] inference done" >&2

"$PY" -u experiments/score_ipadapter_probe_20260930.py --probe-root "$ROOT" --workers 6 \
  > logs/score_cs_sweep_matched_20261002.log 2>&1
echo "[chain] scored" >&2

"$PY" -u experiments/montage_cs_sweep_20261002.py --sweep-root "$ROOT" --max-rows 5 \
  >> logs/score_cs_sweep_matched_20261002.log 2>&1
echo "[chain] ALL DONE" >&2
touch "$ROOT/.chain_complete"
