#!/bin/bash
# Full cs sweep for this track's configuration, baseline only, chained so it
# needs watching once:
#
#   1. inference at every cs value, both ends included
#   2. score every cs directory as an arm, with the conditioning map's own
#      score written in as the row everything is read against (lesson 6 --
#      a gt_bsds_f1 without it is unreadable). That row must come out 0.3164
#      or the scoring path has drifted
#   3. contact sheet, columns in ascending cs
#
# Every step resumes: inference skips tiles already written.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/sh1/deepl/lineart/venv/bin/python
ROOT=results/cs_sweep_20261002
LOG=logs/cs_sweep_20261002.log
mkdir -p logs

"$PY" -u experiments/cs_sweep_20261002.py \
  --cs-values 0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0 \
  --out-root "$ROOT" >> "$LOG" 2>&1
echo "[chain] inference done" >&2

"$PY" -u experiments/score_ipadapter_probe_20260930.py --probe-root "$ROOT" --workers 6 \
  > logs/score_cs_sweep_20261002.log 2>&1
echo "[chain] scored" >&2

"$PY" -u experiments/montage_cs_sweep_20261002.py --sweep-root "$ROOT" --max-rows 5 \
  >> logs/score_cs_sweep_20261002.log 2>&1
echo "[chain] ALL DONE" >&2
touch "$ROOT/.chain_complete"
