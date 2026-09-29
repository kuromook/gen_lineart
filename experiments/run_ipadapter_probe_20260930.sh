#!/bin/bash
# Wait for the probe's inference to finish, then score and build the montage.
# Chained so the whole thing needs watching once, not three times.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/sh1/deepl/lineart/venv/bin/python
ROOT=results/ipadapter_probe_20260930
LOG=logs/ipadapter_probe_20260930.log

while ! grep -aq "done ->" "$LOG"; do sleep 60; done
echo "[chain] inference done, scoring" >&2
"$PY" -u experiments/score_ipadapter_probe_20260930.py --probe-root "$ROOT" --workers 6 \
  > logs/score_ipadapter_probe_20260930.log 2>&1
echo "[chain] montage" >&2
"$PY" -u experiments/montage_ipadapter_probe_20260930.py --probe-root "$ROOT" \
  >> logs/score_ipadapter_probe_20260930.log 2>&1
echo "[chain] ALL DONE" >&2
touch "$ROOT/.chain_complete"
