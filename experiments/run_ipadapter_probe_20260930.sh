#!/bin/bash
# Full probe sequence, chained so it needs watching once.
#
#   1. wait for the first inference invocation (baseline, gt_same, gt_other)
#   2. run the gt_otherfam arm, added after the housei exemplar turned out to
#      confound drawing style with content (see doc/work_log.md 2026-09-30)
#   3. score every arm, with the conditioning map's own score as the baseline
#      row -- lesson 6 makes a gt_bsds_f1 without it unreadable
#   4. build the montage
#
# Every step resumes: inference skips tiles already written, so re-running this
# after an interruption costs only what was missing.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/sh1/deepl/lineart/venv/bin/python
ROOT=results/ipadapter_probe_20260930
LOG=logs/ipadapter_probe_20260930.log

while ! grep -aq "done ->" "$LOG"; do sleep 60; done
echo "[chain] first invocation done" >&2

"$PY" -u experiments/ipadapter_probe_20260930.py --arms gt_otherfam \
  --ip-scales 0.4,0.8 --out-root "$ROOT" >> "$LOG" 2>&1
echo "[chain] gt_otherfam done" >&2

"$PY" -u experiments/score_ipadapter_probe_20260930.py --probe-root "$ROOT" --workers 6 \
  > logs/score_ipadapter_probe_20260930.log 2>&1
echo "[chain] scored" >&2

"$PY" -u experiments/montage_ipadapter_probe_20260930.py --probe-root "$ROOT" \
  >> logs/score_ipadapter_probe_20260930.log 2>&1
echo "[chain] ALL DONE" >&2
touch "$ROOT/.chain_complete"
