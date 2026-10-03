#!/bin/bash
# Does the image-prompt channel move TONE, in the configuration where the
# spatial channel demonstrably works?
#
# The 2026-10-02 decomposition settled content transfer: with the unpaired
# ControlNet, leaking the tile's own GT as the reference was indistinguishable
# from an unrelated image. Tone is the axis left, and it is the one this track
# predicted up front ("if there is a hit it is tone"), because tone was the only
# axis training ever moved.
#
# It is worth asking here specifically because the paired ControlNet LOSES the
# paper: near_white_frac falls 0.279 -> 0.006 as cs rises, against GT's 0.93.
# The references are GT line art at near_white 0.93, handed over in GT polarity
# (black on white) precisely because that is the look under test.
#
# Two cs, per lesson 1, both inside the range where the paired ControlNet is
# demonstrably steering (vs_condition_f1 0.676 and 0.872) and both with the
# paper already dead:
#
#   cs 1.0  control engaged, not yet saturated on the map
#   cs 2.5  control near saturation -- if residuals at 2.5x drown the adapter,
#           the two cs will disagree and that is itself the finding
#
# ip_scale goes to 1.0 as well as the 0.4/0.8 the void probe used, to give tone
# its best shot before any null is believed.
#
# The baseline arm is regenerated rather than reused, as the known cell this run
# is checked against (foundation Working Discipline 2): same seed, caption and
# scheduler as the matched sweep, so it must reproduce near_white 0.008 at cs1.0
# and 0.006 at cs2.5. If it does not, nothing else here is read.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/sh1/deepl/lineart/venv/bin/python
CN=/home/sh1/disk/checkpoint/ControlNet/control_v11p_sd15_lineart
ROOT=results/ipadapter_tone_matched_20261002
LOG=logs/ipadapter_tone_matched_20261002.log
mkdir -p logs

for CS in 1.0 2.5; do
  OUT="$ROOT/cs$CS"
  echo "[chain] cs=$CS" >&2
  "$PY" -u experiments/ipadapter_probe_20260930.py \
    --controlnet-dir "$CN" --controlnet-conditioning-scale "$CS" \
    --arms baseline,gt_same,gt_otherfam --ip-scales 0.4,0.8,1.0 \
    --out-root "$OUT" >> "$LOG" 2>&1
done
echo "[chain] inference done" >&2

for CS in 1.0 2.5; do
  "$PY" -u experiments/score_ipadapter_probe_20260930.py \
    --probe-root "$ROOT/cs$CS" --workers 6 \
    > "logs/score_ipadapter_tone_matched_cs$CS.log" 2>&1
  "$PY" -u experiments/montage_cs_sweep_20261002.py --sweep-root "$ROOT/cs$CS" --max-rows 5 \
    >> "logs/score_ipadapter_tone_matched_cs$CS.log" 2>&1
done
echo "[chain] scored" >&2

"$PY" -u experiments/stroke_decomposition_20261002.py \
  --roots "$ROOT/cs1.0" "$ROOT/cs2.5" \
  --out "$ROOT/stroke_decomposition.csv" --per-tile "$ROOT/stroke_decomposition_per_tile.csv" \
  > logs/decomp_ipadapter_tone_matched.log 2>&1
echo "[chain] ALL DONE" >&2
touch "$ROOT/.chain_complete"
