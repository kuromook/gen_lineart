#!/bin/bash
# The two documented options this track had not used, checked after the
# 2026-10-02 usage review:
#
#   plus        `ip-adapter-plus_sd15` conditions on PATCH embeddings, not the
#               global pooled vector. The track's prediction ("what it carries
#               is appearance, not where the strokes go") rests on the pooled
#               premise, which the h94 model card confirms for the checkpoint we
#               ran -- and which therefore does not cover this one. This is the
#               only variant that could touch problem 1.
#   style_only  InstantStyle as diffusers implements it: inject into up block_0
#               (the style block) alone instead of every block. The docs warn
#               that injecting everywhere "focuses more on the image prompt",
#               and all-layer injection is what we did. The tone run's
#               neither_ink 0.195 -> 0.320, with stripes and screentone
#               appearing, is what content leaking in looks like.
#
# cs is fixed at 1.0 throughout, and the binding matters: 1.0 for matched
# ControlNet / SD1.5 / 512 / this track, chosen because the 192-tile sweep put
# it where control is engaged (vs_condition_f1 0.676) but not yet saturated on
# the map, and where the tone effect was largest. It is not borrowed.
#
# The baseline is regenerated in the first cell as the known-value check: it
# must come back f1 0.2745 / vs_cond 0.6756 / near_white 0.008, the matched
# sweep's cs1.00 row. The adapter is irrelevant to it, so a mismatch means the
# run is not comparable and nothing else is read.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/sh1/deepl/lineart/venv/bin/python
CN=/home/sh1/disk/checkpoint/ControlNet/control_v11p_sd15_lineart
ROOT=results/ipadapter_variants_20261002
LOG=logs/ipadapter_variants_20261002.log
PLUS=ip-adapter-plus_sd15.safetensors
mkdir -p logs

run () {  # label, weight_name, scale_mode, arms, scales
  echo "[chain] $1" >&2
  "$PY" -u experiments/ipadapter_probe_20260930.py \
    --controlnet-dir "$CN" --controlnet-conditioning-scale 1.0 \
    --ip-weight-name "$2" --ip-scale-mode "$3" --arms "$4" --ip-scales "$5" \
    --out-root "$ROOT/$1" >> "$LOG" 2>&1
}

run plus_uniform        "$PLUS"                      uniform    baseline,gt_same,gt_otherfam 0.4,0.8,1.0
run pooled_style_only   ip-adapter_sd15.safetensors  style_only gt_same,gt_otherfam          0.8,1.0
run plus_style_only     "$PLUS"                      style_only gt_same,gt_otherfam          0.8,1.0
echo "[chain] inference done" >&2

for V in plus_uniform pooled_style_only plus_style_only; do
  "$PY" -u experiments/score_ipadapter_probe_20260930.py --probe-root "$ROOT/$V" --workers 6 \
    > "logs/score_ipadapter_variants_$V.log" 2>&1
  "$PY" -u experiments/montage_cs_sweep_20261002.py --sweep-root "$ROOT/$V" --max-rows 5 \
    >> "logs/score_ipadapter_variants_$V.log" 2>&1
done
echo "[chain] scored" >&2

# One decomposition table over the new variants AND the two already measured,
# so problem 1 and problem 2 are read on the same axis across all of them.
"$PY" -u experiments/stroke_decomposition_20261002.py \
  --roots "$ROOT/plus_uniform" "$ROOT/pooled_style_only" "$ROOT/plus_style_only" \
          results/ipadapter_tone_matched_20261002/cs1.0 \
  --out "$ROOT/stroke_decomposition.csv" --per-tile "$ROOT/stroke_decomposition_per_tile.csv" \
  > logs/decomp_ipadapter_variants.log 2>&1
echo "[chain] ALL DONE" >&2
touch "$ROOT/.chain_complete"
