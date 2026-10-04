#!/bin/bash
# InstantStyle layer-wise injection, rerun with SD1.5's block indices.
#
# The first attempt used the diffusers guide's dictionary verbatim,
# {"up": {"block_0": [0.0, 1.0, 0.0]}}, which is written against an SDXL
# pipeline. SD1.5's up_blocks[0] is a plain UpBlock2D with ZERO attention
# layers, so that names nothing, every IP-Adapter processor keeps its 0.0
# default, and the adapter is silently off. All 8 arms came back byte-identical
# to the baseline. Those outputs are deleted; this is the real measurement.
#
# InstantStyle's own infer_style_sd15.py uses target_blocks=["up_blocks.1"],
# SD1.5's first attention-bearing up block and the analogue of SDXL's
# up_blocks.0.
#
# GATE FIRST. A wrong block index fails silently -- no exception, no warning,
# and the numbers read as a clean null. So before the 192-tile arms run, three
# tiles are generated and compared against the baseline; if the adapter is not
# engaged the script stops rather than producing another void table.
#
# cs is 1.0 for matched ControlNet / SD1.5 / 512 / this track: where the sweep
# put control engaged (vs_condition_f1 0.676) but not saturated on the map.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/sh1/deepl/lineart/venv/bin/python
CN=/home/sh1/disk/checkpoint/ControlNet/control_v11p_sd15_lineart
ROOT=results/instantstyle_20261003
LOG=logs/instantstyle_20261003.log
PLUS=ip-adapter-plus_sd15.safetensors
BASELINE=results/ipadapter_variants_20261002/plus_uniform/baseline
mkdir -p logs "$ROOT"

gate () {  # label, weight_name, mode
  rm -rf "$ROOT/_gate_$1"
  "$PY" -u experiments/ipadapter_probe_20260930.py --limit 3 \
    --controlnet-dir "$CN" --controlnet-conditioning-scale 1.0 \
    --ip-weight-name "$2" --ip-scale-mode "$3" --arms gt_same --ip-scales 1.0 \
    --out-root "$ROOT/_gate_$1" >> "$LOG" 2>&1
  "$PY" - "$ROOT/_gate_$1/gt_same_s1.0" "$BASELINE" "$1" <<'PYEOF'
import sys
import numpy as np
from pathlib import Path
from PIL import Image
arm, base, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
deltas = []
for p in sorted(arm.glob("*_out.png")):
    b = base / p.name
    if not b.exists():
        continue
    deltas.append(np.abs(np.asarray(Image.open(p).convert("L"), dtype=int)
                         - np.asarray(Image.open(b).convert("L"), dtype=int)).mean())
m = float(np.mean(deltas)) if deltas else 0.0
print(f"[gate] {label}: mean|delta| vs baseline = {m:.2f} over {len(deltas)} tiles")
sys.exit(0 if m > 1.0 else 1)
PYEOF
}

for cell in "pooled:ip-adapter_sd15.safetensors" "plus:$PLUS"; do
  LABEL="${cell%%:*}"; W="${cell#*:}"
  if ! gate "$LABEL" "$W" style_only; then
    echo "[chain] GATE FAILED for $LABEL -- adapter not engaged, stopping" >&2
    exit 1
  fi
done
echo "[chain] gates passed" >&2

run () {  # label, weight_name, mode
  echo "[chain] $1" >&2
  "$PY" -u experiments/ipadapter_probe_20260930.py \
    --controlnet-dir "$CN" --controlnet-conditioning-scale 1.0 \
    --ip-weight-name "$2" --ip-scale-mode "$3" \
    --arms gt_same,gt_otherfam --ip-scales 0.8,1.0 \
    --out-root "$ROOT/$1" >> "$LOG" 2>&1
}

run pooled_style_only  ip-adapter_sd15.safetensors style_only
run plus_style_only    "$PLUS"                     style_only
run plus_style_layout  "$PLUS"                     style_layout
echo "[chain] inference done" >&2

for V in pooled_style_only plus_style_only plus_style_layout; do
  cp -n "$BASELINE"/*.png "$ROOT/$V/baseline/" 2>/dev/null || {
    mkdir -p "$ROOT/$V/baseline"; cp "$BASELINE"/*.png "$ROOT/$V/baseline/"; }
  "$PY" -u experiments/score_ipadapter_probe_20260930.py --probe-root "$ROOT/$V" --workers 6 \
    > "logs/score_instantstyle_$V.log" 2>&1
  "$PY" -u experiments/montage_cs_sweep_20261002.py --sweep-root "$ROOT/$V" --max-rows 5 \
    >> "logs/score_instantstyle_$V.log" 2>&1
done
echo "[chain] scored" >&2

"$PY" -u experiments/stroke_decomposition_20261002.py \
  --roots "$ROOT/pooled_style_only" "$ROOT/plus_style_only" "$ROOT/plus_style_layout" \
          results/ipadapter_variants_20261002/plus_uniform \
          results/ipadapter_tone_matched_20261002/cs1.0 \
  --out "$ROOT/stroke_decomposition.csv" --per-tile "$ROOT/stroke_decomposition_per_tile.csv" \
  > logs/decomp_instantstyle.log 2>&1
echo "[chain] ALL DONE" >&2
touch "$ROOT/.chain_complete"
