#!/bin/bash
# Task 1 of the LoRA style-contribution diagnostic (see doc/work_log.md
# "2026-09-13" for the full write-up). User's visual read of the sweep:
# at high consistency_weight, pattern/design lines disappear and the output
# looks pulled toward whatever the base diffusion model wants to draw,
# rather than the rough sketch -- consistent with the architecture fact that
# the LoRA lives only on ControlNet's attention projections (rank16,
# to_k/to_q/to_v/to_out.0), never on the frozen base UNet (plain
# v1-5-pruned-emaonly.safetensors, no anime merge). This generates,
# inference-only, the same 5 diagnostic samples under three conditioning
# inputs -- the real rough sketch (reused from existing eval outputs), a
# blank white image, a blank black image, and a fixed-seed noise image --
# at cs=2.5 for 4 representative weights. If the base model's prior
# dominates at high weight, the blank/noise outputs should converge toward
# the real-rough outputs as weight increases (see the scorer for the
# comparison).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
INFER=scripts/infer_controlnet.py

BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors
CONTROLNET_INIT=$HOME/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime
CAPTION="monochrome line art, manga panel, black and white"
SAMPLES=data/diag_valid5.txt
CS=2.5
OUT_ROOT=results/lora_style_diagnostics_20260913
LOG=logs/lora_style_diagnostics_20260913.log

# weight -> checkpoint dir (representative points across the full sweep)
declare -A CKPT_FOR=(
  [0.02]="checkpoints/controlnet_lora_manga_consistency_w0.02_20260908/final"
  [0.2]="checkpoints/controlnet_lora_manga_consistency_w0.2_20260908/final"
  [0.4]="checkpoints/controlnet_lora_manga_consistency_w0.4_20260914/final"
  [0.5]="checkpoints/controlnet_lora_manga_consistency_w0.5_20260908/final"
)
# condition label -> rough-dir (real rough reused from existing eval outputs,
# not regenerated here)
declare -A COND_DIR=(
  [null_white]="data/diag_null_white"
  [null_black]="data/diag_null_black"
  [null_noise]="data/diag_null_noise"
)

mkdir -p logs "$OUT_ROOT"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] prior-only ablation start: weights [${!CKPT_FOR[*]}] cs=$CS"

for W in "${!CKPT_FOR[@]}"; do
  CKPT="${CKPT_FOR[$W]}"
  for COND in "${!COND_DIR[@]}"; do
    OUT="$OUT_ROOT/outputs/w${W}/${COND}"
    if [ -f "$OUT/.complete" ]; then echo "--- w=$W $COND done, skipping"; continue; fi
    echo "--- w=$W condition=$COND"
    mkdir -p "$OUT"
    "$PY" "$INFER" \
      --sample-list "$SAMPLES" --rough-dir "${COND_DIR[$COND]}" \
      --controlnet-dir "$CONTROLNET_INIT" --controlnet-lora-dir "$CKPT" \
      --base-ckpt "$BASE_CKPT" \
      --caption "$CAPTION" \
      --controlnet-conditioning-scale "$CS" \
      --seed 0 \
      --tag "w${W}_${COND}_cs${CS}" --output-dir "$OUT" 2>&1 | grep -vE "^Loading|it/s\]$"
    touch "$OUT/.complete"
  done
done

echo "=== scoring ==="
"$PY" experiments/score_lora_style_diagnostics_20260913.py

echo "[$(date --iso-8601=seconds)] prior-only ablation complete"
