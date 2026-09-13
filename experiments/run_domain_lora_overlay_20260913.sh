#!/bin/bash
# Task 4 of the LoRA style-contribution investigation (see doc/work_log.md
# "2026-09-13"). Task 3 confirmed the ControlNet-only ε-loss LoRA does not
# move CLIP style-similarity to the GT corpus at all (flat across the whole
# consistency_weight sweep) -- expected, since that LoRA has no path onto
# the UNet. This task asks the complementary question: does an EXISTING
# UNet-side style LoRA (which *does* touch the frozen SD1.5 UNet's
# attention layers) actually shift style when stacked on top of the current
# best ControlNet checkpoint?
#
# Style LoRA: checkpoints/domain_lora_line_sd15base_sksv2_20260807/final
# (shared foundation `../lineart`, doc/diffusion_fidelity_budget_policy.md
# "現在の採用構成(2026-08-08時点)"), same base_ckpt
# (v1-5-pruned-emaonly.safetensors), rank16/attn-only, adopted scale 1.4,
# adopted caption suffix "sks style, monochrome line art, manga panel,
# black and white" (the "sks style" trigger prepended to exactly this
# track's own existing caption tail).
#
# ControlNet side stays fixed at this track's new best:
# controlnet_lora_manga_consistency_w0.4_20260914 (round 2, near_white_frac
# 0.813), cs=2.5.
#
# 5 cells: the scale=0/no-trigger cell is the existing round-2 w=0.4/cs2.5
# output (reused, not regenerated -- see the scorer). The other 4 combine
# lora-scale {0.7, 1.4} x caption {with, without "sks style,"} to see
# whether the trigger phrase matters independently of scale.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
INFER=scripts/infer_controlnet.py

BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors
CONTROLNET_INIT=$HOME/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime
CONSISTENCY_LORA=checkpoints/controlnet_lora_manga_consistency_w0.4_20260914/final
STYLE_LORA=/home/sh1/deepl/lineart/checkpoints/domain_lora_line_sd15base_sksv2_20260807/final
BASE_CAPTION="monochrome line art, manga panel, black and white"
TRIGGER_CAPTION="sks style, monochrome line art, manga panel, black and white"
SAMPLES=data/diag_valid5.txt
COND_DIR=data/diag_rough_manga_line
CS=2.5
OUT_ROOT=results/domain_lora_overlay_20260913
LOG=logs/domain_lora_overlay_20260913.log

# label | lora-scale | caption
CELLS=(
  "scale07_notrigger|0.7|$BASE_CAPTION"
  "scale07_trigger|0.7|$TRIGGER_CAPTION"
  "scale14_notrigger|1.4|$BASE_CAPTION"
  "scale14_trigger|1.4|$TRIGGER_CAPTION"
)

mkdir -p logs "$OUT_ROOT"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] domain-LoRA overlay start: cs=$CS"

for entry in "${CELLS[@]}"; do
  IFS='|' read -r LABEL SCALE CAPTION <<< "$entry"
  OUT="$OUT_ROOT/outputs/$LABEL"
  if [ -f "$OUT/.complete" ]; then echo "--- $LABEL done, skipping"; continue; fi
  echo "--- $LABEL (lora-scale=$SCALE, caption=\"$CAPTION\")"
  mkdir -p "$OUT"
  "$PY" "$INFER" \
    --sample-list "$SAMPLES" --rough-dir "$COND_DIR" \
    --controlnet-dir "$CONTROLNET_INIT" --controlnet-lora-dir "$CONSISTENCY_LORA" \
    --base-ckpt "$BASE_CKPT" \
    --lora-dir "$STYLE_LORA" --lora-scale "$SCALE" \
    --caption "$CAPTION" \
    --controlnet-conditioning-scale "$CS" \
    --tag "domain_lora_overlay_${LABEL}_cs${CS}" --output-dir "$OUT" 2>&1 | grep -vE "^Loading|it/s\]$"
  touch "$OUT/.complete"
done

echo "=== scoring ==="
"$PY" experiments/score_domain_lora_overlay_20260913.py

echo "[$(date --iso-8601=seconds)] domain-LoRA overlay complete"
