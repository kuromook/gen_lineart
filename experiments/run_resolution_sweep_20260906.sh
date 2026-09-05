#!/bin/bash
# Track B first experiment (doc/initial_notice.md): put SDXL on a fair
# footing before judging its conditioning fidelity.
#
# Every SDXL number this project has -- including "gt_bsds_f1 0.0945, worst
# of 11 models" and "cs makes it worse" -- was produced at 512x512, both in
# training and inference, from CLI defaults inherited from the SD1.5-era
# scripts. animagine-xl-3.1 is a 1024-native SDXL, so those numbers may be
# measuring the resolution mismatch rather than SDXL.
#
# This sweep separates three things that were previously confounded:
#   * resolution      512 / 768 / 1024
#   * conditioning    cs 0.5 / 1.0 / 2.0 (low side included -- SDXL is the
#                     one family that got *worse* as cs rose)
#   * the LoRA itself  each ControlNet is run both with our 512-trained LoRA
#                     and bare. The bare column is the control that says
#                     whether the Eugeoter ControlNets can follow this
#                     conditioning at all (next-step #3 in the briefing) --
#                     without it, a bad score cannot be attributed to either
#                     the base ControlNet or our fine-tune.
#
# Inference-only: no training here, so this is a weekend-sized job (~75 min)
# whose result decides what Monday's multi-day 1024 training run should be.
#
# --cpu-offload is applied uniformly to every cell. At 1024 it is mandatory
# (the fully-resident fp16 pipeline peaks at 11.1GB and OOMs); applying it
# everywhere keeps resolution from being confounded with the memory policy.
# The two *_nooffload cells at 512/cs1.0 are the control for that choice and
# are simultaneously the historical anchor: they must reproduce
# gt_bsds_f1 0.0945 (anime_lora) and 0.1226 (manga_lora) from
# doc/track_controlnet_realpairs_work_log.md's 11-model table.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python
INFER=scripts/infer_controlnet_sdxl.py

SDXL_BASE=/home/sh1/.cache/huggingface/hub/models--cagliostrolab--animagine-xl-3.1/snapshots/483f0c322568ed13697ed01dd0be07204746d12b
CN_ANIME=/home/sh1/.cache/huggingface/hub/models--Eugeoter--noob-sdxl-controlnet-lineart_anime/snapshots/61ed2d40710b32a5a1c9873f7dec89ff0af9f2a4
CN_MANGA=/home/sh1/.cache/huggingface/hub/models--Eugeoter--noob-sdxl-controlnet-manga_line/snapshots/bc7619904de6489ba7e171cf27a80f0e12822943

CAPTION="monochrome line art, manga panel, black and white"
SAMPLES=data/diag_valid5.txt

RESOLUTIONS=${RESOLUTIONS:-"512 768 1024"}
SCALES=${SCALES:-"0.5 1.0 2.0"}
OUT_ROOT=results/resolution_sweep_20260906
LOG=logs/resolution_sweep_20260906.log
DONE=logs/resolution_sweep_20260906.done

# label | controlnet_init | lora_dir ("-" for the bare base) | rough_dir
MODELS=(
  "anime_lora|$CN_ANIME|checkpoints/controlnet_lora_sdxl_20260829/final|data/diag_rough_lineart_coarse"
  "manga_lora|$CN_MANGA|checkpoints/controlnet_lora_sdxl_manga_20260830/final|data/diag_rough_manga_line"
  "anime_base|$CN_ANIME|-|data/diag_rough_lineart_coarse"
  "manga_base|$CN_MANGA|-|data/diag_rough_manga_line"
)

mkdir -p logs "$OUT_ROOT"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] resolution sweep start: ${#MODELS[@]} models x [$RESOLUTIONS] x cs [$SCALES]"

run_cell() {
  local out_dir=$1 cn_init=$2 lora=$3 rough=$4 res=$5 cs=$6 offload=$7
  if [ -f "$out_dir/.complete" ]; then
    echo "--- $out_dir already done, skipping"
    return
  fi
  mkdir -p "$out_dir"
  local args=(
    --sample-list "$SAMPLES" --rough-dir "$rough"
    --controlnet-dir "$cn_init" --base-ckpt "$SDXL_BASE"
    --caption "$CAPTION" --resolution "$res"
    --controlnet-conditioning-scale "$cs"
    --tag "$(basename "$out_dir")" --output-dir "$out_dir"
  )
  [ "$lora" != "-" ] && args+=(--controlnet-lora-dir "$lora")
  [ "$offload" = "yes" ] && args+=(--cpu-offload)
  "$PY" "$INFER" "${args[@]}" 2>&1 | grep -vE "^Loading|it/s\]$"
  touch "$out_dir/.complete"
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r LABEL CN_INIT LORA ROUGH <<< "$entry"
  if [ "$LORA" != "-" ] && [ ! -f "$LORA/pytorch_lora_weights.safetensors" ]; then
    echo "WARNING: $LABEL has no $LORA/pytorch_lora_weights.safetensors -- skipping" >&2
    continue
  fi
  for RES in $RESOLUTIONS; do
    for CS in $SCALES; do
      echo "--- $LABEL res=$RES cs=$CS"
      run_cell "$OUT_ROOT/outputs/${LABEL}/res${RES}_cs${CS}" \
        "$CN_INIT" "$LORA" "$ROUGH" "$RES" "$CS" yes
    done
  done
done

echo "=== offload control / historical anchor (512, cs1.0, no --cpu-offload) ==="
for entry in "${MODELS[@]}"; do
  IFS='|' read -r LABEL CN_INIT LORA ROUGH <<< "$entry"
  [ "$LORA" = "-" ] && continue
  echo "--- ${LABEL}_nooffload res=512 cs=1.0"
  run_cell "$OUT_ROOT/outputs/${LABEL}_nooffload/res512_cs1.0" \
    "$CN_INIT" "$LORA" "$ROUGH" 512 1.0 no
done

echo "=== scoring ==="
"$PY" experiments/score_resolution_sweep_20260906.py

echo "=== montages ==="
"$PY" experiments/montage_resolution_sweep_20260906.py

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "resolutions=$RESOLUTIONS"
  echo "scales=$SCALES"
  echo "models=${#MODELS[@]}"
  echo "output_dir=$OUT_ROOT"
} > "$DONE"

echo "[$(date --iso-8601=seconds)] resolution sweep complete"
