#!/bin/bash
# Round 2 of the consistency_weight sweep (see doc/work_log.md
# "2026-09-08/10 (Track A)" for round 1). Round 1 (0.02/0.05/0.1/0.2/0.5)
# found weight=0.2 the best so far on near_white_frac at cs2.5 (0.779), but
# weight=0.5 only dipped slightly (0.753) rather than collapsing -- meaning
# the region between 0.2 and 0.5 is unexplored and may still hold a better
# point than 0.2 itself. This round fills that gap rather than narrowing
# tightly around 0.2:
#
#   round 1: 0.02   0.05   0.1  [0.2]        0.5
#   round 2:                0.15    0.25 0.3    0.4
#
# consistency_max_timestep stays fixed at 200 (same as round 1 and the
# original known point), isolating consistency_weight alone per this
# project's methodology. No backfill step needed this round: weight=0.2
# (this round's reference point) already has full cs 1.0/2.5/3.5 eval data
# under results/controlnet_lora_manga_consistency_w0.2_20260908_eval/ from
# round 1 -- the combined scorer reads both rounds' output directories
# directly (see score_consistency_weight_sweep_round2_20260914.py).
#
# Scheduled to start 2026-09-14 00:00 JST via `systemd-run --user
# --on-calendar` (see the one-shot timer set up alongside this script) so
# it can launch unattended over the weekend. Includes its own GPU-busy
# guard at the top since nothing else is expected to be babysitting this
# machine before it fires.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python

EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
LR=${LR:-1e-4}
CONSISTENCY_MAX_TIMESTEP=${CONSISTENCY_MAX_TIMESTEP:-200}
FILE_LIST=data/train_list.txt
ROUGH_DIR=data/rough_manga_line
LINE_DIR=data/line
CAPTION_CSV=data/captions.csv
CAPTION="monochrome line art, manga panel, black and white"
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors
CONTROLNET_INIT=$HOME/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime
CS_VALUES=${CS_VALUES:-"1.0 2.5 3.5"}
WEIGHTS=(${WEIGHTS:-0.15 0.25 0.3 0.4})
DATE_TAG=20260914

SWEEP_LOG=logs/consistency_weight_sweep_round2_20260914.log
SWEEP_DONE=logs/consistency_weight_sweep_round2_20260914.done

mkdir -p logs checkpoints
: > "$SWEEP_LOG"
exec > >(tee -a "$SWEEP_LOG") 2>&1

echo "[$(date --iso-8601=seconds)] consistency_weight sweep round 2 start: weights [${WEIGHTS[*]}]"

# --- guard: don't start if another GPU job is already running ---
echo "[$(date --iso-8601=seconds)] checking GPU is free before starting"
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${USED:-0}" -lt 500 ]; then
    break
  fi
  echo "  GPU busy (${USED} MiB used), waiting 60s"
  sleep 60
done
echo "[$(date --iso-8601=seconds)] GPU free, proceeding"

WRITTEN=$(find "$ROUGH_DIR" -type f 2>/dev/null | wc -l)
EXPECTED=$(wc -l < "$FILE_LIST")
if [ "$WRITTEN" -ne "$EXPECTED" ]; then
  echo "ERROR: $ROUGH_DIR has $WRITTEN files, expected $EXPECTED -- preprocessing incomplete, refusing to start" >&2
  exit 1
fi

for W in "${WEIGHTS[@]}"; do
  TAG=controlnet_lora_manga_consistency_w${W}_${DATE_TAG}
  DONE=logs/${TAG}.done
  CKPT=checkpoints/${TAG}

  if [ -f "$DONE" ]; then
    echo "--- $TAG already complete, skipping"
    continue
  fi

  echo "[$(date --iso-8601=seconds)] --- weight=$W start ---"

  echo "=== smoke test (6 steps, scratch dir) ==="
  SMOKE_CKPT=checkpoints/${TAG}_smoke
  rm -rf "$SMOKE_CKPT"
  "$PY" scripts/train_controlnet_consistency.py \
    --file-list "$FILE_LIST" \
    --rough-dir "$ROUGH_DIR" \
    --line-dir "$LINE_DIR" \
    --base-ckpt "$BASE_CKPT" \
    --controlnet-init "$CONTROLNET_INIT" \
    --controlnet-lora-rank "$LORA_RANK" \
    --caption-csv "$CAPTION_CSV" \
    --output-dir "$SMOKE_CKPT" \
    --max-train-steps 6 \
    --log-steps 1 \
    --consistency-weight "$W" \
    --consistency-max-timestep "$CONSISTENCY_MAX_TIMESTEP"
  echo "smoke test passed, cleaning up scratch checkpoint"
  rm -rf "$SMOKE_CKPT"

  echo "=== full fine-tune ($EPOCHS epochs, consistency_weight=$W) ==="
  "$PY" scripts/train_controlnet_consistency.py \
    --file-list "$FILE_LIST" \
    --rough-dir "$ROUGH_DIR" \
    --line-dir "$LINE_DIR" \
    --base-ckpt "$BASE_CKPT" \
    --controlnet-init "$CONTROLNET_INIT" \
    --controlnet-lora-rank "$LORA_RANK" \
    --caption-csv "$CAPTION_CSV" \
    --output-dir "$CKPT" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --save-steps 500 \
    --log-steps 50 \
    --consistency-weight "$W" \
    --consistency-max-timestep "$CONSISTENCY_MAX_TIMESTEP" \
    --resume-from-checkpoint latest

  echo "=== eval (diag5, cs sweep [$CS_VALUES]) ==="
  for CS in $CS_VALUES; do
    OUT="results/${TAG}_eval/cs${CS}"
    mkdir -p "$OUT"
    "$PY" scripts/infer_controlnet.py \
      --sample-list data/diag_valid5.txt \
      --rough-dir data/diag_rough_manga_line \
      --controlnet-dir "$CONTROLNET_INIT" \
      --controlnet-lora-dir "$CKPT/final" \
      --base-ckpt "$BASE_CKPT" \
      --caption "$CAPTION" \
      --controlnet-conditioning-scale "$CS" \
      --tag "${TAG}_cs${CS}" --output-dir "$OUT"
  done

  rm -rf "$CKPT/resume_state"
  echo "removed resume_state (no longer needed after successful completion)"

  {
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "tag=$TAG"
    echo "epochs=$EPOCHS"
    echo "lora_rank=$LORA_RANK"
    echo "lr=$LR"
    echo "consistency_weight=$W"
    echo "consistency_max_timestep=$CONSISTENCY_MAX_TIMESTEP"
    echo "file_list=$FILE_LIST"
    echo "rough_dir=$ROUGH_DIR"
    echo "line_dir=$LINE_DIR"
    echo "caption_csv=$CAPTION_CSV"
    echo "controlnet_init=$CONTROLNET_INIT"
    echo "checkpoint=$CKPT/final"
    echo "eval_dir=results/${TAG}_eval"
  } > "$DONE"

  /home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
    "consistency_weight=$W (round 2) fine-tune complete" \
    "Review results/${TAG}_eval/" || true

  echo "[$(date --iso-8601=seconds)] --- weight=$W complete ---"
done

echo "=== scoring full round-2 sweep (merged with round 1) ==="
"$PY" experiments/score_consistency_weight_sweep_round2_20260914.py

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "round1_weights=0.02 0.05 0.1 0.2 0.5"
  echo "round2_weights=${WEIGHTS[*]}"
  echo "consistency_max_timestep=$CONSISTENCY_MAX_TIMESTEP"
  echo "cs_values=$CS_VALUES"
} > "$SWEEP_DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "consistency_weight sweep round 2 complete" \
  "Review results/consistency_weight_sweep_round2_20260914/scores.csv and montage" || true

echo "[$(date --iso-8601=seconds)] consistency_weight sweep round 2 complete"
