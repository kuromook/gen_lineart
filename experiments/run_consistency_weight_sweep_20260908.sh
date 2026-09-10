#!/bin/bash
# Priority #1 from doc/initial_notice.md: only one point of
# (consistency_weight, consistency_max_timestep) has ever been trained --
# (0.1, 200), the current best checkpoint
# (checkpoints/controlnet_lora_manga_consistency_20260904). This isolates
# consistency_weight alone (per the isolation-experiment methodology this
# project uses: one variable at a time), holding consistency_max_timestep=200
# and every other hyperparameter identical to that run.
#
# New grid brackets the known point on both sides:
#   WEIGHTS = 0.02, 0.05, [0.1 known], 0.2, 0.5
# Each full run took ~11h for the known point (2026-09-04, 10:03->20:59), so
# 4 new runs is ~44h -- fits the Mon-Thu multi-day batch window with room for
# Thursday review/analysis or a second-round consistency_max_timestep sweep.
#
# The known 0.1 checkpoint was only ever evaluated at cs=1.0 (the
# infer_controlnet.py default at the time). To compare fairly against the
# new points -- and per this track's own operating rule, never judge on a
# single controlnet_conditioning_scale -- this script's first step
# re-evaluates the existing 0.1 checkpoint at cs 1.0/2.5/3.5 too (inference
# only, no retraining, a few minutes of GPU time).
#
# Resumable across a multi-day run: each weight's iteration is skipped if its
# logs/<tag>.done marker already exists, so this script can be safely
# restarted after an interruption without redoing finished work.
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
WEIGHTS=(${WEIGHTS:-0.02 0.05 0.2 0.5})

SWEEP_LOG=logs/consistency_weight_sweep_20260908.log
SWEEP_DONE=logs/consistency_weight_sweep_20260908.done

mkdir -p logs checkpoints
: > "$SWEEP_LOG"
exec > >(tee -a "$SWEEP_LOG") 2>&1

echo "[$(date --iso-8601=seconds)] consistency_weight sweep start: weights [${WEIGHTS[*]}] (known anchor: 0.1)"

WRITTEN=$(find "$ROUGH_DIR" -type f 2>/dev/null | wc -l)
EXPECTED=$(wc -l < "$FILE_LIST")
if [ "$WRITTEN" -ne "$EXPECTED" ]; then
  echo "ERROR: $ROUGH_DIR has $WRITTEN files, expected $EXPECTED -- preprocessing incomplete, refusing to start" >&2
  exit 1
fi

# --- backfill: re-evaluate the existing weight=0.1 checkpoint at cs 1.0/2.5/3.5 ---
ANCHOR_TAG=controlnet_lora_manga_consistency_w0.1_20260908
ANCHOR_CKPT=../lineart-controlnet-realpairs/checkpoints/controlnet_lora_manga_consistency_20260904/final
ANCHOR_DONE=logs/${ANCHOR_TAG}.done
if [ -f "$ANCHOR_DONE" ]; then
  echo "--- $ANCHOR_TAG already backfilled, skipping"
else
  echo "=== backfill eval: existing weight=0.1 checkpoint, cs sweep (no training) ==="
  for CS in $CS_VALUES; do
    OUT="results/${ANCHOR_TAG}_eval/cs${CS}"
    mkdir -p "$OUT"
    "$PY" scripts/infer_controlnet.py \
      --sample-list data/diag_valid5.txt \
      --rough-dir data/diag_rough_manga_line \
      --controlnet-dir "$CONTROLNET_INIT" \
      --controlnet-lora-dir "$ANCHOR_CKPT" \
      --base-ckpt "$BASE_CKPT" \
      --caption "$CAPTION" \
      --controlnet-conditioning-scale "$CS" \
      --tag "${ANCHOR_TAG}_cs${CS}" --output-dir "$OUT"
  done
  {
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "tag=$ANCHOR_TAG"
    echo "consistency_weight=0.1"
    echo "consistency_max_timestep=$CONSISTENCY_MAX_TIMESTEP"
    echo "checkpoint=$ANCHOR_CKPT"
    echo "eval_dir=results/${ANCHOR_TAG}_eval"
    echo "note=inference-only backfill, not retrained"
  } > "$ANCHOR_DONE"
fi

# --- new weight grid: full train + cs-sweep eval for each ---
for W in "${WEIGHTS[@]}"; do
  TAG=controlnet_lora_manga_consistency_w${W}_20260908
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
    "consistency_weight=$W fine-tune complete" \
    "Review results/${TAG}_eval/" || true

  echo "[$(date --iso-8601=seconds)] --- weight=$W complete ---"
done

echo "=== scoring full sweep ==="
"$PY" experiments/score_consistency_weight_sweep_20260908.py

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "weights=0.1(anchor,backfilled) ${WEIGHTS[*]}"
  echo "consistency_max_timestep=$CONSISTENCY_MAX_TIMESTEP"
  echo "cs_values=$CS_VALUES"
} > "$SWEEP_DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "consistency_weight sweep complete (all weights)" \
  "Review results/consistency_weight_sweep_20260908/scores.csv and montage" || true

echo "[$(date --iso-8601=seconds)] consistency_weight sweep complete"
