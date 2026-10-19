#!/bin/bash
# Track D (lineart-pair-signal) hypothesis 2: does training loss correlate
# with gt_bsds_f1, or are they decoupled? Never measured before because no
# run in this project ever saved intermediate weights (verified 2026-09-14
# -- every checkpoint on disk is a run's `final` state only). This is a
# diagnostic instrumented replication of the already-validated
# consistency_weight=0.2 config (round 1, same recipe as
# run_consistency_weight_sweep_20260908.sh / _round2_20260914.sh), adding
# --eval-snapshot-steps so intermediate LoRA snapshots are kept instead of
# discarded, then scoring every snapshot against the same 192-tile
# lineart_family holdout at a fixed controlnet_conditioning_scale=2.5
# (round 1's known-good operating point; never judge at cs=1.0 alone --
# project lesson 1).
#
# Explicitly a narrow, user-approved exception to this repo's own
# doc/initial_notice.md "out of scope: training a new generative model" --
# this trains nothing new architecturally and proposes no fix; it exists
# only to produce the intermediate data points hypothesis 2 needs. See
# lineart-pair-signal/doc/work_log.md 2026-09-14.
#
# Estimated cost: ~10.5h training (same as the existing w=0.2 run) + ~3-6h
# inference across 11 snapshots x 192 tiles + scoring. Started Monday, own
# GPU-busy guard, no babysitting required.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python

EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
LR=${LR:-1e-4}
WEIGHT=${WEIGHT:-0.2}
CONSISTENCY_MAX_TIMESTEP=${CONSISTENCY_MAX_TIMESTEP:-200}
EVAL_SNAPSHOT_STEPS=${EVAL_SNAPSHOT_STEPS:-1000}
INFER_CS=${INFER_CS:-2.5}
FILE_LIST=data/train_list.txt
ROUGH_DIR=data/rough_manga_line
LINE_DIR=data/line
CAPTION_CSV=data/captions.csv
CAPTION="monochrome line art, manga panel, black and white"
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors
CONTROLNET_INIT=$HOME/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime

HOLDOUT_LIST=data/holdout_lineart_family.txt
HOLDOUT_ROUGH=data/holdout_lineart_family_rough_manga_line
HOLDOUT_GT=data/holdout_lineart_family_gt_line

TAG=controlnet_lora_manga_consistency_w${WEIGHT}_snapshot_probe_20260914
CKPT=checkpoints/${TAG}
RESULTS_ROOT=results/${TAG}
DONE=logs/${TAG}.done

LOG=logs/${TAG}.log
mkdir -p logs checkpoints results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] loss-quality snapshot probe start (weight=$WEIGHT, eval_snapshot_steps=$EVAL_SNAPSHOT_STEPS)"

if [ -f "$DONE" ]; then
  echo "already complete ($DONE exists), exiting"
  exit 0
fi

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

echo "=== smoke test (6 steps, eval-snapshot-steps=2, scratch dir) ==="
SMOKE_CKPT=${CKPT}_smoke
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
  --eval-snapshot-steps 2 \
  --consistency-weight "$WEIGHT" \
  --consistency-max-timestep "$CONSISTENCY_MAX_TIMESTEP"
if [ ! -f "$SMOKE_CKPT/step_2/pytorch_lora_weights.safetensors" ]; then
  echo "ERROR: smoke test did not produce a step_2 eval snapshot as expected" >&2
  exit 1
fi
echo "=== verifying smoke snapshot is loadable by infer_controlnet.py (1 tile) ==="
SMOKE_OUT=results/${TAG}_smoketest
mkdir -p "$SMOKE_OUT"
head -1 "$HOLDOUT_LIST" > "$SMOKE_OUT/one_tile.txt"
"$PY" scripts/infer_controlnet.py \
  --sample-list "$SMOKE_OUT/one_tile.txt" \
  --rough-dir "$HOLDOUT_ROUGH" \
  --controlnet-dir "$CONTROLNET_INIT" \
  --controlnet-lora-dir "$SMOKE_CKPT/step_2" \
  --base-ckpt "$BASE_CKPT" \
  --caption "$CAPTION" \
  --controlnet-conditioning-scale "$INFER_CS" \
  --tag "${TAG}_smoketest" --output-dir "$SMOKE_OUT"
rm -rf "$SMOKE_CKPT" "$SMOKE_OUT"
echo "smoke test passed, cleaned up scratch checkpoint and output"

echo "=== full instrumented fine-tune ($EPOCHS epochs, weight=$WEIGHT, snapshot every $EVAL_SNAPSHOT_STEPS steps) ==="
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
  --eval-snapshot-steps "$EVAL_SNAPSHOT_STEPS" \
  --consistency-weight "$WEIGHT" \
  --consistency-max-timestep "$CONSISTENCY_MAX_TIMESTEP" \
  --resume-from-checkpoint latest

echo "=== inference: every snapshot x 192-tile lineart_family holdout, cs=$INFER_CS ==="
for SNAP_DIR in "$CKPT"/step_* "$CKPT"/final; do
  [ -d "$SNAP_DIR" ] || continue
  SNAP=$(basename "$SNAP_DIR")
  OUT="$RESULTS_ROOT/$SNAP"
  if [ -f "$OUT/.infer_done" ]; then
    echo "--- $SNAP already inferred, skipping"
    continue
  fi
  echo "[$(date --iso-8601=seconds)] --- inferring $SNAP ---"
  "$PY" scripts/infer_controlnet.py \
    --sample-list "$HOLDOUT_LIST" \
    --rough-dir "$HOLDOUT_ROUGH" \
    --controlnet-dir "$CONTROLNET_INIT" \
    --controlnet-lora-dir "$SNAP_DIR" \
    --base-ckpt "$BASE_CKPT" \
    --caption "$CAPTION" \
    --controlnet-conditioning-scale "$INFER_CS" \
    --tag "${TAG}_${SNAP}" --output-dir "$OUT"
  touch "$OUT/.infer_done"
done

echo "=== scoring: loss-vs-quality correlation (lineart-pair-signal tooling) ==="
PAIR_SIGNAL=/home/sh1/deepl/lineart-pair-signal
"$PY" "$PAIR_SIGNAL/tools/evaluation/score_loss_quality_snapshots.py" \
  --results-root "$RESULTS_ROOT" \
  --gt-dir "$HOLDOUT_GT" \
  --sample-list "$HOLDOUT_LIST" \
  --training-log "$LOG" \
  --loss-window 200 \
  --workers 6 --timeout 120 \
  --output-csv "$RESULTS_ROOT/loss_quality_summary.csv" \
  --per-tile-csv-dir "$RESULTS_ROOT/per_tile"

rm -rf "$CKPT/resume_state"
echo "removed resume_state (no longer needed after successful completion)"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "weight=$WEIGHT"
  echo "eval_snapshot_steps=$EVAL_SNAPSHOT_STEPS"
  echo "infer_cs=$INFER_CS"
  echo "results_root=$RESULTS_ROOT"
  echo "summary_csv=$RESULTS_ROOT/loss_quality_summary.csv"
} > "$DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "Track D hypothesis-2 loss/quality snapshot probe complete" \
  "Review $RESULTS_ROOT/loss_quality_summary.csv" || true

echo "[$(date --iso-8601=seconds)] loss-quality snapshot probe complete"
