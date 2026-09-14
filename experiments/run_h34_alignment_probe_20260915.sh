#!/bin/bash
# Track D, hypotheses 3/4 step (b): does training on well-aligned pairs make
# the model learn stroke placement, or does it still just inherit the
# condition map's strokes?
#
#   hypothesis 3 (pairs too loosely aligned to learn placement from) predicts
#     the aligned arm moves its strokes toward GT (more output near GT only,
#     higher recall at the same precision, stroke set drifting toward GT);
#   hypothesis 4 (a generative objective cannot express selection) predicts
#     both arms behave alike: strokes inherited from the condition, training
#     changing tone and weight.
#
# Two arms, identical except for which 1,837 training pairs they see:
#   aligned: raw-rough/GT chance-corrected alignment >= 0.5
#   control: random pairs matched on data source x GT-ink quintile, seed 20260915
# (both lists from tools/evaluation/pair_alignment_strata.py, see
# doc/work_log.md "Hypotheses 3/4 Step (a)"). Condition preprocessor:
# lineart_coarse, not manga_line -- manga_line leaves 19% of training
# conditions nearly empty and carries 12% of GT strokes on training pairs vs
# 30% on the holdout, a confound this experiment must not inherit.
#
# Recipe otherwise identical to the w=0.2 snapshot probe
# (../lineart-controlnet-sd15-refine/experiments/run_loss_quality_snapshot_probe_20260914.sh):
# v1-5-pruned-emaonly + control_v11p_sd15s2_lineart_anime init (the same init
# Track A's earlier lineart_coarse run used), LoRA rank 16, lr 1e-4,
# consistency_weight 0.2 below t=200, 10 epochs (2,290 steps at batch 2 x
# accumulation 4). Snapshots every 500 steps. Per snapshot: 192-tile
# lineart_family holdout inference on lineart_coarse conditions at cs 2.5,
# scored. After both arms: stroke-level churn and fixed-t validation loss.
#
# Reads Track A's data and scripts; writes only under this tree
# (checkpoints/, results/, logs/ are gitignored). User-approved training,
# 2026-09-15 (condition, control and length chosen by the user).
set -euo pipefail

TRACK_D=/home/sh1/deepl/lineart-pair-signal
TRACK_A=/home/sh1/deepl/lineart-controlnet-sd15-refine
PY=/home/sh1/deepl/lineart/venv/bin/python
export PYTHONUNBUFFERED=1

EPOCHS=${EPOCHS:-10}
LORA_RANK=16
LR=1e-4
WEIGHT=0.2
CONSISTENCY_MAX_TIMESTEP=200
EVAL_SNAPSHOT_STEPS=${EVAL_SNAPSHOT_STEPS:-500}
INFER_CS=2.5
BASE_CKPT=$HOME/disk/checkpoint/Stable-diffusion/v1-5-pruned-emaonly.safetensors
CONTROLNET_INIT=$HOME/disk/checkpoint/ControlNet/control_v11p_sd15s2_lineart_anime
TRAIN_ROUGH=$TRACK_A/data/rough_lineart_coarse
TRAIN_LINE=$TRACK_A/data/line
CAPTION_CSV=$TRACK_A/data/captions.csv
INFER_CAPTION="monochrome line art, manga panel, black and white"

HOLDOUT_LIST=$TRACK_A/data/holdout_lineart_family.txt
HOLDOUT_GT=$TRACK_A/data/holdout_lineart_family_gt_line
HOLDOUT_COND=/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning

LISTS=$TRACK_D/results/pair_alignment_strata_20260915
declare -A ARM_LIST=(
  [aligned]=$LISTS/list_aligned_rough_ge0.5.txt
  [control]=$LISTS/list_control_random_matched_sources_gtink.txt
)
ARMS=(aligned control)

TAG=h34_alignment_probe_20260915
RESULTS=$TRACK_D/results/$TAG
LOGDIR=$TRACK_D/logs
DONE=$LOGDIR/$TAG.done
mkdir -p "$LOGDIR" "$RESULTS" "$TRACK_D/checkpoints"
MAINLOG=$LOGDIR/$TAG.log
exec > >(tee -a "$MAINLOG") 2>&1

cd "$TRACK_A"
echo "[$(date --iso-8601=seconds)] $TAG start (arms: ${ARMS[*]}, epochs=$EPOCHS, snapshots every $EVAL_SNAPSHOT_STEPS)"
[ -f "$DONE" ] && { echo "already complete"; exit 0; }

for arm in "${ARMS[@]}"; do
  n=$(wc -l < "${ARM_LIST[$arm]}")
  missing=$(while read -r t; do [ -f "$TRAIN_ROUGH/$t" ] && [ -f "$TRAIN_LINE/$t" ] || echo "$t"; done < "${ARM_LIST[$arm]}" | wc -l)
  echo "arm $arm: $n pairs, $missing missing files"
  [ "$missing" -eq 0 ] || { echo "ERROR: missing training files for $arm" >&2; exit 1; }
done
[ "$(grep -cxFf <(ls "$HOLDOUT_COND") "$HOLDOUT_LIST")" -eq "$(wc -l < "$HOLDOUT_LIST")" ] || { echo "ERROR: holdout lineart_coarse conditions incomplete" >&2; exit 1; }

echo "[$(date --iso-8601=seconds)] checking GPU is free"
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${USED:-0}" -lt 500 ] && break
  echo "  GPU busy (${USED} MiB), waiting 60s"; sleep 60
done

train_arm() {
  local arm=$1 ckpt=$2 log=$3 extra=("${@:4}")
  "$PY" scripts/train_controlnet_consistency.py \
    --file-list "${ARM_LIST[$arm]}" --rough-dir "$TRAIN_ROUGH" --line-dir "$TRAIN_LINE" \
    --base-ckpt "$BASE_CKPT" --controlnet-init "$CONTROLNET_INIT" --controlnet-lora-rank "$LORA_RANK" \
    --caption-csv "$CAPTION_CSV" --output-dir "$ckpt" --lr "$LR" \
    --consistency-weight "$WEIGHT" --consistency-max-timestep "$CONSISTENCY_MAX_TIMESTEP" \
    "${extra[@]}" 2>&1 | tee -a "$log"
}

infer() {
  local lora=$1 out=$2 list=$3
  mkdir -p "$out"
  "$PY" scripts/infer_controlnet.py \
    --sample-list "$list" --rough-dir "$HOLDOUT_COND" \
    --controlnet-dir "$CONTROLNET_INIT" --controlnet-lora-dir "$lora" \
    --base-ckpt "$BASE_CKPT" --caption "$INFER_CAPTION" \
    --controlnet-conditioning-scale "$INFER_CS" --tag "$(basename "$out")" --output-dir "$out" 2>&1 | grep -vE "^Loading|Fetching|it/s\]$"
}

echo "=== smoke test: 6 steps on the aligned list, snapshot at step 2, loaded by inference on one holdout tile ==="
SMOKE_CKPT=$TRACK_D/checkpoints/${TAG}_smoke
SMOKE_OUT=$RESULTS/_smoke
rm -rf "$SMOKE_CKPT" "$SMOKE_OUT"; mkdir -p "$SMOKE_OUT"
train_arm aligned "$SMOKE_CKPT" "$SMOKE_OUT/train.log" --max-train-steps 6 --log-steps 1 --eval-snapshot-steps 2
[ -f "$SMOKE_CKPT/step_2/pytorch_lora_weights.safetensors" ] || { echo "ERROR: smoke produced no step_2 snapshot" >&2; exit 1; }
head -1 "$HOLDOUT_LIST" > "$SMOKE_OUT/one.txt"
infer "$SMOKE_CKPT/step_2" "$SMOKE_OUT/step_2" "$SMOKE_OUT/one.txt"
[ "$(ls "$SMOKE_OUT/step_2" | grep -c _out.png)" -eq 1 ] || { echo "ERROR: smoke inference wrote no output" >&2; exit 1; }
rm -rf "$SMOKE_CKPT" "$SMOKE_OUT"
echo "smoke test passed"

for arm in "${ARMS[@]}"; do
  CKPT=$TRACK_D/checkpoints/${TAG}_$arm
  ARMRES=$RESULTS/$arm
  TRAINLOG=$LOGDIR/${TAG}_${arm}_train.log
  mkdir -p "$ARMRES"
  if [ ! -d "$CKPT/final" ]; then
    echo "[$(date --iso-8601=seconds)] === train $arm ==="
    train_arm "$arm" "$CKPT" "$TRAINLOG" --epochs "$EPOCHS" --save-steps 500 --log-steps 50 \
      --eval-snapshot-steps "$EVAL_SNAPSHOT_STEPS" --resume-from-checkpoint latest
  fi
  echo "[$(date --iso-8601=seconds)] === infer $arm snapshots on 192 holdout tiles ==="
  for SNAP_DIR in "$CKPT"/step_*; do
    SNAP=$(basename "$SNAP_DIR"); OUT=$ARMRES/$SNAP
    [ -f "$OUT/.infer_done" ] && continue
    infer "$SNAP_DIR" "$OUT" "$HOLDOUT_LIST"
    touch "$OUT/.infer_done"
  done
  echo "[$(date --iso-8601=seconds)] === score $arm ==="
  "$PY" "$TRACK_D/tools/evaluation/score_loss_quality_snapshots.py" \
    --results-root "$ARMRES" --gt-dir "$HOLDOUT_GT" --sample-list "$HOLDOUT_LIST" \
    --training-log "$TRAINLOG" --loss-window 200 --workers 6 --timeout 120 \
    --output-csv "$ARMRES/loss_quality_summary.csv" --per-tile-csv-dir "$ARMRES/per_tile"
  rm -rf "$CKPT/resume_state"
done

cd "$TRACK_D"
for arm in "${ARMS[@]}"; do
  CKPT=$TRACK_D/checkpoints/${TAG}_$arm
  ARMRES=$RESULTS/$arm
  echo "[$(date --iso-8601=seconds)] === stroke churn $arm ==="
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" "$PY" tools/evaluation/stroke_churn.py --no-final \
    --results-root "$ARMRES" --gt-dir "$HOLDOUT_GT" --sample-list "$HOLDOUT_LIST" \
    --output-dir "$ARMRES/stroke_churn" > "$ARMRES/stroke_churn.log" 2>&1
  echo "[$(date --iso-8601=seconds)] === fixed-t validation loss $arm ==="
  "$PY" tools/evaluation/fixed_t_validation_loss.py \
    --ckpt-root "$CKPT" --results-root "$ARMRES" \
    --sample-list "$HOLDOUT_LIST" --gt-dir "$HOLDOUT_GT" --rough-dir "$HOLDOUT_COND" \
    --output-dir "$ARMRES/fixed_t" > "$ARMRES/fixed_t.log" 2>&1
done

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "arms=${ARMS[*]}"
  echo "epochs=$EPOCHS eval_snapshot_steps=$EVAL_SNAPSHOT_STEPS cs=$INFER_CS condition=lineart_coarse"
  echo "results=$RESULTS"
} > "$DONE"
/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh "Track D h3/h4 alignment probe complete" "Review $RESULTS" || true
echo "[$(date --iso-8601=seconds)] $TAG complete"
