#!/bin/bash
# Track D, hypothesis 5 (scale): is "the pairs teach copying, not placement"
# a property of the objective, or just of 8,467 pairs being too few?
#
# Public ControlNets are trained on hundreds of thousands of pairs; this
# project has 8,467. Hypothesis 5 was deferred while hypotheses 1-4 were
# open. With 1 refuted, 2 showing a nearly flat objective and 3/4 favouring
# "the objective cannot express selection", scale is the one remaining
# alternative explanation, so it gets measured.
#
# Design: everything fixed except the number of distinct training pairs.
# Same lineart_coarse conditions, same 2,290 steps (so the number of updates
# is identical and only the data seen per update differs), same recipe as
# run_h34_alignment_probe_20260915.sh. Nested subsets so the curve is not
# confounded by which pairs are in play:
#
#   460  subset of  1,837  subset of  8,467
#
# The 1,837 point already exists -- it is that probe's `control` arm (random,
# matched on source x GT-ink quintile). This script runs the 460 and 8,467
# points and leaves the comparison to analysis.
#
# The measure that decides it is NOT f1: it is the share of GT stroke length
# drawn where the conditioning map lacks that stroke
# (`tools/evaluation/output_vs_condition_proximity.py`, column
# "GT drawn: cond lacks it"). At 1,837 pairs it sat at ~0.10 in both arms of
# the alignment probe. If that column rises with pair count, hypothesis 4
# weakens; if it stays flat while the model merely copies its condition
# better, hypothesis 4 holds at this scale too.
#
# Cost: 2 arms x (~2.5h training + ~1.9h inference) + scoring ~= 9-10h.
set -euo pipefail

TRACK_D=/home/sh1/deepl/lineart-pair-signal
TRACK_A=/home/sh1/deepl/lineart-controlnet-sd15-refine
PY=/home/sh1/deepl/lineart/venv/bin/python
export PYTHONUNBUFFERED=1

STEPS=2290           # same number of updates as the alignment probe
EVAL_SNAPSHOT_STEPS=500
LORA_RANK=16
LR=1e-4
WEIGHT=0.2
CONSISTENCY_MAX_TIMESTEP=200
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
MID_LIST=$LISTS/list_control_random_matched_sources_gtink.txt   # the existing 1,837 point
SMALL_LIST=$LISTS/list_scale_460.txt
FULL_LIST=$LISTS/list_scale_8467.txt

TAG=scale_curve_20260916
RESULTS=$TRACK_D/results/$TAG
LOGDIR=$TRACK_D/logs
DONE=$LOGDIR/$TAG.done
mkdir -p "$LOGDIR" "$RESULTS"
exec > >(tee -a "$LOGDIR/$TAG.log") 2>&1

cd "$TRACK_A"
echo "[$(date --iso-8601=seconds)] $TAG start (steps=$STEPS, condition=lineart_coarse)"
[ -f "$DONE" ] && { echo "already complete"; exit 0; }

# nested subsets, deterministic
"$PY" - "$MID_LIST" "$SMALL_LIST" "$FULL_LIST" "$TRACK_A/data/train_list.txt" <<'PY'
import sys, numpy as np
mid, small, full, train = sys.argv[1:5]
names = [l.strip() for l in open(mid) if l.strip()]
rng = np.random.default_rng(20260916)
idx = rng.choice(len(names), size=460, replace=False)
open(small, "w").write("\n".join(sorted(names[i] for i in idx)) + "\n")
allnames = [l.strip() for l in open(train) if l.strip()]
open(full, "w").write("\n".join(sorted(allnames)) + "\n")
print(f"subsets: 460 ⊂ {len(names)} ⊂ {len(allnames)}")
PY

declare -A ARM_LIST=( [scale_460]=$SMALL_LIST [scale_8467]=$FULL_LIST )
ARMS=(scale_460 scale_8467)

echo "[$(date --iso-8601=seconds)] checking GPU is free"
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${USED:-0}" -lt 500 ] && break
  echo "  GPU busy (${USED} MiB), waiting 120s"; sleep 120
done

for arm in "${ARMS[@]}"; do
  CKPT=$TRACK_D/checkpoints/${TAG}_$arm
  ARMRES=$RESULTS/$arm
  TRAINLOG=$LOGDIR/${TAG}_${arm}_train.log
  mkdir -p "$ARMRES"
  if [ ! -d "$CKPT/final" ]; then
    echo "[$(date --iso-8601=seconds)] === train $arm ($(wc -l < "${ARM_LIST[$arm]}") pairs, $STEPS steps) ==="
    "$PY" scripts/train_controlnet_consistency.py \
      --file-list "${ARM_LIST[$arm]}" --rough-dir "$TRAIN_ROUGH" --line-dir "$TRAIN_LINE" \
      --base-ckpt "$BASE_CKPT" --controlnet-init "$CONTROLNET_INIT" --controlnet-lora-rank "$LORA_RANK" \
      --caption-csv "$CAPTION_CSV" --output-dir "$CKPT" --lr "$LR" \
      --max-train-steps "$STEPS" --save-steps 500 --log-steps 50 \
      --eval-snapshot-steps "$EVAL_SNAPSHOT_STEPS" \
      --consistency-weight "$WEIGHT" --consistency-max-timestep "$CONSISTENCY_MAX_TIMESTEP" \
      --resume-from-checkpoint latest 2>&1 | tee -a "$TRAINLOG"
  fi

  echo "[$(date --iso-8601=seconds)] === infer $arm ==="
  for SNAP_DIR in "$CKPT"/step_*; do
    SNAP=$(basename "$SNAP_DIR"); OUT=$ARMRES/$SNAP
    [ -f "$OUT/.infer_done" ] && continue
    mkdir -p "$OUT"
    "$PY" scripts/infer_controlnet.py \
      --sample-list "$HOLDOUT_LIST" --rough-dir "$HOLDOUT_COND" \
      --controlnet-dir "$CONTROLNET_INIT" --controlnet-lora-dir "$SNAP_DIR" \
      --base-ckpt "$BASE_CKPT" --caption "$INFER_CAPTION" \
      --controlnet-conditioning-scale "$INFER_CS" --tag "${TAG}_${arm}_${SNAP}" --output-dir "$OUT" 2>&1 | grep -vE "^Loading|Fetching|it/s\]$"
    touch "$OUT/.infer_done"
  done

  echo "[$(date --iso-8601=seconds)] === score $arm ==="
  "$PY" "$TRACK_D/tools/evaluation/score_loss_quality_snapshots.py" \
    --results-root "$ARMRES" --gt-dir "$HOLDOUT_GT" --sample-list "$HOLDOUT_LIST" \
    --training-log "$TRAINLOG" --loss-window 200 --workers 6 --timeout 120 \
    --output-csv "$ARMRES/loss_quality_summary.csv" --per-tile-csv-dir "$ARMRES/per_tile"

  echo "[$(date --iso-8601=seconds)] === proximity $arm (the deciding measure) ==="
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" "$PY" "$TRACK_D/tools/evaluation/output_vs_condition_proximity.py" \
    --results-root "$ARMRES" --gt-dir "$HOLDOUT_GT" --cond-dir "$HOLDOUT_COND" \
    --sample-list "$HOLDOUT_LIST" 2>&1 | tee "$ARMRES/output_vs_condition_proximity.txt"

  rm -rf "$CKPT/resume_state"
done

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "arms=${ARMS[*]} (plus the existing 1,837-pair control arm of h34_alignment_probe_20260915)"
  echo "steps=$STEPS condition=lineart_coarse cs=$INFER_CS"
  echo "results=$RESULTS"
} > "$DONE"
/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh "Track D scale curve complete" "Review $RESULTS" || true
echo "[$(date --iso-8601=seconds)] $TAG complete"
