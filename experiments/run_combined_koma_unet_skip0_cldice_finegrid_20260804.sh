#!/bin/bash
# clDice topology-loss probe on top of the skip-connection-removed direct
# regression family (Shit et al., "clDice", https://arxiv.org/abs/2003.07311,
# implemented this session as lineart/losses.py::soft_cldice_loss +
# scripts/train_i2i_survey.py --cldice-weight/--cldice-iters).
#
# Motivation: skip0 (see run_combined_koma_unet_skip0_finegrid_20260803.sh)
# fixed early-training collapse but did not improve stroke continuity at
# the practically relevant epoch40-60 plateau; checkpoint blending
# (weight/logit average of ep025+ep055) also failed to combine fidelity
# and confidence. None of those directly optimize for stroke connectivity.
# clDice is the first candidate in this project that puts continuity
# directly into the loss function via a differentiable soft-skeleton
# precision/sensitivity term, independent of raw pixel overlap.
#
# Mirrors run_combined_koma_unet_skip0_finegrid_20260803.sh exactly (same
# manifest, same 60-epoch/5-epoch-checkpoint trajectory protocol, same
# eval tool stack) with only --cldice-weight/--cldice-iters added, so
# clDice presence is the sole varied axis against that existing skip0
# trajectory (which is itself already comparable to the with-skip
# baseline finegrid_20260802).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_unet_skip0_cldice_finegrid_20260804
EPOCHS=${EPOCHS:-60}
SAVE_EVERY=${SAVE_EVERY:-5}
CLDICE_WEIGHT=${CLDICE_WEIGHT:-0.3}
CLDICE_ITERS=${CLDICE_ITERS:-10}
TRAIN_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
TRAIN_RAW_ROUGH=dataset/pairs_480/train/rough
TRAIN_LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}
OUT_ROOT=results/${TAG}_trajectory
FIXED_CSV=results/${TAG}_trajectory_fixed_metrics.csv
STAB_CSV=results/${TAG}_trajectory_stability_metrics.csv
MONTAGE=results/compare_${TAG}_trajectory.png

mkdir -p logs results "$CKPT" "$OUT_ROOT"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] combined_koma unet_skip0+cldice fine-grid (5..${EPOCHS} step ${SAVE_EVERY}) start"
echo "tag=$TAG epochs=$EPOCHS save_every=$SAVE_EVERY cldice_weight=$CLDICE_WEIGHT cldice_iters=$CLDICE_ITERS train_list=$TRAIN_LIST"

echo "=== train unet_skip0 + cldice, ${EPOCHS} epochs, checkpoint every ${SAVE_EVERY} ==="
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "$CKPT" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --line-dir "$TRAIN_LINE_DIR" \
  --epochs "$EPOCHS" \
  --workers 0 \
  --model unet_skip0 \
  --lr 8e-5 --pos-weight 3.0 \
  --bce-weight 0.8 --l1-weight 0.2 \
  --shape-weight 0.0 --ink-weight 0.0 \
  --edge-weight 0.5 \
  --cldice-weight "$CLDICE_WEIGHT" --cldice-iters "$CLDICE_ITERS" \
  --save-every "$SAVE_EVERY" \
  --require-cuda

echo "=== infer + evaluate every saved checkpoint ==="
MODEL_NAMES=""
ep=$SAVE_EVERY
while [ "$ep" -le "$EPOCHS" ]; do
  epfmt=$(printf "%03d" "$ep")
  name="${TAG}_ep${epfmt}"
  MODEL_NAMES="$MODEL_NAMES $name"
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "$CKPT/epoch${epfmt}.pth" \
    --file-list "$EVAL_LIST" \
    --rough-dir "$EVAL_RAW_ROUGH" \
    --output-dir "$OUT_ROOT/$name" \
    --autocontrast
  ln -sfn "$(pwd)/$OUT_ROOT/$name" "results/$name"
  ep=$((ep + SAVE_EVERY))
done

echo "=== fixed metrics (all checkpoints) ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models $MODEL_NAMES \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$FIXED_CSV"

echo "=== stroke-stability metrics (all checkpoints) ==="
"$PY" tools/evaluation/evaluate_stroke_stability.py \
  --models $MODEL_NAMES \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$STAB_CSV"

echo "=== montage (subset: every other checkpoint to keep it readable) ==="
MONTAGE_ARGS=""
ep=$SAVE_EVERY
idx=0
while [ "$ep" -le "$EPOCHS" ]; do
  epfmt=$(printf "%03d" "$ep")
  name="${TAG}_ep${epfmt}"
  if [ $((idx % 2)) -eq 0 ]; then
    MONTAGE_ARGS="$MONTAGE_ARGS --model ep${epfmt}=$OUT_ROOT/$name"
  fi
  ep=$((ep + SAVE_EVERY))
  idx=$((idx + 1))
done
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  $MONTAGE_ARGS \
  --output "$MONTAGE"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "save_every=$SAVE_EVERY"
  echo "cldice_weight=$CLDICE_WEIGHT"
  echo "cldice_iters=$CLDICE_ITERS"
  echo "fixed_csv=$FIXED_CSV"
  echo "stability_csv=$STAB_CSV"
  echo "montage=$MONTAGE"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart combined_koma unet_skip0+cldice fine-grid (5..${EPOCHS} step ${SAVE_EVERY}) trajectory complete" \
  "Review $MONTAGE, $FIXED_CSV, $STAB_CSV" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma unet_skip0+cldice fine-grid trajectory complete"
