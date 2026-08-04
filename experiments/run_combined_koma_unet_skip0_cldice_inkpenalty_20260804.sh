#!/bin/bash
# clDice follow-up B: full weight, explicit ink counter-term.
#
# Motivation (doc/work_log.md 2026-08-04 clDice result): the first clDice
# probe (run_combined_koma_unet_skip0_cldice_finegrid_20260804.sh,
# cldice_weight=0.3, ink_weight=0.0) genuinely improved topo_sensitivity/
# hard_cldice but achieved it partly by painting wider/darker (raw
# endpoint_count and ink_ratio both roughly doubled vs skip0-alone at the
# same epochs) rather than tracing more precisely. This run keeps the
# full cldice_weight (so the topology pressure that produced the
# topo_sensitivity gain is untouched) and adds a small ink_loss weight
# (lineart/losses.py::ink_loss, already wired into train_i2i_survey.py
# but unused at 0.0 in every prior recipe here) to directly penalize
# excess ink -- testing whether the "paint it thick" loophole can be
# closed without giving up the topology benefit, as opposed to the
# _lightweight sibling script's "just turn the pressure down" approach.
#
# Mirrors run_combined_koma_unet_skip0_cldice_finegrid_20260804.sh exactly
# except INK_WEIGHT default and TAG.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_unet_skip0_cldice_inkpenalty_20260804
EPOCHS=${EPOCHS:-60}
SAVE_EVERY=${SAVE_EVERY:-5}
CLDICE_WEIGHT=${CLDICE_WEIGHT:-0.3}
CLDICE_ITERS=${CLDICE_ITERS:-10}
INK_WEIGHT=${INK_WEIGHT:-0.05}
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

echo "[$(date --iso-8601=seconds)] combined_koma unet_skip0+cldice(inkpenalty) fine-grid (5..${EPOCHS} step ${SAVE_EVERY}) start"
echo "tag=$TAG epochs=$EPOCHS save_every=$SAVE_EVERY cldice_weight=$CLDICE_WEIGHT cldice_iters=$CLDICE_ITERS ink_weight=$INK_WEIGHT train_list=$TRAIN_LIST"

echo "=== train unet_skip0 + cldice(inkpenalty), ${EPOCHS} epochs, checkpoint every ${SAVE_EVERY} ==="
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
  --shape-weight 0.0 --ink-weight "$INK_WEIGHT" \
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
  echo "ink_weight=$INK_WEIGHT"
  echo "fixed_csv=$FIXED_CSV"
  echo "stability_csv=$STAB_CSV"
  echo "montage=$MONTAGE"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart combined_koma unet_skip0+cldice(inkpenalty) fine-grid (5..${EPOCHS} step ${SAVE_EVERY}) trajectory complete" \
  "Review $MONTAGE, $FIXED_CSV, $STAB_CSV" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma unet_skip0+cldice(inkpenalty) fine-grid trajectory complete"
