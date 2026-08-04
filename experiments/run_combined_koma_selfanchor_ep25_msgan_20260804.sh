#!/bin/bash
# Self-anchor two-stage decomposition (doc/work_log.md "Next Weekend's Plan"
# option 2): freeze an early precise-but-soft direct-regression checkpoint
# (epoch025 of combined_koma_direct_unet_finegrid_20260802, model "unet")
# as a fixed anchor, and train a small correction/binarization head on top
# of it -- reusing the existing ResidualCleanupGenerator
# (aux_logits + bounded_correction) architecture and the already-validated
# cleanup+msgan loss recipe from run_combined_koma_lucy_mild_msgan_20260729.sh
# unchanged, with the anchor's own raw output standing in for the
# atari/ResNet-GAN aux hint (no lucy_mild preprocessing step -- the anchor
# is already a direct rough-to-line rendering, not raw atari texture).
#
# Motivation: the checkpoint/logit blend probe (weight- and logit-averaging
# ep025+ep055 of the same run) did not cleanly recover both fidelity and
# confidence at once (see doc/work_log.md, same session) -- this is the
# heavier fallback option, explicitly training a correction head instead of
# blending in weight/logit space.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_selfanchor_ep25_msgan_20260804
EPOCHS=${EPOCHS:-3}
TRAIN_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
TRAIN_RAW_ROUGH=dataset/pairs_480/train/rough
TRAIN_LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough
ANCHOR_CKPT=/home/sh1/disk/lineart_checkpoints/combined_koma_direct_unet_finegrid_20260802/epoch025.pth

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
STAB_METRICS=results/stroke_stability_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

ANCHOR_TRAIN=results/${TAG}_anchor_train
ANCHOR_EVAL=results/${TAG}_anchor_eval

CKPT=checkpoints/${TAG}
OUT=results/${TAG}

# References already on disk from this session for the montage/metrics:
# combined_koma_direct_unet_finegrid_ep025 (anchor alone, unconfident)
# combined_koma_direct_unet_finegrid_ep055 (confident-but-drifting alone)
EP025_OUT=results/combined_koma_direct_unet_finegrid_ep025
EP055_OUT=results/combined_koma_direct_unet_finegrid_ep055

mkdir -p logs results "$CKPT"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] combined_koma self-anchor (ep025) msgan start"
echo "tag=$TAG epochs=$EPOCHS anchor=$ANCHOR_CKPT train_list=$TRAIN_LIST"

echo "=== materialize anchor aux (train + eval), model=unet ep025 ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$ANCHOR_CKPT" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --output-dir "$ANCHOR_TRAIN" \
  --autocontrast
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$ANCHOR_CKPT" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --output-dir "$ANCHOR_EVAL" \
  --autocontrast

echo "=== train self-anchor cleanup+msgan on koma data ==="
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "$CKPT" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --line-dir "$TRAIN_LINE_DIR" \
  --aux-dir "$ANCHOR_TRAIN" \
  --epochs "$EPOCHS" \
  --workers 0 \
  --model cleanup \
  --gan --multiscale-gan \
  --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
  --bce-weight 0.75 --l1-weight 0.03 \
  --shape-weight 0.08 --ink-weight 0.14 \
  --binary-weight 0.10 --structure-weight 0.04 \
  --adv-weight 0.03 --feature-match-weight 0.08 \
  --require-cuda

echo "=== infer self-anchor cleanup+msgan on eval list ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$CKPT/best.pth" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --aux-dir "$ANCHOR_EVAL" \
  --output-dir "$OUT" \
  --autocontrast

echo "=== montage (anchor alone / confident-alone / self-anchor result / GT) ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model ep025_anchor_alone="$EP025_OUT" \
  --model ep055_confident_alone="$EP055_OUT" \
  --model selfanchor_result="$OUT" \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    combined_koma_direct_unet_finegrid_ep025 \
    combined_koma_direct_unet_finegrid_ep055 \
    "$TAG" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

echo "=== stroke stability metrics ==="
"$PY" tools/evaluation/evaluate_stroke_stability.py \
  --models \
    combined_koma_direct_unet_finegrid_ep025 \
    combined_koma_direct_unet_finegrid_ep055 \
    "$TAG" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$STAB_METRICS"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "anchor_checkpoint=$ANCHOR_CKPT"
  echo "train_list=$TRAIN_LIST"
  echo "montage=$MONTAGE"
  echo "metrics=$METRICS"
  echo "stability_metrics=$STAB_METRICS"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart combined_koma self-anchor (ep025) msgan complete" \
  "Review $MONTAGE, $METRICS, $STAB_METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma self-anchor (ep025) msgan complete"
