#!/bin/bash
# Test whether the "wobble" instability seen in the 100-epoch single-stage
# direct-regression run (combined_koma_direct_unet_100ep_20260801, F1@2px
# 0.187, chamfer 10.19) improves with more training tiles -- same
# architecture/loss recipe, only the training data changes: the densified
# pool (5373 tiles, from experiments/run_koma_dense_retile_20260801.sh's
# loosened-dedup retiling of the same 5 already-extracted koma sources,
# i.e. more overlapping crops of the SAME underlying pages, not genuinely
# new content) instead of the original 1489-tile combined_koma_20260729.
#
# Epoch count: 28, not 100. The 100-epoch run's total step budget was
# 100 * 744 steps/epoch (1489 tiles / batch 2) = 74,400 steps. At batch 2
# the densified set is 5373/2 = 2686 steps/epoch, so 28 epochs ~= 75,200
# steps -- matches the original run's total step budget (and wall-clock
# time, ~8h) so this is a same-training-budget, more-data comparison
# rather than a same-epoch-count one (100 epochs on 3.6x the data would
# have taken ~29h, encroaching on the 2026-08-03 00:00 JST Direction 4 cron).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_direct_unet_dense_28ep_20260801
EPOCHS=${EPOCHS:-28}
TRAIN_LIST=dataset/pairs_480/valid_train_combined_koma_dense_20260801.txt
TRAIN_RAW_ROUGH=dataset/pairs_480/train/rough
TRAIN_LINE_DIR=dataset/pairs_480/train/line_combined_koma_dense_20260801
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough
BASELINE_MODEL_NAME=combined_koma_lucy_mild_msgan_20260729
BASELINE_OUT=results/${BASELINE_MODEL_NAME}
SPARSE_MODEL_NAME=combined_koma_direct_unet_100ep_20260801
SPARSE_OUT=results/${SPARSE_MODEL_NAME}

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

CKPT=checkpoints/${TAG}
OUT=results/${TAG}

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] combined_koma direct unet (densified data, 28ep) start"
echo "tag=$TAG epochs=$EPOCHS train_list=$TRAIN_LIST line_dir=$TRAIN_LINE_DIR"
echo "tile_count=$(wc -l < "$TRAIN_LIST")"

echo "=== train direct unet (densified, no aux, plain BCE+L1+edge, no GAN) ==="
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "$CKPT" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --line-dir "$TRAIN_LINE_DIR" \
  --epochs "$EPOCHS" \
  --workers 0 \
  --model unet \
  --lr 8e-5 --pos-weight 3.0 \
  --bce-weight 0.8 --l1-weight 0.2 \
  --shape-weight 0.0 --ink-weight 0.0 \
  --edge-weight 0.5 \
  --save-every 4 \
  --require-cuda

echo "=== infer direct unet (densified, 28ep) on eval list ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$CKPT/best.pth" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --output-dir "$OUT" \
  --autocontrast

echo "=== montage ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model lucy_mild_msgan="$BASELINE_OUT" \
  --model direct_unet_1489tiles="$SPARSE_OUT" \
  --model direct_unet_5373tiles="$OUT" \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    "$BASELINE_MODEL_NAME" \
    "$SPARSE_MODEL_NAME" \
    "$TAG" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

# CNN+GAN-family work runs on cleanup-refiner, but Monday's cron job
# (scripts/train_controlnet.py) only exists on diffusion-controlnet --
# switch back automatically since the user may be asleep when this
# finishes.
echo "=== switching back to diffusion-controlnet branch ==="
git checkout diffusion-controlnet

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "train_list=$TRAIN_LIST"
  echo "train_line_dir=$TRAIN_LINE_DIR"
  echo "montage=$MONTAGE"
  echo "metrics=$METRICS"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart combined_koma direct unet densified-data (28ep) run complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma direct unet densified-data (28ep) run complete"
