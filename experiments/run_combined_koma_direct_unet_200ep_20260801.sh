#!/bin/bash
# 200-epoch version of run_combined_koma_direct_unet_100ep_20260801.sh, on
# the ORIGINAL 1489-tile combined_koma_20260729 list (not the densified
# 5373-tile pool). Tests the per-tile-exposure hypothesis from the
# densified-data result (2026-08-01): that run held total gradient steps
# fixed while growing tile count 3.6x, which cut per-tile exposure from
# 100 to 28 repeats and made the "wobble" (crisp but spatially unstable
# ink, F1@2px 0.187 at 100 epochs) measurably worse (F1@2px 0.150). This
# run tests the other direction: same 1489 tiles, double the exposure
# count (100 -> 200 repeats), to see whether more repetition on the same
# examples stabilizes the wobble into cleaner strokes.
#
# Chosen over doubling epochs on the densified pool because: (a) it
# isolates exposure count as the only variable (content stays fixed), and
# (b) it's the only option that fits before the 2026-08-03 00:00 JST
# Direction 4 cron -- 1489 tiles x 200 epochs ~= 16.1h (290s/epoch x 200),
# vs 5373 tiles x 200 epochs ~= 58.2h (infeasible, only ~25.5h remained
# when this was launched).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_direct_unet_200ep_20260801
EPOCHS=${EPOCHS:-200}
TRAIN_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
TRAIN_RAW_ROUGH=dataset/pairs_480/train/rough
TRAIN_LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough
BASELINE_MODEL_NAME=combined_koma_lucy_mild_msgan_20260729
BASELINE_OUT=results/${BASELINE_MODEL_NAME}
SPARSE100_MODEL_NAME=combined_koma_direct_unet_100ep_20260801
SPARSE100_OUT=results/${SPARSE100_MODEL_NAME}
DENSE28_MODEL_NAME=combined_koma_direct_unet_dense_28ep_20260801
DENSE28_OUT=results/${DENSE28_MODEL_NAME}

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

CKPT=checkpoints/${TAG}
OUT=results/${TAG}

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] combined_koma direct single-stage unet 200-epoch (1489 tiles) start"
echo "tag=$TAG epochs=$EPOCHS train_list=$TRAIN_LIST line_dir=$TRAIN_LINE_DIR"

echo "=== train direct unet (1489 tiles, no aux, plain BCE+L1+edge, no GAN), 200 epochs ==="
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
  --save-every 20 \
  --require-cuda

echo "=== infer direct unet (200ep) on eval list ==="
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
  --model direct_unet_100ep="$SPARSE100_OUT" \
  --model direct_unet_dense_28ep="$DENSE28_OUT" \
  --model direct_unet_200ep="$OUT" \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    "$BASELINE_MODEL_NAME" \
    "$SPARSE100_MODEL_NAME" \
    "$DENSE28_MODEL_NAME" \
    "$TAG" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

# CNN+GAN-family work runs on cleanup-refiner, but Monday's cron job
# (scripts/train_controlnet.py) only exists on diffusion-controlnet --
# switch back automatically since the user may be asleep when this
# finishes (~16h run).
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
  "Lineart combined_koma direct single-stage unet 200-epoch (1489 tiles) run complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma direct single-stage unet 200-epoch run complete"
