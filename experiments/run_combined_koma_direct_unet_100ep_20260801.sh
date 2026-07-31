#!/bin/bash
# 100-epoch version of run_combined_koma_direct_unet_20260801.sh, giving
# the single-stage direct-regression architecture (--model unet, no
# atari/aux hint) a fair shot: the 3-epoch version was inconclusive
# (near-blank output, F1@2px 0.012) but training loss was still dropping
# steadily with no sign of convergence, unlike the atari+cleanup family
# which starts "warm" from a pretrained atari baseline and only learns a
# small correction. notebooks/gen_lineart.ipynb's original model trained
# 50 epochs (loss 0.594 -> 0.207); this budgets 100 for headroom.
#
# Estimated ~4.8 min/epoch (measured from the 3-epoch run: 870s/3) ->
# ~8 hours total. Runs unattended overnight; --save-every 10 gives
# periodic checkpoints (epoch010.pth, epoch020.pth, ...) as a crash
# safety net (weight-only resume via --resume-generator if needed, not a
# full train-state resume like train_controlnet.py's --resume-from-checkpoint).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_direct_unet_100ep_20260801
EPOCHS=${EPOCHS:-100}
TRAIN_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
TRAIN_RAW_ROUGH=dataset/pairs_480/train/rough
TRAIN_LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough
BASELINE_MODEL_NAME=combined_koma_lucy_mild_msgan_20260729
BASELINE_OUT=results/${BASELINE_MODEL_NAME}
NOADV_MODEL_NAME=combined_koma_lucy_mild_noadv_20260801
NOADV_OUT=results/${NOADV_MODEL_NAME}

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

CKPT=checkpoints/${TAG}
OUT=results/${TAG}

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] combined_koma direct single-stage unet 100-epoch start"
echo "tag=$TAG epochs=$EPOCHS train_list=$TRAIN_LIST line_dir=$TRAIN_LINE_DIR"

echo "=== train direct unet (no aux, plain BCE+L1+edge, no GAN), 100 epochs ==="
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
  --save-every 10 \
  --require-cuda

echo "=== infer direct unet (100ep) on eval list ==="
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
  --model lucy_mild_noadv="$NOADV_OUT" \
  --model direct_unet_100ep="$OUT" \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    "$BASELINE_MODEL_NAME" \
    "$NOADV_MODEL_NAME" \
    "$TAG" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

# CNN+GAN-family work runs on cleanup-refiner, but Monday's cron job
# (scripts/train_controlnet.py) only exists on diffusion-controlnet --
# switch back automatically since the user will be asleep when this
# finishes (~8h run).
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
  "Lineart combined_koma direct single-stage unet 100-epoch run complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma direct single-stage unet 100-epoch run complete"
