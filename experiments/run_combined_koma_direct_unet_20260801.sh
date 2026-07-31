#!/bin/bash
# Replicate the pre-leak-fix-era notebook (notebooks/gen_lineart.ipynb)
# architecture directly on the current clean combined_koma_20260729 data:
# single-stage direct regression (rough -> line, --model unet, no atari/aux
# hint, no residual-anchor bounded correction) with the same plain
# BCE(pos_weight=3)+L1+edge_loss recipe as the 2026-08-01 noadv ablation
# (--model cleanup). That ablation held the atari+cleanup two-stage
# architecture fixed and only removed the adversarial loss -- it got worse,
# not better, and traced the soft/marbled ceiling to the atari generator's
# own soft output propagating through cleanup's tanh-bounded correction
# (max_delta=4.0). This is the test that isolates the other variable:
# architecture (two-stage bounded-correction vs single-stage direct
# regression), holding the loss recipe fixed and identical to that run.
#
# No aux-dir means no atari materialization or aux preprocessing step is
# needed at all -- should be faster than any of tonight's other runs.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_direct_unet_20260801
EPOCHS=${EPOCHS:-3}
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

echo "[$(date --iso-8601=seconds)] combined_koma direct single-stage unet (no atari/aux) start"
echo "tag=$TAG epochs=$EPOCHS train_list=$TRAIN_LIST line_dir=$TRAIN_LINE_DIR"

echo "=== train direct unet (no aux, plain BCE+L1+edge, no GAN) on koma data ==="
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
  --require-cuda

echo "=== infer direct unet on eval list ==="
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
  --model direct_unet="$OUT" \
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
# switch back automatically since the user may be asleep when this finishes.
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
  "Lineart combined_koma direct single-stage unet ablation complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma direct single-stage unet ablation complete"
