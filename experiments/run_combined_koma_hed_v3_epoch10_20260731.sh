#!/bin/bash
# Third attempt at Direction 8 (HedUNetGenerator, model=hed): same
# residual-anchored architecture as combined_koma_hed_v2_20260731, but with
# EPOCHS=10 instead of 3.
#
# v1 (no residual anchor, 3 epochs) was chronically under-inked (F1@2px
# 0.190). v2 (residual anchor added, 3 epochs) improved but still lagged
# (F1@2px 0.235) while the same fix let dualhead (a shallow trunk) reach
# parity with the adopted cleanup-family models. Unlike dualhead's shallow
# conv trunk, hed's full U-Net encoder/decoder is trained entirely from
# random initialization (only the final out_conv is anchored to the aux
# input) -- 10 epochs is this project's standard budget for from-scratch
# unet-family training (e.g. *_warm_clean_bce_e10 runs), so this isolates
# whether v2's shortfall was simply an undertrained trunk rather than an
# architecture problem.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_hed_v3_20260731
PREV_TAG=combined_koma_lucy_mild_msgan_20260729
DARK_TAG=combined_koma_cleanupdark_20260730
DUALHEAD_V2_TAG=combined_koma_dualhead_v2_20260731
HED_V1_TAG=combined_koma_hed_20260731
HED_V2_TAG=combined_koma_hed_v2_20260731
EPOCHS=${EPOCHS:-10}
TRAIN_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
TRAIN_RAW_ROUGH=dataset/pairs_480/train/rough
TRAIN_LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

LUCY_MILD_TRAIN_AUX=results/${PREV_TAG}_lucy_mild_train
LUCY_MILD_EVAL_AUX=results/${PREV_TAG}_lucy_mild_eval
BASE_MODEL_NAME=${PREV_TAG}_baseline_bce_eval8
BASE_OUT=results/${BASE_MODEL_NAME}
PREV_MODEL_NAME=${PREV_TAG}
PREV_OUT=results/${PREV_TAG}
DARK_MODEL_NAME=${DARK_TAG}
DARK_OUT=results/${DARK_TAG}
DUALHEAD_V2_OUT=results/${DUALHEAD_V2_TAG}
HED_V1_OUT=results/${HED_V1_TAG}
HED_V2_OUT=results/${HED_V2_TAG}

CKPT=checkpoints/${TAG}
OUT=results/${TAG}

for d in "$LUCY_MILD_TRAIN_AUX" "$LUCY_MILD_EVAL_AUX" "$BASE_OUT" "$PREV_OUT" "$DARK_OUT" "$DUALHEAD_V2_OUT" "$HED_V1_OUT" "$HED_V2_OUT"; do
  if [ ! -d "$d" ]; then
    echo "missing expected reused artifact dir: $d" >&2
    exit 1
  fi
done

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] combined_koma hed v3 (10 epochs) start"
echo "tag=$TAG epochs=$EPOCHS train_list=$TRAIN_LIST line_dir=$TRAIN_LINE_DIR"

echo "=== train hed v3 (residual-anchored, side_weight=0.06, epochs=$EPOCHS) on koma data ==="
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "$CKPT" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --line-dir "$TRAIN_LINE_DIR" \
  --aux-dir "$LUCY_MILD_TRAIN_AUX" \
  --epochs "$EPOCHS" \
  --workers 0 \
  --model hed \
  --gan --multiscale-gan \
  --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
  --bce-weight 0.75 --l1-weight 0.03 \
  --shape-weight 0.08 --ink-weight 0.14 \
  --binary-weight 0.10 --structure-weight 0.04 \
  --side-weight 0.06 \
  --adv-weight 0.03 --feature-match-weight 0.08 \
  --require-cuda

echo "=== infer hed v3 (koma) on eval list ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$CKPT/best.pth" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --aux-dir "$LUCY_MILD_EVAL_AUX" \
  --output-dir "$OUT" \
  --autocontrast

echo "=== montage (baseline / lucy_mild / cleanupdark / dualhead_v2 / hed_v1 / hed_v2 / hed_v3 / GT) ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model plain_bce_koma="$BASE_OUT" \
  --model lucy_mild_koma="$PREV_OUT" \
  --model cleanupdark_koma="$DARK_OUT" \
  --model dualhead_v2="$DUALHEAD_V2_OUT" \
  --model hed_v1="$HED_V1_OUT" \
  --model hed_v2="$HED_V2_OUT" \
  --model hed_v3_e10="$OUT" \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    "$BASE_MODEL_NAME" \
    "$PREV_MODEL_NAME" \
    "$DARK_MODEL_NAME" \
    "$DUALHEAD_V2_TAG" \
    "$HED_V1_TAG" \
    "$HED_V2_TAG" \
    "$TAG" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "train_list=$TRAIN_LIST"
  echo "train_line_dir=$TRAIN_LINE_DIR"
  echo "model=hed side_weight=0.06 residual_anchor=true"
  echo "montage=$MONTAGE"
  echo "metrics=$METRICS"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart combined_koma hed v3 (10 epochs) retrain complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma hed v3 (10 epochs) retrain complete"
