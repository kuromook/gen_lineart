#!/bin/bash
# Train a small model-family survey and build one shared clean-lineart004 montage.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-model_survey_e2}
EPOCHS=${EPOCHS:-2}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
BASE_CKPT=${BASE_CKPT:-checkpoints/shape1_clean_split_bce/best.pth}
LOG=logs/${TAG}.log
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png
DONE=logs/${TAG}.done

mkdir -p logs results
: > "$LOG"

run_candidate() {
  local label=$1
  shift
  local ckpt=checkpoints/${TAG}_${label}
  local out=results/${TAG}_${label}
  mkdir -p "$out"
  echo "=== train ${label} ===" | tee -a "$LOG"
  "$PY" scripts/train_i2i_survey.py \
    --checkpoint-dir "$ckpt" \
    --file-list "$TRAIN_LIST" \
    --epochs "$EPOCHS" \
    --workers 0 \
    "$@" 2>&1 | tee -a "$LOG"

  echo "=== infer ${label} ===" | tee -a "$LOG"
  while read -r sample; do
    [[ -z "$sample" ]] && continue
    name="${sample%.jpg}"
    "$PY" scripts/inference_i2i.py \
      --checkpoint "$ckpt/best.pth" \
      --input "dataset/pairs_480/test/rough/${name}.jpg" \
      --output "$out/${name}_out.png" \
      --autocontrast 2>&1 | tee -a "$LOG"
  done < "$EVAL_LIST"
}

run_candidate unet_skip50 --model unet_skip50 --resume-generator "$BASE_CKPT" --strict-resume --lr 1e-5
run_candidate unet_gan --model unet --gan --resume-generator "$BASE_CKPT" --strict-resume --lr 1e-5 --lr-d 2e-5
run_candidate resnet --model resnet --lr 2e-4 --pos-weight 3.0
run_candidate resnet_gan --model resnet --gan --lr 2e-4 --lr-d 2e-5 --pos-weight 3.0

"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --model shape1_clean_split_bce=results/shape1_clean_split_bce_lineart004 \
  --model milddup800=results/shape1_clean_split_bce_milddup800_ft10_lr1e5 \
  --model unet_skip50=results/${TAG}_unet_skip50 \
  --model unet_gan=results/${TAG}_unet_gan \
  --model resnet=results/${TAG}_resnet \
  --model resnet_gan=results/${TAG}_resnet_gan \
  --split test \
  --output "$MONTAGE" 2>&1 | tee -a "$LOG"

"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    shape1_clean_split_bce_lineart004 \
    shape1_clean_split_bce_milddup800_ft10_lr1e5 \
    ${TAG}_unet_skip50 \
    ${TAG}_unet_gan \
    ${TAG}_resnet \
    ${TAG}_resnet_gan \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS" 2>&1 | tee -a "$LOG"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "train_list=$TRAIN_LIST"
  echo "eval_list=$EVAL_LIST"
  echo "metrics=$METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

echo "done marker: $DONE"
