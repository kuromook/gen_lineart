#!/bin/bash
# Two-stage 2ch line refiner: structure warmup, then binary/ink tightening.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-line_refiner_staged_e2e2}
STAGE1_EPOCHS=${STAGE1_EPOCHS:-2}
STAGE2_EPOCHS=${STAGE2_EPOCHS:-2}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
TRAIN_AUX=${TRAIN_AUX:-results/line_refiner_e2_atari_train}
EVAL_AUX=${EVAL_AUX:-results/line_refiner_e2_atari_eval}
AUX_DROPOUT=${AUX_DROPOUT:-0.25}
AUX_SCALE_MIN=${AUX_SCALE_MIN:-0.0}
LOG=logs/${TAG}.log
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png
DONE=logs/${TAG}.done

STAGE1_CKPT=checkpoints/${TAG}_stage1_structure
STAGE2_CKPT=checkpoints/${TAG}_stage2_tight
STAGE1_OUT=results/${TAG}_stage1_structure
STAGE2_OUT=results/${TAG}_stage2_tight

mkdir -p logs results "$STAGE1_OUT" "$STAGE2_OUT"
: > "$LOG"

run_infer() {
  local ckpt=$1
  local out=$2
  local label=$3
  echo "=== infer ${label} ===" | tee -a "$LOG"
  while read -r sample; do
    [[ -z "$sample" ]] && continue
    name="${sample%.jpg}"
    "$PY" scripts/inference_i2i.py \
      --checkpoint "$ckpt/best.pth" \
      --input "dataset/pairs_480/test/rough/${name}.jpg" \
      --aux-input "$EVAL_AUX/${name}_out.png" \
      --output "$out/${name}_out.png" \
      --autocontrast 2>&1 | tee -a "$LOG"
  done < "$EVAL_LIST"
}

echo "=== train stage1_structure ===" | tee -a "$LOG"
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "$STAGE1_CKPT" \
  --file-list "$TRAIN_LIST" \
  --aux-dir "$TRAIN_AUX" \
  --aux-dropout "$AUX_DROPOUT" \
  --aux-scale-min "$AUX_SCALE_MIN" \
  --epochs "$STAGE1_EPOCHS" \
  --workers 0 \
  --model unet \
  --lr 1e-4 --pos-weight 5.0 \
  --bce-weight 0.85 --l1-weight 0.05 \
  --shape-weight 0.10 --ink-weight 0.08 \
  --binary-weight 0.02 --adv-weight 0.0 2>&1 | tee -a "$LOG"

run_infer "$STAGE1_CKPT" "$STAGE1_OUT" stage1_structure

echo "=== train stage2_tight ===" | tee -a "$LOG"
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "$STAGE2_CKPT" \
  --file-list "$TRAIN_LIST" \
  --aux-dir "$TRAIN_AUX" \
  --aux-dropout "$AUX_DROPOUT" \
  --aux-scale-min "$AUX_SCALE_MIN" \
  --resume-generator "$STAGE1_CKPT/best.pth" \
  --strict-resume \
  --epochs "$STAGE2_EPOCHS" \
  --workers 0 \
  --model unet \
  --lr 5e-5 --pos-weight 5.0 \
  --bce-weight 0.85 --l1-weight 0.03 \
  --shape-weight 0.08 --ink-weight 0.14 \
  --binary-weight 0.12 --adv-weight 0.0 2>&1 | tee -a "$LOG"

run_infer "$STAGE2_CKPT" "$STAGE2_OUT" stage2_tight

"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --model milddup800=results/shape1_clean_split_bce_milddup800_ft10_lr1e5 \
  --model resnet_atari="$EVAL_AUX" \
  --model refiner_e2=results/line_refiner_e2_refiner_unet \
  --model tight_bin12=results/line_refiner_tight_e2_refiner_unet_bin12_ink14 \
  --model stage1="$STAGE1_OUT" \
  --model stage2="$STAGE2_OUT" \
  --split test \
  --output "$MONTAGE" 2>&1 | tee -a "$LOG"

"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    shape1_clean_split_bce_milddup800_ft10_lr1e5 \
    line_refiner_e2_refiner_unet \
    line_refiner_tight_e2_refiner_unet_bin12_ink14 \
    ${TAG}_stage1_structure \
    ${TAG}_stage2_tight \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS" 2>&1 | tee -a "$LOG"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "stage1_epochs=$STAGE1_EPOCHS"
  echo "stage2_epochs=$STAGE2_EPOCHS"
  echo "aux_dropout=$AUX_DROPOUT"
  echo "aux_scale_min=$AUX_SCALE_MIN"
  echo "train_aux=$TRAIN_AUX"
  echo "eval_aux=$EVAL_AUX"
  echo "stage1_checkpoint=$STAGE1_CKPT/best.pth"
  echo "stage2_checkpoint=$STAGE2_CKPT/best.pth"
  echo "metrics=$METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart staged 2ch refiner complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
