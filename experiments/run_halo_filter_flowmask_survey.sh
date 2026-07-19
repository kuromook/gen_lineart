#!/bin/bash
# Deepen halo handling along two axes:
# 1. filter/deblur variants for atari aux
# 2. flow-aware mask gating instead of plain atari context
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-halo_filter_flowmask_e2}
EPOCHS=${EPOCHS:-2}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
RAW_TRAIN_AUX=${RAW_TRAIN_AUX:-results/line_refiner_e2_atari_train}
RAW_EVAL_AUX=${RAW_EVAL_AUX:-results/line_refiner_e2_atari_eval}
LOG=logs/${TAG}.log
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png
DONE=logs/${TAG}.done

mkdir -p logs results
: > "$LOG"

preprocess_pair() {
  local mode=$1
  local train_aux=results/${TAG}_${mode}_train
  local eval_aux=results/${TAG}_${mode}_eval
  echo "=== preprocess ${mode} aux ===" | tee -a "$LOG"
  "$PY" scripts/preprocess_atari_aux.py \
    --input-dir "$RAW_TRAIN_AUX" \
    --output-dir "$train_aux" \
    --mode "$mode" 2>&1 | tee -a "$LOG"
  "$PY" scripts/preprocess_atari_aux.py \
    --input-dir "$RAW_EVAL_AUX" \
    --output-dir "$eval_aux" \
    --mode "$mode" 2>&1 | tee -a "$LOG"
}

train_candidate() {
  local label=$1
  local model=$2
  local train_aux=$3
  local eval_aux=$4
  shift 4
  local ckpt=checkpoints/${TAG}_${label}
  local out=results/${TAG}_${label}
  mkdir -p "$out"
  echo "=== train ${label} ===" | tee -a "$LOG"
  "$PY" scripts/train_i2i_survey.py \
    --checkpoint-dir "$ckpt" \
    --file-list "$TRAIN_LIST" \
    --aux-dir "$train_aux" \
    --epochs "$EPOCHS" \
    --workers 0 \
    --model "$model" \
    "$@" 2>&1 | tee -a "$LOG"

  echo "=== infer ${label} ===" | tee -a "$LOG"
  while read -r sample; do
    [[ -z "$sample" ]] && continue
    name="${sample%.jpg}"
    "$PY" scripts/inference_i2i.py \
      --checkpoint "$ckpt/best.pth" \
      --input "dataset/pairs_480/test/rough/${name}.jpg" \
      --aux-input "$eval_aux/${name}_out.png" \
      --output "$out/${name}_out.png" \
      --autocontrast 2>&1 | tee -a "$LOG"
  done < "$EVAL_LIST"
}

COMMON_ARGS=(
  --gan --multiscale-gan
  --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0
  --bce-weight 0.75 --l1-weight 0.03
  --shape-weight 0.08 --ink-weight 0.14
  --binary-weight 0.10 --structure-weight 0.04
  --adv-weight 0.03 --feature-match-weight 0.08
)

for mode in wiener lucy bilateral flowdog flowmask; do
  preprocess_pair "$mode"
done

train_candidate wiener_aux_msgan cleanup \
  "results/${TAG}_wiener_train" "results/${TAG}_wiener_eval" \
  "${COMMON_ARGS[@]}"

train_candidate lucy_aux_msgan cleanup \
  "results/${TAG}_lucy_train" "results/${TAG}_lucy_eval" \
  "${COMMON_ARGS[@]}"

train_candidate bilateral_aux_msgan cleanup \
  "results/${TAG}_bilateral_train" "results/${TAG}_bilateral_eval" \
  "${COMMON_ARGS[@]}"

train_candidate flowdog_aux_msgan cleanup \
  "results/${TAG}_flowdog_train" "results/${TAG}_flowdog_eval" \
  "${COMMON_ARGS[@]}"

train_candidate flowmask_cleanup flowmaskcleanup \
  "results/${TAG}_flowmask_train" "results/${TAG}_flowmask_eval" \
  "${COMMON_ARGS[@]}"

train_candidate flowmask_cleanup_dropout flowmaskcleanup \
  "results/${TAG}_flowmask_train" "results/${TAG}_flowmask_eval" \
  "${COMMON_ARGS[@]}" --aux-dropout 0.25 --aux-scale-min 0.35

"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --model milddup800=results/shape1_clean_split_bce_milddup800_ft10_lr1e5 \
  --model "raw_atari=$RAW_EVAL_AUX" \
  --model cleanup_msgan_fm=results/line_refiner_msgan_e2_cleanup_msgan_fm \
  --model dog_aux_msgan=results/halo_mitigation_e2_dog_aux_msgan \
  --model wiener_aux_msgan=results/${TAG}_wiener_aux_msgan \
  --model lucy_aux_msgan=results/${TAG}_lucy_aux_msgan \
  --model bilateral_aux_msgan=results/${TAG}_bilateral_aux_msgan \
  --model flowdog_aux_msgan=results/${TAG}_flowdog_aux_msgan \
  --model flowmask_cleanup=results/${TAG}_flowmask_cleanup \
  --model flowmask_dropout=results/${TAG}_flowmask_cleanup_dropout \
  --split test \
  --output "$MONTAGE" 2>&1 | tee -a "$LOG"

"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    shape1_clean_split_bce_milddup800_ft10_lr1e5 \
    line_refiner_msgan_e2_cleanup_msgan_fm \
    halo_mitigation_e2_dog_aux_msgan \
    ${TAG}_wiener_aux_msgan \
    ${TAG}_lucy_aux_msgan \
    ${TAG}_bilateral_aux_msgan \
    ${TAG}_flowdog_aux_msgan \
    ${TAG}_flowmask_cleanup \
    ${TAG}_flowmask_cleanup_dropout \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS" 2>&1 | tee -a "$LOG"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "raw_train_aux=$RAW_TRAIN_AUX"
  echo "raw_eval_aux=$RAW_EVAL_AUX"
  echo "metrics=$METRICS"
  echo "montage=$MONTAGE"
  echo "filter_modes=wiener,lucy,bilateral,flowdog"
  echo "mask_modes=flowmask_cleanup,flowmask_cleanup_dropout"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart halo filter/flowmask survey complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
