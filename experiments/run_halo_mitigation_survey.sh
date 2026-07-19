#!/bin/bash
# Compare halo mitigation strategies around the msgan+FM cleanup family.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-halo_mitigation_e2}
EPOCHS=${EPOCHS:-2}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
RAW_TRAIN_AUX=${RAW_TRAIN_AUX:-results/line_refiner_e2_atari_train}
RAW_EVAL_AUX=${RAW_EVAL_AUX:-results/line_refiner_e2_atari_eval}
MSGAN_CKPT=${MSGAN_CKPT:-checkpoints/line_refiner_msgan_e2_cleanup_msgan_fm/best.pth}
LOG=logs/${TAG}.log
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png
DONE=logs/${TAG}.done

SRC_TRAIN_AUX=results/${TAG}_source_msgan_train
SRC_EVAL_AUX=results/${TAG}_source_msgan_eval
DOG_TRAIN_AUX=results/${TAG}_dog_train
DOG_EVAL_AUX=results/${TAG}_dog_eval
CUTOFF_TRAIN_AUX=results/${TAG}_cutoff_train
CUTOFF_EVAL_AUX=results/${TAG}_cutoff_eval

mkdir -p logs results
: > "$LOG"

materialize_source_msgan() {
  echo "=== materialize source_msgan aux ===" | tee -a "$LOG"
  "$PY" scripts/materialize_i2i_aux.py \
    --checkpoint "$MSGAN_CKPT" \
    --file-list "$TRAIN_LIST" \
    --rough-dir dataset/pairs_480/train/rough \
    --aux-dir "$RAW_TRAIN_AUX" \
    --output-dir "$SRC_TRAIN_AUX" \
    --autocontrast 2>&1 | tee -a "$LOG"
  "$PY" scripts/materialize_i2i_aux.py \
    --checkpoint "$MSGAN_CKPT" \
    --file-list "$EVAL_LIST" \
    --rough-dir dataset/pairs_480/test/rough \
    --aux-dir "$RAW_EVAL_AUX" \
    --output-dir "$SRC_EVAL_AUX" \
    --autocontrast 2>&1 | tee -a "$LOG"
}

materialize_postprocess_aux() {
  echo "=== materialize postprocess aux ===" | tee -a "$LOG"
  "$PY" scripts/preprocess_atari_aux.py \
    --input-dir "$RAW_TRAIN_AUX" \
    --output-dir "$DOG_TRAIN_AUX" \
    --mode dog 2>&1 | tee -a "$LOG"
  "$PY" scripts/preprocess_atari_aux.py \
    --input-dir "$RAW_EVAL_AUX" \
    --output-dir "$DOG_EVAL_AUX" \
    --mode dog 2>&1 | tee -a "$LOG"
  "$PY" scripts/preprocess_atari_aux.py \
    --input-dir "$RAW_TRAIN_AUX" \
    --output-dir "$CUTOFF_TRAIN_AUX" \
    --mode cutoff 2>&1 | tee -a "$LOG"
  "$PY" scripts/preprocess_atari_aux.py \
    --input-dir "$RAW_EVAL_AUX" \
    --output-dir "$CUTOFF_EVAL_AUX" \
    --mode cutoff 2>&1 | tee -a "$LOG"
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

materialize_source_msgan
materialize_postprocess_aux

# 1. source halo mitigation: use msgan-cleaned atari as the next aux.
train_candidate source_msgan_aux cleanup "$SRC_TRAIN_AUX" "$SRC_EVAL_AUX" \
  --gan --multiscale-gan \
  --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
  --bce-weight 0.75 --l1-weight 0.03 \
  --shape-weight 0.08 --ink-weight 0.14 \
  --binary-weight 0.10 --structure-weight 0.04 \
  --adv-weight 0.03 --feature-match-weight 0.08

# 2. postprocess atari: pass a DoG halo-suppressed hint instead of raw atari.
train_candidate dog_aux_msgan cleanup "$DOG_TRAIN_AUX" "$DOG_EVAL_AUX" \
  --gan --multiscale-gan \
  --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
  --bce-weight 0.75 --l1-weight 0.03 \
  --shape-weight 0.08 --ink-weight 0.14 \
  --binary-weight 0.10 --structure-weight 0.04 \
  --adv-weight 0.03 --feature-match-weight 0.08

# 3. application order: use atari as context/mask, not as direct residual base.
train_candidate mask_order_msgan maskcleanup "$RAW_TRAIN_AUX" "$RAW_EVAL_AUX" \
  --gan --multiscale-gan \
  --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
  --bce-weight 0.75 --l1-weight 0.03 \
  --shape-weight 0.08 --ink-weight 0.14 \
  --binary-weight 0.10 --structure-weight 0.04 \
  --adv-weight 0.03 --feature-match-weight 0.08

"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --model milddup800=results/shape1_clean_split_bce_milddup800_ft10_lr1e5 \
  --model "raw_atari=$RAW_EVAL_AUX" \
  --model cleanup_msgan_fm=results/line_refiner_msgan_e2_cleanup_msgan_fm \
  --model source_msgan_aux=results/${TAG}_source_msgan_aux \
  --model dog_aux_msgan=results/${TAG}_dog_aux_msgan \
  --model mask_order_msgan=results/${TAG}_mask_order_msgan \
  --split test \
  --output "$MONTAGE" 2>&1 | tee -a "$LOG"

"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    shape1_clean_split_bce_milddup800_ft10_lr1e5 \
    line_refiner_msgan_e2_cleanup_msgan_fm \
    ${TAG}_source_msgan_aux \
    ${TAG}_dog_aux_msgan \
    ${TAG}_mask_order_msgan \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS" 2>&1 | tee -a "$LOG"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "raw_train_aux=$RAW_TRAIN_AUX"
  echo "raw_eval_aux=$RAW_EVAL_AUX"
  echo "msgan_checkpoint=$MSGAN_CKPT"
  echo "source_train_aux=$SRC_TRAIN_AUX"
  echo "source_eval_aux=$SRC_EVAL_AUX"
  echo "dog_train_aux=$DOG_TRAIN_AUX"
  echo "dog_eval_aux=$DOG_EVAL_AUX"
  echo "cutoff_train_aux=$CUTOFF_TRAIN_AUX"
  echo "cutoff_eval_aux=$CUTOFF_EVAL_AUX"
  echo "metrics=$METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart halo mitigation survey complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
