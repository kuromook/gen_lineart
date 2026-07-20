#!/bin/bash
# Halo-loss survey using the halo-loss worktree code and shared main artifacts.
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/home/sh1/deepl/lineart-halo-loss}
RUN_ROOT=${RUN_ROOT:-/home/sh1/deepl/lineart}
cd "$RUN_ROOT"

PY=./venv/bin/python
TAG=${1:-halo_loss_e2}
EPOCHS=${EPOCHS:-2}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
HIGH_LIST=${HIGH_LIST:-dataset/pairs_480/agreement_halo_e2_high240.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
TRAIN_AUX=${TRAIN_AUX:-results/lucy_mask_deep_e2_lucy_mild_train}
EVAL_AUX=${EVAL_AUX:-results/lucy_mask_deep_e2_lucy_mild_eval}
LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
HALO_METRICS=results/halo_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

mkdir -p logs results
: > "$LOG"

train_candidate() {
  local label=$1
  local file_list=$2
  shift 2
  local ckpt=checkpoints/${TAG}_${label}
  local out=results/${TAG}_${label}
  mkdir -p "$out"
  echo "=== train ${label} ===" | tee -a "$LOG"
  "$PY" "$CODE_ROOT/scripts/train_i2i_survey.py" \
    --checkpoint-dir "$ckpt" \
    --file-list "$file_list" \
    --aux-dir "$TRAIN_AUX" \
    --epochs "$EPOCHS" \
    --workers 0 \
    --model cleanup \
    --gan --multiscale-gan \
    --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
    --bce-weight 0.75 --l1-weight 0.03 \
    --shape-weight 0.08 --ink-weight 0.14 \
    --binary-weight 0.10 --structure-weight 0.04 \
    --adv-weight 0.03 --feature-match-weight 0.08 \
    "$@" 2>&1 | tee -a "$LOG"

  echo "=== infer ${label} ===" | tee -a "$LOG"
  while read -r sample; do
    [[ -z "$sample" ]] && continue
    name="${sample%.jpg}"
    "$PY" "$CODE_ROOT/scripts/inference_i2i.py" \
      --checkpoint "$ckpt/best.pth" \
      --input "dataset/pairs_480/test/rough/${name}.jpg" \
      --aux-input "$EVAL_AUX/${name}_out.png" \
      --output "$out/${name}_out.png" \
      --autocontrast 2>&1 | tee -a "$LOG"
  done < "$EVAL_LIST"
}

train_candidate mild_halo06 "$TRAIN_LIST" \
  --halo-weight 0.06

train_candidate mild_halo10_faint04 "$TRAIN_LIST" \
  --halo-weight 0.10 --faint-weight 0.04

train_candidate high_halo10_faint04 "$HIGH_LIST" \
  --halo-weight 0.10 --faint-weight 0.04

"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --model milddup800=results/shape1_clean_split_bce_milddup800_ft10_lr1e5 \
  --model dog=results/halo_mitigation_e2_dog_aux_msgan \
  --model lucy_mild=results/lucy_mask_deep_e2_lucy_mild_aux_msgan \
  --model lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan \
  --model agree_high=results/agreement_halo_e2_high_agreement \
  --model mild_halo06=results/${TAG}_mild_halo06 \
  --model mild_halo10_faint04=results/${TAG}_mild_halo10_faint04 \
  --model high_halo10_faint04=results/${TAG}_high_halo10_faint04 \
  --split test \
  --output "$MONTAGE" 2>&1 | tee -a "$LOG"

"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    shape1_clean_split_bce_milddup800_ft10_lr1e5 \
    halo_mitigation_e2_dog_aux_msgan \
    lucy_mask_deep_e2_lucy_mild_aux_msgan \
    lucy_mask_deep_e2_lucy_thin_aux_msgan \
    agreement_halo_e2_high_agreement \
    ${TAG}_mild_halo06 \
    ${TAG}_mild_halo10_faint04 \
    ${TAG}_high_halo10_faint04 \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS" 2>&1 | tee -a "$LOG"

"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --models \
    shape1_clean_split_bce_milddup800_ft10_lr1e5 \
    halo_mitigation_e2_dog_aux_msgan \
    lucy_mask_deep_e2_lucy_mild_aux_msgan \
    lucy_mask_deep_e2_lucy_thin_aux_msgan \
    agreement_halo_e2_high_agreement \
    ${TAG}_mild_halo06 \
    ${TAG}_mild_halo10_faint04 \
    ${TAG}_high_halo10_faint04 \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$HALO_METRICS" 2>&1 | tee -a "$LOG"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "code_root=$CODE_ROOT"
  echo "run_root=$RUN_ROOT"
  echo "train_list=$TRAIN_LIST"
  echo "high_list=$HIGH_LIST"
  echo "train_aux=$TRAIN_AUX"
  echo "eval_aux=$EVAL_AUX"
  echo "metrics=$METRICS"
  echo "halo_metrics=$HALO_METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart halo loss survey complete" \
  "Review $MONTAGE, $METRICS, and $HALO_METRICS" || true

echo "done marker: $DONE"
