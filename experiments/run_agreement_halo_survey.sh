#!/bin/bash
# Test whether high rough-line agreement training reduces halo.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-agreement_halo_e2}
EPOCHS=${EPOCHS:-2}
SPLIT_COUNT=${SPLIT_COUNT:-240}
BASE_LIST=${BASE_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
TRAIN_AUX=${TRAIN_AUX:-results/lucy_mask_deep_e2_lucy_mild_train}
EVAL_AUX=${EVAL_AUX:-results/lucy_mask_deep_e2_lucy_mild_eval}
LOG=logs/${TAG}.log
DONE=logs/${TAG}.done

AGREE_CSV=results/${TAG}_agreement_scores.csv
HIGH_LIST=dataset/pairs_480/${TAG}_high${SPLIT_COUNT}.txt
LOW_LIST=dataset/pairs_480/${TAG}_low${SPLIT_COUNT}.txt
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
HALO_METRICS=results/halo_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

mkdir -p logs results
: > "$LOG"

score_splits() {
  echo "=== score rough-line agreement ===" | tee -a "$LOG"
  "$PY" tools/evaluation/score_pair_agreement.py \
    --file-list "$BASE_LIST" \
    --rough-dir dataset/pairs_480/train/rough \
    --line-dir dataset/pairs_480/train/line \
    --output-csv "$AGREE_CSV" \
    --high-list "$HIGH_LIST" \
    --low-list "$LOW_LIST" \
    --count "$SPLIT_COUNT" 2>&1 | tee -a "$LOG"
}

train_candidate() {
  local label=$1
  local file_list=$2
  local ckpt=checkpoints/${TAG}_${label}
  local out=results/${TAG}_${label}
  mkdir -p "$out"
  echo "=== train ${label} ===" | tee -a "$LOG"
  "$PY" scripts/train_i2i_survey.py \
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
    --adv-weight 0.03 --feature-match-weight 0.08 2>&1 | tee -a "$LOG"

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

score_splits
train_candidate high_agreement "$HIGH_LIST"
train_candidate low_agreement "$LOW_LIST"

"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --model milddup800=results/shape1_clean_split_bce_milddup800_ft10_lr1e5 \
  --model dog=results/halo_mitigation_e2_dog_aux_msgan \
  --model lucy_mild=results/lucy_mask_deep_e2_lucy_mild_aux_msgan \
  --model lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan \
  --model high_agree=results/${TAG}_high_agreement \
  --model low_agree=results/${TAG}_low_agreement \
  --split test \
  --output "$MONTAGE" 2>&1 | tee -a "$LOG"

"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    shape1_clean_split_bce_milddup800_ft10_lr1e5 \
    halo_mitigation_e2_dog_aux_msgan \
    lucy_mask_deep_e2_lucy_mild_aux_msgan \
    lucy_mask_deep_e2_lucy_thin_aux_msgan \
    ${TAG}_high_agreement \
    ${TAG}_low_agreement \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS" 2>&1 | tee -a "$LOG"

"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --models \
    shape1_clean_split_bce_milddup800_ft10_lr1e5 \
    halo_mitigation_e2_dog_aux_msgan \
    lucy_mask_deep_e2_lucy_mild_aux_msgan \
    lucy_mask_deep_e2_lucy_thin_aux_msgan \
    ${TAG}_high_agreement \
    ${TAG}_low_agreement \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$HALO_METRICS" 2>&1 | tee -a "$LOG"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "split_count=$SPLIT_COUNT"
  echo "base_list=$BASE_LIST"
  echo "high_list=$HIGH_LIST"
  echo "low_list=$LOW_LIST"
  echo "train_aux=$TRAIN_AUX"
  echo "eval_aux=$EVAL_AUX"
  echo "agreement_scores=$AGREE_CSV"
  echo "metrics=$METRICS"
  echo "halo_metrics=$HALO_METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart agreement halo survey complete" \
  "Review $MONTAGE, $METRICS, and $HALO_METRICS" || true

echo "done marker: $DONE"
