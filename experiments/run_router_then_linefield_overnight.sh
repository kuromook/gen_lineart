#!/bin/bash
# Overnight chain:
# 1. Router/MoE oracle over current expert candidates.
# 2. Initial supervised line-field refiner run.
# Only notify after the final line-field stage completes.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-router_linefield_overnight_e2}
LINEFIELD_TAG=${LINEFIELD_TAG:-linefield_initial_e2}
EPOCHS=${EPOCHS:-2}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
LINEFIELD_TRAIN_AUX=${LINEFIELD_TRAIN_AUX:-results/lucy_mask_deep_e2_lucy_mild_train}
LINEFIELD_EVAL_AUX=${LINEFIELD_EVAL_AUX:-results/lucy_mask_deep_e2_lucy_mild_eval}
LOG=logs/${TAG}.log
DONE=logs/${TAG}.done

ROUTER_ORACLE_DIR=results/${TAG}_oracle
ROUTER_ORACLE_CSV=results/${TAG}_oracle_expert_metrics.csv
ROUTER_ORACLE_SUMMARY=results/${TAG}_oracle_summary.json
ROUTER_MONTAGE=results/compare_${TAG}_oracle.png
ROUTER_METRICS=results/fixed_output_metrics_${TAG}_oracle_compare.csv

LINEFIELD_CKPT=checkpoints/${LINEFIELD_TAG}
LINEFIELD_RESULTS=results/${LINEFIELD_TAG}
LINEFIELD_DEBUG=results/${LINEFIELD_TAG}_debug
LINEFIELD_MONTAGE=results/compare_${LINEFIELD_TAG}.png
LINEFIELD_METRICS=results/fixed_output_metrics_${LINEFIELD_TAG}_compare.csv

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] router + line-field overnight start"

echo "=== router/MoE oracle ==="
"$PY" scripts/evaluate_oracle_moe.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --oracle-dir "$ROUTER_ORACLE_DIR" \
  --output-csv "$ROUTER_ORACLE_CSV" \
  --summary-json "$ROUTER_ORACLE_SUMMARY" \
  --score-mode balanced \
  --model bin12=results/line_refiner_tight_e2_refiner_unet_bin12_ink14 \
  --model bin20=results/line_refiner_tight_e2_refiner_unet_bin20_ink16 \
  --model cleanup_msgan=results/line_refiner_msgan_e2_cleanup_msgan_fm \
  --model dog=results/halo_mitigation_e2_dog_aux_msgan \
  --model lucy_mild=results/lucy_mask_deep_e2_lucy_mild_aux_msgan \
  --model lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan \
  --model flowdog=results/halo_filter_flowmask_e2_flowdog_aux_msgan

"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --model milddup800=results/shape1_clean_split_bce_milddup800_ft10_lr1e5 \
  --model bin12=results/line_refiner_tight_e2_refiner_unet_bin12_ink14 \
  --model bin20=results/line_refiner_tight_e2_refiner_unet_bin20_ink16 \
  --model dog=results/halo_mitigation_e2_dog_aux_msgan \
  --model lucy_mild=results/lucy_mask_deep_e2_lucy_mild_aux_msgan \
  --model lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan \
  --model oracle="$ROUTER_ORACLE_DIR" \
  --split test \
  --output "$ROUTER_MONTAGE"

"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    shape1_clean_split_bce_milddup800_ft10_lr1e5 \
    line_refiner_tight_e2_refiner_unet_bin12_ink14 \
    line_refiner_tight_e2_refiner_unet_bin20_ink16 \
    halo_mitigation_e2_dog_aux_msgan \
    lucy_mask_deep_e2_lucy_mild_aux_msgan \
    lucy_mask_deep_e2_lucy_thin_aux_msgan \
    ${TAG}_oracle \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$ROUTER_METRICS"

echo "=== line-field refiner train ==="
"$PY" scripts/train_line_field_refiner.py \
  --checkpoint-dir "$LINEFIELD_CKPT" \
  --file-list "$TRAIN_LIST" \
  --aux-dir "$LINEFIELD_TRAIN_AUX" \
  --epochs "$EPOCHS" \
  --workers 0 \
  --batch-size 2 \
  --lr 6e-5 \
  --pos-weight 5.0 \
  --center-pos-weight 12.0 \
  --bce-weight 0.70 \
  --l1-weight 0.05 \
  --shape-weight 0.08 \
  --ink-weight 0.12 \
  --center-weight 0.25 \
  --offset-weight 0.08

echo "=== line-field refiner inference ==="
mkdir -p "$LINEFIELD_RESULTS" "$LINEFIELD_DEBUG"
while read -r sample; do
  [[ -z "$sample" ]] && continue
  name="${sample%.jpg}"
  "$PY" scripts/inference_line_field.py \
    --checkpoint "$LINEFIELD_CKPT/best.pth" \
    --input "dataset/pairs_480/test/rough/${name}.jpg" \
    --aux-input "$LINEFIELD_EVAL_AUX/${name}_out.png" \
    --output "$LINEFIELD_RESULTS/${name}_out.png" \
    --debug-dir "$LINEFIELD_DEBUG" \
    --autocontrast
done < "$EVAL_LIST"

"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --model milddup800=results/shape1_clean_split_bce_milddup800_ft10_lr1e5 \
  --model dog=results/halo_mitigation_e2_dog_aux_msgan \
  --model lucy_mild=results/lucy_mask_deep_e2_lucy_mild_aux_msgan \
  --model lucy_thin=results/lucy_mask_deep_e2_lucy_thin_aux_msgan \
  --model oracle="$ROUTER_ORACLE_DIR" \
  --model linefield="$LINEFIELD_RESULTS" \
  --split test \
  --output "$LINEFIELD_MONTAGE"

"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    shape1_clean_split_bce_milddup800_ft10_lr1e5 \
    halo_mitigation_e2_dog_aux_msgan \
    lucy_mask_deep_e2_lucy_mild_aux_msgan \
    lucy_mask_deep_e2_lucy_thin_aux_msgan \
    ${TAG}_oracle \
    ${LINEFIELD_TAG} \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$LINEFIELD_METRICS"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "linefield_tag=$LINEFIELD_TAG"
  echo "epochs=$EPOCHS"
  echo "router_summary=$ROUTER_ORACLE_SUMMARY"
  echo "router_montage=$ROUTER_MONTAGE"
  echo "router_metrics=$ROUTER_METRICS"
  echo "linefield_checkpoint=$LINEFIELD_CKPT/best.pth"
  echo "linefield_montage=$LINEFIELD_MONTAGE"
  echo "linefield_metrics=$LINEFIELD_METRICS"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart router + line-field overnight complete" \
  "Review $LINEFIELD_MONTAGE, $LINEFIELD_METRICS, and $ROUTER_ORACLE_SUMMARY" || true

echo "[$(date --iso-8601=seconds)] router + line-field overnight complete"
echo "done marker: $DONE"
