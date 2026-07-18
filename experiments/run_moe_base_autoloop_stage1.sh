#!/bin/bash
# Wait for the clean unique baseline, then prepare review artifacts and handoff.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
WAIT_UNIT=${WAIT_UNIT:-lineart-shape1-base-clean-unique-bce.service}
EXPERIMENT=shape1_base_clean_unique_bce
TRAIN_LIST=dataset/pairs_480/valid_train_base_clean_unique.txt
EVAL_LIST=dataset/pairs_480/eval_fixed_clean_lineart004.txt
CKPT=checkpoints/$EXPERIMENT
RESULTS=results/$EXPERIMENT
LOG=logs/train_shape1_base_clean_unique_bce.service.log
STAGE_LOG=logs/moe_base_autoloop_stage1.log
METRICS=results/fixed_output_metrics_base_clean_unique_lineart004.csv
COMBINED_METRICS=results/fixed_output_metrics_base_clean_lineart004_compare.csv
AUDIT_SUMMARY=results/pair_dataset_integrity_summary_base_clean_unique.csv
AUDIT_FINDINGS=results/pair_dataset_integrity_findings_base_clean_unique.csv
GLOBAL_AUDIT_SUMMARY=results/pair_dataset_integrity_summary_autoloop_global.csv
GLOBAL_AUDIT_FINDINGS=results/pair_dataset_integrity_findings_autoloop_global.csv
MONTAGE=results/compare_shape1_base_clean_unique_lineart004.png
MULTI_MONTAGE=results/compare_clean_baselines_lineart004.png
HANDOFF=doc/autoloop_handoff_shape1_base_clean_unique_bce.md
DONE_MARKER=logs/moe_base_autoloop_stage1.done
NOTIFY_TITLE="Lineart MoE autoloop stage1 complete"
NOTIFY_BODY="Review $HANDOFF and $MULTI_MONTAGE"

mkdir -p logs results doc
exec > >(tee -a "$STAGE_LOG") 2>&1

echo "[$(date --iso-8601=seconds)] autoloop stage1 start"

if systemctl --user is-active --quiet "$WAIT_UNIT"; then
  echo "waiting for $WAIT_UNIT to finish"
  while systemctl --user is-active --quiet "$WAIT_UNIT"; do
    tail -5 "$LOG" || true
    sleep 300
  done
  echo "$WAIT_UNIT is no longer active"
else
  echo "$WAIT_UNIT is not active; continuing with available artifacts"
fi

echo "checking checkpoint and primary outputs"
test -f "$CKPT/best.pth"

if [[ ! -f "$METRICS" || ! -f "$MONTAGE" ]]; then
  echo "primary evaluation artifacts missing; running evaluation tail"
  mkdir -p "$RESULTS"
  while read -r SAMPLE; do
    [[ -z "$SAMPLE" ]] && continue
    NAME="${SAMPLE%.jpg}"
    "$PY" scripts/inference.py \
      --checkpoint "$CKPT/best.pth" \
      --input "dataset/pairs_480/test/rough/${NAME}.jpg" \
      --output "$RESULTS/${NAME}_out.png" \
      --autocontrast
  done < "$EVAL_LIST"

  "$PY" tools/evaluation/evaluate_fixed_outputs.py \
    --models "$EXPERIMENT" \
    --sample-list "$EVAL_LIST" \
    --split test \
    --output-csv "$METRICS"

  "$PY" tools/compare/make_model_eval_compare.py \
    --sample-list "$EVAL_LIST" \
    --model-dir "$RESULTS" \
    --model-label "$EXPERIMENT" \
    --split test \
    --output "$MONTAGE"
fi

echo "running focused integrity audit"
"$PY" tools/evaluation/audit_pair_dataset_integrity.py \
  --train-lists "$TRAIN_LIST" \
  --eval-lists dataset/pairs_480/valid_test.txt "$EVAL_LIST" \
  --output-summary "$AUDIT_SUMMARY" \
  --output-findings "$AUDIT_FINDINGS"

echo "running global list audit snapshot"
"$PY" tools/evaluation/audit_pair_dataset_integrity.py \
  --line-dir-override kurips960=dataset/pairs_480/train/line_kurip_vlm_accept_refined_s960_clean_t192_cc8 \
  --output-summary "$GLOBAL_AUDIT_SUMMARY" \
  --output-findings "$GLOBAL_AUDIT_FINDINGS"

echo "building comparison montage"
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --model shape1_leaky=results/shape1_leaky_lineart004 \
  --model clean_split_bce=results/shape1_clean_split_bce_lineart004 \
  --model "$EXPERIMENT=$RESULTS" \
  --split test \
  --output "$MULTI_MONTAGE"

echo "building combined metrics where outputs are available"
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models shape1_leaky_lineart004 shape1_clean_split_bce_lineart004 "$EXPERIMENT" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$COMBINED_METRICS" || true

echo "writing handoff"
"$PY" tools/evaluation/write_experiment_handoff.py \
  --experiment "$EXPERIMENT" \
  --train-list "$TRAIN_LIST" \
  --checkpoint-dir "$CKPT" \
  --log "$LOG" \
  --metrics-csv "$COMBINED_METRICS" \
  --audit-summary "$AUDIT_SUMMARY" \
  --audit-findings "$AUDIT_FINDINGS" \
  --montage "$MULTI_MONTAGE" \
  --output "$HANDOFF"

echo "[$(date --iso-8601=seconds)] autoloop stage1 complete"
echo "handoff: $HANDOFF"
{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "handoff=$HANDOFF"
  echo "montage=$MULTI_MONTAGE"
  echo "metrics=$COMBINED_METRICS"
} > "$DONE_MARKER"

experiments/send_autoloop_notification.sh "$NOTIFY_TITLE" "$NOTIFY_BODY" || true
echo "done marker: $DONE_MARKER"
