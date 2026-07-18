#!/bin/bash
# Experiment 1: keep eval/test clean, but allow train-internal similar tiles.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
EXPERIMENT=shape1_std15_clean_split_moredupes_bce
CKPT=checkpoints/$EXPERIMENT
RESULTS=results/$EXPERIMENT
TRAIN_LIST=dataset/pairs_480/valid_train_std15_clean_split_moredupes.txt
EVAL_LIST=dataset/pairs_480/eval_fixed_clean_lineart004.txt
LOG=logs/train_${EXPERIMENT}.service.log
METRICS=results/fixed_output_metrics_${EXPERIMENT}_lineart004.csv
COMBINED_METRICS=results/fixed_output_metrics_exp1_moredupes_lineart004_compare.csv
AUDIT_SUMMARY=results/pair_dataset_integrity_summary_${EXPERIMENT}.csv
AUDIT_FINDINGS=results/pair_dataset_integrity_findings_${EXPERIMENT}.csv
MONTAGE=results/compare_${EXPERIMENT}_lineart004.png
MULTI_MONTAGE=results/compare_exp1_moredupes_lineart004.png
DONE_MARKER=logs/${EXPERIMENT}.done
NOTIFY_TITLE="Lineart experiment 1 complete"
NOTIFY_BODY="Review $MULTI_MONTAGE and $COMBINED_METRICS"

mkdir -p logs results

"$PY" tools/pair_extraction/build_clean_split_lists.py \
  --train-in dataset/pairs_480/valid_train_std15.txt \
  --test-in dataset/pairs_480/valid_test.txt \
  --train-out "$TRAIN_LIST" \
  --eval-out "$EVAL_LIST"

echo "experiment: $EXPERIMENT"
echo "train_list: $TRAIN_LIST ($(wc -l < "$TRAIN_LIST"))"
echo "eval_list:  $EVAL_LIST ($(wc -l < "$EVAL_LIST"))"

"$PY" tools/evaluation/audit_pair_dataset_integrity.py \
  --train-lists "$TRAIN_LIST" \
  --eval-lists dataset/pairs_480/valid_test.txt "$EVAL_LIST" \
  --output-summary "$AUDIT_SUMMARY" \
  --output-findings "$AUDIT_FINDINGS"

"$PY" scripts/train.py \
  --file-list "$TRAIN_LIST" \
  --checkpoint-dir "$CKPT" \
  --epochs 200 \
  --lr 1e-4 \
  --pos-weight 5.0 \
  --edge-weight 0.0 \
  --shape-weight 0.0 \
  --ink-weight 0.0 \
  --autocontrast

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

"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --model shape1_clean_split_bce=results/shape1_clean_split_bce_lineart004 \
  --model "$EXPERIMENT=$RESULTS" \
  --split test \
  --output "$MULTI_MONTAGE"

"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models shape1_clean_split_bce_lineart004 "$EXPERIMENT" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$COMBINED_METRICS" || true

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "experiment=$EXPERIMENT"
  echo "train_list=$TRAIN_LIST"
  echo "train_rows=$(wc -l < "$TRAIN_LIST")"
  echo "checkpoint=$CKPT/best.pth"
  echo "montage=$MULTI_MONTAGE"
  echo "metrics=$COMBINED_METRICS"
  echo "audit_summary=$AUDIT_SUMMARY"
  echo "audit_findings=$AUDIT_FINDINGS"
} > "$DONE_MARKER"

experiments/send_autoloop_notification.sh "$NOTIFY_TITLE" "$NOTIFY_BODY" || true
echo "done marker: $DONE_MARKER"
