#!/bin/bash
# Fine-tune clean_split_bce on a small mild-duplicate list, then evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
EXPERIMENT=shape1_clean_split_bce_milddup800_ft10_lr1e5
CKPT=checkpoints/$EXPERIMENT
RESULTS=results/$EXPERIMENT
SEED_LIST=dataset/pairs_480/valid_train_base_clean_unique.txt
POOL_LIST=dataset/pairs_480/valid_train_std15_clean_split_moredupes.txt
TRAIN_LIST=dataset/pairs_480/valid_train_milddup800_clean.txt
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
RESUME=checkpoints/shape1_clean_split_bce/best.pth
METRICS=results/fixed_output_metrics_${EXPERIMENT}_clean_lineart004.csv
COMBINED_METRICS=results/fixed_output_metrics_${EXPERIMENT}_compare.csv
AUDIT_SUMMARY=results/pair_dataset_integrity_summary_${EXPERIMENT}.csv
AUDIT_FINDINGS=results/pair_dataset_integrity_findings_${EXPERIMENT}.csv
MONTAGE=results/compare_${EXPERIMENT}_clean_lineart004.png
MULTI_MONTAGE=results/compare_${EXPERIMENT}_vs_clean_split_bce.png
DONE_MARKER=logs/${EXPERIMENT}.done

mkdir -p logs results "$RESULTS"

"$PY" tools/pair_extraction/build_mild_duplicate_list.py \
  --seed-list "$SEED_LIST" \
  --pool-list "$POOL_LIST" \
  --out "$TRAIN_LIST" \
  --target-rows 800 \
  --max-per-canonical 2 \
  --allow-exact-duplicates

"$PY" tools/evaluation/audit_pair_dataset_integrity.py \
  --train-lists "$TRAIN_LIST" \
  --eval-lists dataset/pairs_480/valid_test.txt "$EVAL_LIST" \
  --output-summary "$AUDIT_SUMMARY" \
  --output-findings "$AUDIT_FINDINGS"

"$PY" scripts/train.py \
  --file-list "$TRAIN_LIST" \
  --checkpoint-dir "$CKPT" \
  --resume "$RESUME" \
  --epochs 10 \
  --lr 1e-5 \
  --pos-weight 5.0 \
  --edge-weight 0.0 \
  --shape-weight 0.0 \
  --ink-weight 0.0 \
  --autocontrast

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
  --output-csv "$COMBINED_METRICS"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "experiment=$EXPERIMENT"
  echo "train_list=$TRAIN_LIST"
  echo "train_rows=$(wc -l < "$TRAIN_LIST")"
  echo "resume=$RESUME"
  echo "checkpoint=$CKPT/best.pth"
  echo "montage=$MULTI_MONTAGE"
  echo "metrics=$COMBINED_METRICS"
  echo "audit_summary=$AUDIT_SUMMARY"
  echo "audit_findings=$AUDIT_FINDINGS"
} > "$DONE_MARKER"

experiments/send_autoloop_notification.sh \
  "Lineart milddup800 fine-tune complete" \
  "Review $MULTI_MONTAGE and $COMBINED_METRICS" || true
echo "done marker: $DONE_MARKER"
