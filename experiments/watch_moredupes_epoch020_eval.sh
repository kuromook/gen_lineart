#!/bin/bash
# Stop the long moredupes run at epoch020 and evaluate that checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
UNIT=${UNIT:-lineart-exp1-moredupes-bce.service}
EXPERIMENT=shape1_std15_clean_split_moredupes_bce
LABEL=${LABEL:-shape1_std15_clean_split_moredupes_bce_epoch020}
CKPT=checkpoints/$EXPERIMENT/epoch020.pth
RESULTS=results/$LABEL
EVAL_LIST=dataset/pairs_480/eval_fixed_clean_lineart004.txt
LOG=logs/${LABEL}.eval.log
METRICS=results/fixed_output_metrics_${LABEL}_lineart004.csv
COMBINED_METRICS=results/fixed_output_metrics_exp1_moredupes_epoch020_lineart004_compare.csv
MONTAGE=results/compare_${LABEL}_lineart004.png
MULTI_MONTAGE=results/compare_exp1_moredupes_epoch020_lineart004.png
DONE_MARKER=logs/${LABEL}.done

mkdir -p logs results "$RESULTS"

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "waiting_for=$CKPT"
  until [[ -s "$CKPT" ]]; do
    if ! systemctl --user is-active --quiet "$UNIT"; then
      echo "$UNIT is not active before $CKPT appeared" >&2
      exit 1
    fi
    sleep 60
  done

  echo "checkpoint_ready_at=$(date --iso-8601=seconds)"
  echo "stopping $UNIT"
  systemctl --user stop "$UNIT" || true

  while read -r SAMPLE; do
    [[ -z "$SAMPLE" ]] && continue
    NAME="${SAMPLE%.jpg}"
    "$PY" scripts/inference.py \
      --checkpoint "$CKPT" \
      --input "dataset/pairs_480/test/rough/${NAME}.jpg" \
      --output "$RESULTS/${NAME}_out.png" \
      --autocontrast
  done < "$EVAL_LIST"

  "$PY" tools/evaluation/evaluate_fixed_outputs.py \
    --models "$LABEL" \
    --sample-list "$EVAL_LIST" \
    --split test \
    --output-csv "$METRICS"

  "$PY" tools/compare/make_model_eval_compare.py \
    --sample-list "$EVAL_LIST" \
    --model-dir "$RESULTS" \
    --model-label "$LABEL" \
    --split test \
    --output "$MONTAGE"

  "$PY" tools/compare/make_multi_model_eval_compare.py \
    --sample-list "$EVAL_LIST" \
    --model shape1_clean_split_bce=results/shape1_clean_split_bce_lineart004 \
    --model "$LABEL=$RESULTS" \
    --split test \
    --output "$MULTI_MONTAGE"

  "$PY" tools/evaluation/evaluate_fixed_outputs.py \
    --models shape1_clean_split_bce_lineart004 "$LABEL" \
    --sample-list "$EVAL_LIST" \
    --split test \
    --output-csv "$COMBINED_METRICS" || true

  {
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "experiment=$EXPERIMENT"
    echo "label=$LABEL"
    echo "checkpoint=$CKPT"
    echo "montage=$MULTI_MONTAGE"
    echo "metrics=$COMBINED_METRICS"
  } > "$DONE_MARKER"

  experiments/send_autoloop_notification.sh \
    "Lineart epoch020 evaluation complete" \
    "Review $MULTI_MONTAGE and $COMBINED_METRICS" || true
  echo "done marker: $DONE_MARKER"
} 2>&1 | tee -a "$LOG"
