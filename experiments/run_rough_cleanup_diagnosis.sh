#!/bin/bash
# Test rough input cleansing before atari generation.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-rough_cleanup_diag_shuffle60}
SAMPLE_LIST=${SAMPLE_LIST:-dataset/pairs_480/atari_halo_diag_shuffle60.txt}
ROUGH_DIR=${ROUGH_DIR:-dataset/pairs_480/train/rough}
ATARI_CKPT=${ATARI_CKPT:-checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth}
LOG=logs/${TAG}.log
DONE=logs/${TAG}.done

RAW_ATARI=results/line_refiner_e2_atari_train
BG_ROUGH=results/${TAG}_rough_background
LINE_ROUGH=results/${TAG}_rough_line
BG_LINE_ROUGH=results/${TAG}_rough_background_line
LINE_BG_ROUGH=results/${TAG}_rough_line_background

BG_ATARI=results/${TAG}_atari_background
LINE_ATARI=results/${TAG}_atari_line
BG_LINE_ATARI=results/${TAG}_atari_background_line
LINE_BG_ATARI=results/${TAG}_atari_line_background

METRICS=results/haze_uncertainty_metrics_${TAG}.csv
MONTAGE=results/compare_${TAG}.png

mkdir -p logs results
: > "$LOG"

clean_rough() {
  local mode=$1
  local out_dir=$2
  echo "=== clean rough ${mode} ===" | tee -a "$LOG"
  "$PY" tools/preprocess/clean_rough_input.py \
    --file-list "$SAMPLE_LIST" \
    --input-dir "$ROUGH_DIR" \
    --output-dir "$out_dir" \
    --mode "$mode" 2>&1 | tee -a "$LOG"
}

make_atari() {
  local label=$1
  local rough_dir=$2
  local out_dir=$3
  echo "=== materialize atari ${label} ===" | tee -a "$LOG"
  "$PY" scripts/materialize_i2i_aux.py \
    --checkpoint "$ATARI_CKPT" \
    --file-list "$SAMPLE_LIST" \
    --rough-dir "$rough_dir" \
    --output-dir "$out_dir" \
    --autocontrast 2>&1 | tee -a "$LOG"
}

clean_rough background "$BG_ROUGH"
clean_rough line "$LINE_ROUGH"
clean_rough background_line "$BG_LINE_ROUGH"
clean_rough line_background "$LINE_BG_ROUGH"

make_atari background "$BG_ROUGH" "$BG_ATARI"
make_atari line "$LINE_ROUGH" "$LINE_ATARI"
make_atari background_line "$BG_LINE_ROUGH" "$BG_LINE_ATARI"
make_atari line_background "$LINE_BG_ROUGH" "$LINE_BG_ATARI"

"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --sample-list "$SAMPLE_LIST" \
  --split train \
  --models \
    raw_atari="$RAW_ATARI" \
    rough_background="$BG_ROUGH" \
    atari_background="$BG_ATARI" \
    rough_line="$LINE_ROUGH" \
    atari_line="$LINE_ATARI" \
    rough_background_line="$BG_LINE_ROUGH" \
    atari_background_line="$BG_LINE_ATARI" \
    rough_line_background="$LINE_BG_ROUGH" \
    atari_line_background="$LINE_BG_ATARI" \
  --output-csv "$METRICS" 2>&1 | tee -a "$LOG"

"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$SAMPLE_LIST" \
  --split train \
  --model raw_atari="$RAW_ATARI" \
  --model rough_bg="$BG_ROUGH" \
  --model atari_bg="$BG_ATARI" \
  --model rough_line="$LINE_ROUGH" \
  --model atari_line="$LINE_ATARI" \
  --model rough_bg_line="$BG_LINE_ROUGH" \
  --model atari_bg_line="$BG_LINE_ATARI" \
  --model rough_line_bg="$LINE_BG_ROUGH" \
  --model atari_line_bg="$LINE_BG_ATARI" \
  --output "$MONTAGE" 2>&1 | tee -a "$LOG"

{
  echo "tag=$TAG"
  echo "sample_list=$SAMPLE_LIST"
  echo "metrics=$METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

echo "done: $DONE" | tee -a "$LOG"
