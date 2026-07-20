#!/bin/bash
# Split halo regeneration between aux conditioning strength and loss/model behavior.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-aux_loss_halo_diag_shuffle60}
SAMPLE_LIST=${SAMPLE_LIST:-dataset/pairs_480/atari_halo_diag_shuffle60.txt}
ROUGH_DIR=dataset/pairs_480/train/rough
BASE_AUX=${BASE_AUX:-results/lucy_mask_deep_e2_lucy_thin_train}
LOG=logs/${TAG}.log
DONE=logs/${TAG}.done

LUCY_THIN_CKPT=checkpoints/lucy_mask_deep_e2_lucy_thin_aux_msgan/best.pth

mkdir -p logs results
: > "$LOG"

run_batch() {
  local label=$1
  local ckpt=$2
  local aux_dir=$3
  local out_dir=results/${TAG}_${label}
  echo "=== infer ${label} ===" | tee -a "$LOG"
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "$ckpt" \
    --file-list "$SAMPLE_LIST" \
    --rough-dir "$ROUGH_DIR" \
    --aux-dir "$aux_dir" \
    --output-dir "$out_dir" \
    --autocontrast 2>&1 | tee -a "$LOG"
}

echo "=== aux conditioning strength variants ===" | tee -a "$LOG"
for mode in identity weak75 weak50 hard20 hard35 softcut20 blur open; do
  aux_dir=results/${TAG}_aux_${mode}
  if [[ "$mode" == "identity" ]]; then
    aux_dir="$BASE_AUX"
  else
    "$PY" tools/evaluation/transform_aux_strength.py \
      --input-dir "$BASE_AUX" \
      --output-dir "$aux_dir" \
      --mode "$mode" 2>&1 | tee -a "$LOG"
  fi
  run_batch "aux_${mode}" "$LUCY_THIN_CKPT" "$aux_dir"
done

AUX_METRICS=results/halo_metrics_${TAG}_aux_strength.csv
"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --sample-list "$SAMPLE_LIST" \
  --split train \
  --models \
    base_aux="$BASE_AUX" \
    aux_identity=results/${TAG}_aux_identity \
    aux_weak75=results/${TAG}_aux_weak75 \
    aux_weak50=results/${TAG}_aux_weak50 \
    aux_hard20=results/${TAG}_aux_hard20 \
    aux_hard35=results/${TAG}_aux_hard35 \
    aux_softcut20=results/${TAG}_aux_softcut20 \
    aux_blur=results/${TAG}_aux_blur \
    aux_open=results/${TAG}_aux_open \
  --output-csv "$AUX_METRICS" 2>&1 | tee -a "$LOG"

AUX_MONTAGE=results/compare_${TAG}_aux_strength.png
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$SAMPLE_LIST" \
  --split train \
  --model base_aux="$BASE_AUX" \
  --model identity=results/${TAG}_aux_identity \
  --model weak75=results/${TAG}_aux_weak75 \
  --model weak50=results/${TAG}_aux_weak50 \
  --model hard20=results/${TAG}_aux_hard20 \
  --model hard35=results/${TAG}_aux_hard35 \
  --model softcut20=results/${TAG}_aux_softcut20 \
  --model blur=results/${TAG}_aux_blur \
  --model open=results/${TAG}_aux_open \
  --output "$AUX_MONTAGE" 2>&1 | tee -a "$LOG"

echo "=== reconstruction/loss variants with fixed lucy_thin aux ===" | tee -a "$LOG"
run_batch loss_cleanup_msgan checkpoints/line_refiner_msgan_e2_cleanup_msgan_fm/best.pth "$BASE_AUX"
run_batch loss_bin12 checkpoints/line_refiner_tight_e2_refiner_unet_bin12_ink14/best.pth "$BASE_AUX"
run_batch loss_bin20 checkpoints/line_refiner_tight_e2_refiner_unet_bin20_ink16/best.pth "$BASE_AUX"
run_batch loss_struct08 checkpoints/line_refiner_structure_e2_cleanup_struct08/best.pth "$BASE_AUX"
run_batch loss_struct08_msgan checkpoints/line_refiner_structure_e2_cleanup_struct08_msgan/best.pth "$BASE_AUX"
run_batch loss_width06 checkpoints/line_refiner_width_e2_cleanup_width06/best.pth "$BASE_AUX"
run_batch loss_halo_high checkpoints/halo_loss_e2_high_halo10_faint04/best.pth "$BASE_AUX"

LOSS_METRICS=results/halo_metrics_${TAG}_loss_compare.csv
"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --sample-list "$SAMPLE_LIST" \
  --split train \
  --models \
    base_aux="$BASE_AUX" \
    loss_cleanup_msgan=results/${TAG}_loss_cleanup_msgan \
    loss_bin12=results/${TAG}_loss_bin12 \
    loss_bin20=results/${TAG}_loss_bin20 \
    loss_struct08=results/${TAG}_loss_struct08 \
    loss_struct08_msgan=results/${TAG}_loss_struct08_msgan \
    loss_width06=results/${TAG}_loss_width06 \
    loss_halo_high=results/${TAG}_loss_halo_high \
  --output-csv "$LOSS_METRICS" 2>&1 | tee -a "$LOG"

LOSS_MONTAGE=results/compare_${TAG}_loss_compare.png
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$SAMPLE_LIST" \
  --split train \
  --model base_aux="$BASE_AUX" \
  --model cleanup_msgan=results/${TAG}_loss_cleanup_msgan \
  --model bin12=results/${TAG}_loss_bin12 \
  --model bin20=results/${TAG}_loss_bin20 \
  --model struct08=results/${TAG}_loss_struct08 \
  --model struct08_msgan=results/${TAG}_loss_struct08_msgan \
  --model width06=results/${TAG}_loss_width06 \
  --model halo_high=results/${TAG}_loss_halo_high \
  --output "$LOSS_MONTAGE" 2>&1 | tee -a "$LOG"

{
  echo "tag=$TAG"
  echo "sample_list=$SAMPLE_LIST"
  echo "base_aux=$BASE_AUX"
  echo "aux_metrics=$AUX_METRICS"
  echo "aux_montage=$AUX_MONTAGE"
  echo "loss_metrics=$LOSS_METRICS"
  echo "loss_montage=$LOSS_MONTAGE"
} > "$DONE"

echo "done: $DONE" | tee -a "$LOG"
