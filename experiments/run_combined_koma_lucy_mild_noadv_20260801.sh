#!/bin/bash
# Ablation of the lucy_mild_aux_msgan recipe: same residual-anchor
# architecture (--model cleanup) and same atari/lucy_mild aux hint, but with
# the adversarial/feature-matching/shape/ink/binary/structure loss terms
# all removed, replaced with the plain-regression recipe from the
# pre-leak-fix era (notebooks/gen_lineart.ipynb): BCE(pos_weight=3) + L1 +
# Canny edge_loss, no GAN at all.
#
# Motivation (doc/work_log.md 2026-07-31/08-01, user-prompted): the
# earliest model (leaky-era, unusable eval numbers) reportedly produced
# more genuinely line-art-like (crisper, less soft/marbled) output than the
# now-adopted lucy_mild_aux_msgan, despite having none of the current
# recipe's adversarial/structure-loss machinery. Today's complex recipe
# accumulated through fixes for leak and region-alignment problems in the
# *data*, not necessarily because the extra loss terms were proven
# necessary on their own merits -- now that combined_koma_20260729 is
# clean data, this tests whether just removing the adversarial term alone
# (holding architecture, LR, and data fixed) recovers crisper output.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_lucy_mild_noadv_20260801
EPOCHS=${EPOCHS:-3}
TRAIN_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
TRAIN_RAW_ROUGH=dataset/pairs_480/train/rough
TRAIN_LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough
ATARI_CKPT=checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png

ATARI_TRAIN=results/${TAG}_atari_train
ATARI_EVAL=results/${TAG}_atari_eval
LUCY_MILD_TRAIN_AUX=results/${TAG}_lucy_mild_train
LUCY_MILD_EVAL_AUX=results/${TAG}_lucy_mild_eval

CKPT=checkpoints/${TAG}
OUT=results/${TAG}

# Compare directly against the adopted lucy_mild_aux_msgan checkpoint's
# existing eval outputs (same eval list), not a fresh baseline inference.
BASELINE_MODEL_NAME=combined_koma_lucy_mild_msgan_20260729
BASELINE_OUT=results/${BASELINE_MODEL_NAME}

mkdir -p logs results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] combined_koma lucy_mild noadv (plain regression ablation) start"
echo "tag=$TAG epochs=$EPOCHS train_list=$TRAIN_LIST line_dir=$TRAIN_LINE_DIR"

echo "=== materialize atari aux (train + eval) ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$ATARI_CKPT" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --output-dir "$ATARI_TRAIN" \
  --autocontrast
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$ATARI_CKPT" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --output-dir "$ATARI_EVAL" \
  --autocontrast

echo "=== preprocess lucy_mild aux (train + eval) ==="
"$PY" scripts/preprocess_atari_aux.py \
  --input-dir "$ATARI_TRAIN" --output-dir "$LUCY_MILD_TRAIN_AUX" --mode lucy_mild
"$PY" scripts/preprocess_atari_aux.py \
  --input-dir "$ATARI_EVAL" --output-dir "$LUCY_MILD_EVAL_AUX" --mode lucy_mild

echo "=== train lucy_mild_aux_noadv (plain BCE+L1+edge, no GAN) on koma data ==="
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "$CKPT" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --line-dir "$TRAIN_LINE_DIR" \
  --aux-dir "$LUCY_MILD_TRAIN_AUX" \
  --epochs "$EPOCHS" \
  --workers 0 \
  --model cleanup \
  --lr 8e-5 --pos-weight 3.0 \
  --bce-weight 0.8 --l1-weight 0.2 \
  --shape-weight 0.0 --ink-weight 0.0 \
  --edge-weight 0.5 \
  --require-cuda

echo "=== infer lucy_mild_aux_noadv (koma) on eval list ==="
"$PY" scripts/inference_i2i_batch.py \
  --checkpoint "$CKPT/best.pth" \
  --file-list "$EVAL_LIST" \
  --rough-dir "$EVAL_RAW_ROUGH" \
  --aux-dir "$LUCY_MILD_EVAL_AUX" \
  --output-dir "$OUT" \
  --autocontrast

echo "=== montage ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model lucy_mild_msgan="$BASELINE_OUT" \
  --model lucy_mild_noadv="$OUT" \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    "$BASELINE_MODEL_NAME" \
    "$TAG" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "train_list=$TRAIN_LIST"
  echo "train_line_dir=$TRAIN_LINE_DIR"
  echo "atari_checkpoint=$ATARI_CKPT"
  echo "montage=$MONTAGE"
  echo "metrics=$METRICS"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart combined_koma lucy_mild noadv (plain regression) ablation complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma lucy_mild noadv ablation complete"
