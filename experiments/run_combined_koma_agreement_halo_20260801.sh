#!/bin/bash
# Re-test the agreement-halo hypothesis (rough/line correspondence quality
# drives the soft/marbled ceiling) on the current clean combined_koma_20260729
# data, following the same methodology as the earlier agreement_halo_e2
# survey (2026-07-20, dataset/pairs_480/valid_train_milddup800_clean.txt),
# which found training on high-agreement-only tiles beat every mixed-data
# candidate on F1@2px/recall while low-agreement-only training produced
# much fainter, less confident ink (halo_band_faint_ratio 0.53 vs 0.20).
#
# User hypothesis (2026-08-01): the soft/marbled ceiling's lack of
# confidence may come from mixing low rough-line correspondence tiles in
# with high-correspondence ones during training, forcing the model to
# hedge. agreement_halo_e2 already supports this on the old dataset; this
# reruns it on combined_koma_20260729 to confirm it still holds on the
# current clean data and against the current adopted baseline
# (combined_koma_lucy_mild_msgan_20260729).
#
# Speed trick: reuses the already-materialized atari/lucy_mild aux hints
# from the 2026-08-01 noadv ablation run (all 1489 tiles present) instead
# of regenerating them -- that preprocessing step alone took ~32 minutes
# last time, dwarfing the actual training time (~6 minutes for 3 epochs on
# all 1489 tiles with this small "cleanup" architecture).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_agreement_halo_20260801
EPOCHS=${EPOCHS:-2}
SPLIT_COUNT=${SPLIT_COUNT:-450}
BASE_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough
TRAIN_AUX=results/combined_koma_lucy_mild_noadv_20260801_lucy_mild_train
EVAL_AUX=results/combined_koma_lucy_mild_noadv_20260801_lucy_mild_eval
BASELINE_MODEL_NAME=combined_koma_lucy_mild_msgan_20260729
BASELINE_OUT=results/${BASELINE_MODEL_NAME}

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
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] combined_koma agreement halo re-test start"
echo "tag=$TAG epochs=$EPOCHS split_count=$SPLIT_COUNT base_list=$BASE_LIST reused_aux=$TRAIN_AUX"

if [[ ! -d "$TRAIN_AUX" || ! -d "$EVAL_AUX" ]]; then
  echo "FATAL: expected pre-materialized aux dirs missing: $TRAIN_AUX / $EVAL_AUX" >&2
  exit 1
fi

echo "=== score rough-line agreement (combined_koma_20260729) ==="
"$PY" tools/evaluation/score_pair_agreement.py \
  --file-list "$BASE_LIST" \
  --rough-dir dataset/pairs_480/train/rough \
  --line-dir "$LINE_DIR" \
  --output-csv "$AGREE_CSV" \
  --high-list "$HIGH_LIST" \
  --low-list "$LOW_LIST" \
  --count "$SPLIT_COUNT"

train_candidate() {
  local label=$1
  local file_list=$2
  local ckpt=checkpoints/${TAG}_${label}
  local out=results/${TAG}_${label}
  mkdir -p "$out"
  echo "=== train ${label} ($(wc -l < "$file_list") tiles) ==="
  "$PY" scripts/train_i2i_survey.py \
    --checkpoint-dir "$ckpt" \
    --file-list "$file_list" \
    --line-dir "$LINE_DIR" \
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
    --require-cuda

  echo "=== infer ${label} on eval list ==="
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "$ckpt/best.pth" \
    --file-list "$EVAL_LIST" \
    --rough-dir "$EVAL_RAW_ROUGH" \
    --aux-dir "$EVAL_AUX" \
    --output-dir "$out" \
    --autocontrast
}

train_candidate high_agreement "$HIGH_LIST"
train_candidate low_agreement "$LOW_LIST"

echo "=== montage ==="
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  --model lucy_mild_msgan_mixed="$BASELINE_OUT" \
  --model high_agree="results/${TAG}_high_agreement" \
  --model low_agree="results/${TAG}_low_agreement" \
  --output "$MONTAGE"

echo "=== fixed metrics ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models \
    "$BASELINE_MODEL_NAME" \
    ${TAG}_high_agreement \
    ${TAG}_low_agreement \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS"

echo "=== halo metrics ==="
"$PY" tools/evaluation/evaluate_halo_outputs.py \
  --models \
    "$BASELINE_MODEL_NAME" \
    ${TAG}_high_agreement \
    ${TAG}_low_agreement \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$HALO_METRICS"

# This experiment is CNN+GAN-family work and runs on cleanup-refiner, but
# the Monday 2026-08-03 00:00 JST cron job (scripts/train_controlnet.py)
# only exists on diffusion-controlnet -- switch back automatically since
# the user may not be present (asleep/session closed) when this finishes.
echo "=== switching back to diffusion-controlnet branch ==="
git checkout diffusion-controlnet

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "split_count=$SPLIT_COUNT"
  echo "base_list=$BASE_LIST"
  echo "high_list=$HIGH_LIST"
  echo "low_list=$LOW_LIST"
  echo "train_aux=$TRAIN_AUX (reused)"
  echo "eval_aux=$EVAL_AUX (reused)"
  echo "agreement_scores=$AGREE_CSV"
  echo "metrics=$METRICS"
  echo "halo_metrics=$HALO_METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart combined_koma agreement halo re-test complete" \
  "Review $MONTAGE, $METRICS, and $HALO_METRICS" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma agreement halo re-test complete"
