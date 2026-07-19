#!/bin/bash
# Train selected follow-up candidates and build one shared clean-lineart004 montage.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-model_followup_e5}
EPOCHS=${EPOCHS:-5}
CANDIDATES=${CANDIDATES:-"resnet_gan_binary resnet_gan_advsharp"}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
BASE_CKPT=${BASE_CKPT:-checkpoints/shape1_clean_split_bce/best.pth}
ADVSHARP_CKPT=${ADVSHARP_CKPT:-checkpoints/model_resnet_sharp_e5_resnet_gan_advsharp/best.pth}
LOG=logs/${TAG}.log
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png
DONE=logs/${TAG}.done

mkdir -p logs results
: > "$LOG"

run_candidate() {
  local label=$1
  shift
  local ckpt=checkpoints/${TAG}_${label}
  local out=results/${TAG}_${label}
  mkdir -p "$out"
  echo "=== train ${label} ===" | tee -a "$LOG"
  "$PY" scripts/train_i2i_survey.py \
    --checkpoint-dir "$ckpt" \
    --file-list "$TRAIN_LIST" \
    --epochs "$EPOCHS" \
    --workers 0 \
    "$@" 2>&1 | tee -a "$LOG"

  echo "=== infer ${label} ===" | tee -a "$LOG"
  while read -r sample; do
    [[ -z "$sample" ]] && continue
    name="${sample%.jpg}"
    "$PY" scripts/inference_i2i.py \
      --checkpoint "$ckpt/best.pth" \
      --input "dataset/pairs_480/test/rough/${name}.jpg" \
      --output "$out/${name}_out.png" \
      --autocontrast 2>&1 | tee -a "$LOG"
  done < "$EVAL_LIST"
}

run_named_candidate() {
  case "$1" in
    unet_gan)
      run_candidate unet_gan \
        --model unet --gan \
        --resume-generator "$BASE_CKPT" --strict-resume \
        --lr 1e-5 --lr-d 2e-5
      ;;
    unet_skip50)
      run_candidate unet_skip50 \
        --model unet_skip50 \
        --resume-generator "$BASE_CKPT" --strict-resume \
        --lr 1e-5
      ;;
    resnet_gan)
      run_candidate resnet_gan \
        --model resnet --gan \
        --lr 2e-4 --lr-d 2e-5 --pos-weight 3.0
      ;;
    resnet_gan_binary)
      run_candidate resnet_gan_binary \
        --model resnet --gan \
        --lr 2e-4 --lr-d 2e-5 \
        --pos-weight 5.0 \
        --bce-weight 0.85 --l1-weight 0.05 \
        --shape-weight 0.08 --ink-weight 0.08 \
        --adv-weight 0.02
      ;;
    resnet_gan_advsharp)
      run_candidate resnet_gan_advsharp \
        --model resnet --gan \
        --lr 2e-4 --lr-d 3e-5 \
        --pos-weight 5.0 \
        --bce-weight 0.75 --l1-weight 0.05 \
        --shape-weight 0.08 --ink-weight 0.06 \
        --adv-weight 0.05
      ;;
    resnet_gan_advsharp_binft)
      run_candidate resnet_gan_advsharp_binft \
        --model resnet --gan \
        --resume-generator "$ADVSHARP_CKPT" --strict-resume \
        --lr 5e-5 --lr-d 2e-5 \
        --pos-weight 5.0 \
        --bce-weight 0.75 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.08 \
        --binary-weight 0.08 --adv-weight 0.04
      ;;
    resnet_gan_advsharp_bin04)
      run_candidate resnet_gan_advsharp_bin04 \
        --model resnet --gan \
        --resume-generator "$ADVSHARP_CKPT" --strict-resume \
        --lr 5e-5 --lr-d 2e-5 \
        --pos-weight 5.0 \
        --bce-weight 0.75 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.08 \
        --binary-weight 0.04 --adv-weight 0.04
      ;;
    resnet_gan_advsharp_bin08_ink10)
      run_candidate resnet_gan_advsharp_bin08_ink10 \
        --model resnet --gan \
        --resume-generator "$ADVSHARP_CKPT" --strict-resume \
        --lr 5e-5 --lr-d 2e-5 \
        --pos-weight 5.0 \
        --bce-weight 0.75 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.10 \
        --binary-weight 0.08 --adv-weight 0.04
      ;;
    resnet_gan_advsharp_bin12_ink12)
      run_candidate resnet_gan_advsharp_bin12_ink12 \
        --model resnet --gan \
        --resume-generator "$ADVSHARP_CKPT" --strict-resume \
        --lr 5e-5 --lr-d 2e-5 \
        --pos-weight 5.0 \
        --bce-weight 0.75 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.12 \
        --binary-weight 0.12 --adv-weight 0.04
      ;;
    resnet)
      run_candidate resnet \
        --model resnet \
        --lr 2e-4 --pos-weight 3.0
      ;;
    *)
      echo "unknown candidate: $1" >&2
      exit 2
      ;;
  esac
}

compare_args=(
  --sample-list "$EVAL_LIST"
  --model shape1_clean_split_bce=results/shape1_clean_split_bce_lineart004
  --model milddup800=results/shape1_clean_split_bce_milddup800_ft10_lr1e5
)
metric_models=(
  shape1_clean_split_bce_lineart004
  shape1_clean_split_bce_milddup800_ft10_lr1e5
)

for candidate in $CANDIDATES; do
  run_named_candidate "$candidate"
  compare_args+=(--model "${candidate}=results/${TAG}_${candidate}")
  metric_models+=("${TAG}_${candidate}")
done

"$PY" tools/compare/make_multi_model_eval_compare.py \
  "${compare_args[@]}" \
  --split test \
  --output "$MONTAGE" 2>&1 | tee -a "$LOG"

"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models "${metric_models[@]}" \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$METRICS" 2>&1 | tee -a "$LOG"

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "candidates=$CANDIDATES"
  echo "train_list=$TRAIN_LIST"
  echo "eval_list=$EVAL_LIST"
  echo "metrics=$METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart model follow-up complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
