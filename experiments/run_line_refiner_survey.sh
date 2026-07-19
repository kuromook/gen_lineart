#!/bin/bash
# Train 2-channel line refiners using rough + fixed ResNet-GAN atari output.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=${1:-line_refiner_e2}
EPOCHS=${EPOCHS:-2}
CANDIDATES=${CANDIDATES:-"refiner_unet refiner_skip50"}
SKIP_AUX=${SKIP_AUX:-0}
TRAIN_LIST=${TRAIN_LIST:-dataset/pairs_480/valid_train_milddup800_clean.txt}
EVAL_LIST=${EVAL_LIST:-dataset/pairs_480/eval_clean_lineart004_8.txt}
ATARI_CKPT=${ATARI_CKPT:-checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth}
TRAIN_AUX=${TRAIN_AUX:-results/${TAG}_atari_train}
EVAL_AUX=${EVAL_AUX:-results/${TAG}_atari_eval}
LOG=logs/${TAG}.log
METRICS=results/fixed_output_metrics_${TAG}_compare.csv
MONTAGE=results/compare_${TAG}.png
DONE=logs/${TAG}.done

mkdir -p logs results
: > "$LOG"

materialize_aux() {
  local list=$1
  local rough_dir=$2
  local out_dir=$3
  if [[ "$SKIP_AUX" == "1" ]]; then
    echo "=== skip aux ${out_dir} ===" | tee -a "$LOG"
    return
  fi
  echo "=== materialize aux ${out_dir} ===" | tee -a "$LOG"
  "$PY" scripts/materialize_i2i_aux.py \
    --checkpoint "$ATARI_CKPT" \
    --file-list "$list" \
    --rough-dir "$rough_dir" \
    --output-dir "$out_dir" \
    --autocontrast 2>&1 | tee -a "$LOG"
}

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
    --aux-dir "$TRAIN_AUX" \
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
      --aux-input "$EVAL_AUX/${name}_out.png" \
      --output "$out/${name}_out.png" \
      --autocontrast 2>&1 | tee -a "$LOG"
  done < "$EVAL_LIST"
}

run_named_candidate() {
  case "$1" in
    refiner_unet)
      run_candidate refiner_unet \
        --model unet \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.85 --l1-weight 0.04 \
        --shape-weight 0.08 --ink-weight 0.10 \
        --binary-weight 0.08 --adv-weight 0.0
      ;;
    refiner_skip50)
      run_candidate refiner_skip50 \
        --model unet_skip50 \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.85 --l1-weight 0.04 \
        --shape-weight 0.08 --ink-weight 0.10 \
        --binary-weight 0.08 --adv-weight 0.0
      ;;
    refiner_unet_bin12_ink14)
      run_candidate refiner_unet_bin12_ink14 \
        --model unet \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.85 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.14 \
        --binary-weight 0.12 --adv-weight 0.0
      ;;
    refiner_unet_bin20_ink16)
      run_candidate refiner_unet_bin20_ink16 \
        --model unet \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.85 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.16 \
        --binary-weight 0.20 --adv-weight 0.0
      ;;
    refiner_unet_bin28_ink18)
      run_candidate refiner_unet_bin28_ink18 \
        --model unet \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.85 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.18 \
        --binary-weight 0.28 --adv-weight 0.0
      ;;
    refiner_unet_skel03)
      run_candidate refiner_unet_skel03 \
        --model unet \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.80 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.14 \
        --binary-weight 0.12 --skeleton-weight 0.03 \
        --adv-weight 0.0
      ;;
    refiner_unet_skel06)
      run_candidate refiner_unet_skel06 \
        --model unet \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.75 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.14 \
        --binary-weight 0.12 --skeleton-weight 0.06 \
        --adv-weight 0.0
      ;;
    refiner_unet_skel10)
      run_candidate refiner_unet_skel10 \
        --model unet \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.70 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.14 \
        --binary-weight 0.12 --skeleton-weight 0.10 \
        --adv-weight 0.0
      ;;
    cleanup_bin12)
      run_candidate cleanup_bin12 \
        --model cleanup \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.85 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.14 \
        --binary-weight 0.12 --adv-weight 0.0
      ;;
    cleanup_skel06)
      run_candidate cleanup_skel06 \
        --model cleanup \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.75 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.14 \
        --binary-weight 0.12 --skeleton-weight 0.06 \
        --adv-weight 0.0
      ;;
    cleanup_ink18)
      run_candidate cleanup_ink18 \
        --model cleanup \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.85 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.18 \
        --binary-weight 0.16 --adv-weight 0.0
      ;;
    refiner_unet_msgan_fm)
      run_candidate refiner_unet_msgan_fm \
        --model unet --gan --multiscale-gan \
        --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
        --bce-weight 0.75 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.12 \
        --binary-weight 0.08 --adv-weight 0.03 \
        --feature-match-weight 0.08
      ;;
    cleanup_msgan_fm)
      run_candidate cleanup_msgan_fm \
        --model cleanup --gan --multiscale-gan \
        --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
        --bce-weight 0.75 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.14 \
        --binary-weight 0.10 --skeleton-weight 0.03 \
        --adv-weight 0.03 --feature-match-weight 0.08
      ;;
    refiner_unet_width06)
      run_candidate refiner_unet_width06 \
        --model unet \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.85 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.14 \
        --binary-weight 0.12 --width-weight 0.06 \
        --adv-weight 0.0
      ;;
    cleanup_width06)
      run_candidate cleanup_width06 \
        --model cleanup \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.85 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.14 \
        --binary-weight 0.12 --width-weight 0.06 \
        --adv-weight 0.0
      ;;
    cleanup_width12)
      run_candidate cleanup_width12 \
        --model cleanup \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.85 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.16 \
        --binary-weight 0.12 --width-weight 0.12 \
        --adv-weight 0.0
      ;;
    refiner_unet_struct08)
      run_candidate refiner_unet_struct08 \
        --model unet \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.80 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.14 \
        --binary-weight 0.12 --structure-weight 0.08 \
        --adv-weight 0.0
      ;;
    cleanup_struct08)
      run_candidate cleanup_struct08 \
        --model cleanup \
        --lr 1e-4 --pos-weight 5.0 \
        --bce-weight 0.80 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.14 \
        --binary-weight 0.12 --structure-weight 0.08 \
        --adv-weight 0.0
      ;;
    cleanup_struct08_msgan)
      run_candidate cleanup_struct08_msgan \
        --model cleanup --gan --multiscale-gan \
        --lr 8e-5 --lr-d 2e-5 --pos-weight 5.0 \
        --bce-weight 0.75 --l1-weight 0.03 \
        --shape-weight 0.08 --ink-weight 0.14 \
        --binary-weight 0.10 --structure-weight 0.08 \
        --adv-weight 0.03 --feature-match-weight 0.08
      ;;
    *)
      echo "unknown candidate: $1" >&2
      exit 2
      ;;
  esac
}

materialize_aux "$TRAIN_LIST" dataset/pairs_480/train/rough "$TRAIN_AUX"
materialize_aux "$EVAL_LIST" dataset/pairs_480/test/rough "$EVAL_AUX"

compare_args=(
  --sample-list "$EVAL_LIST"
  --model milddup800=results/shape1_clean_split_bce_milddup800_ft10_lr1e5
  --model "resnet_atari=$EVAL_AUX"
)
metric_models=(
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
  echo "atari_checkpoint=$ATARI_CKPT"
  echo "train_aux=$TRAIN_AUX"
  echo "eval_aux=$EVAL_AUX"
  echo "train_list=$TRAIN_LIST"
  echo "eval_list=$EVAL_LIST"
  echo "metrics=$METRICS"
  echo "montage=$MONTAGE"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart 2ch refiner survey complete" \
  "Review $MONTAGE and $METRICS" || true

echo "done marker: $DONE"
