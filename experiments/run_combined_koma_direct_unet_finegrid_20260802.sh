#!/bin/bash
# Fine-grained (every 5 epochs, 5..60) fresh training run of the single-stage
# direct-regression U-Net on the same 1489-tile combined_koma_20260729 pool,
# to pin down the epoch-40-ish stability peak found by evaluating the
# 200-epoch run's own checkpoint trajectory (evaluate_stroke_stability.py at
# epoch020/040/060/.../200): long_component_ratio and components_per_1k_ink_px
# both peaked at epoch 40 and monotonically degraded from there through 200,
# while ink_ratio/F1 stayed roughly flat through ~160 before also declining --
# i.e. the model's stability peaks early and then erodes even as confidence
# keeps growing, contradicting the earlier "more epochs/more repetition
# stabilizes the wobble" hypothesis (already rejected by the 100ep vs 200ep
# comparison; this run resolves *where* the real optimum is, independently
# of that run's specific checkpoints).
#
# This is a fresh independent run (not resumed from the 200ep run's
# checkpoints) so it also serves as a replication check on the epoch-40
# finding, not just a finer readout of the same trajectory.
#
# Time budget: user has a hard deadline -- the Direction 4 ControlNet cron
# job (experiments/run_controlnet_direction4_longrun_20260803.sh) fires
# 2026-08-03 00:00 JST and needs this worktree back on diffusion-controlnet
# well before then. 60 epochs at the ~297s/epoch pace measured from the
# 200-epoch run is ~5h -- comfortable margin before the 00:00 cutoff when
# launched at 2026-08-02 15:30ish JST.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./venv/bin/python
TAG=combined_koma_direct_unet_finegrid_20260802
EPOCHS=${EPOCHS:-60}
SAVE_EVERY=${SAVE_EVERY:-5}
TRAIN_LIST=dataset/pairs_480/valid_train_combined_koma_20260729.txt
TRAIN_RAW_ROUGH=dataset/pairs_480/train/rough
TRAIN_LINE_DIR=dataset/pairs_480/train/line_combined_koma_20260729
EVAL_LIST=dataset/pairs_480/eval_clean_lineart004_8.txt
EVAL_RAW_ROUGH=dataset/pairs_480/test/rough

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}
OUT_ROOT=results/${TAG}_trajectory
FIXED_CSV=results/${TAG}_trajectory_fixed_metrics.csv
STAB_CSV=results/${TAG}_trajectory_stability_metrics.csv
MONTAGE=results/compare_${TAG}_trajectory.png

mkdir -p logs results "$CKPT" "$OUT_ROOT"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] combined_koma direct unet fine-grid (5..${EPOCHS} step ${SAVE_EVERY}) start"
echo "tag=$TAG epochs=$EPOCHS save_every=$SAVE_EVERY train_list=$TRAIN_LIST"

echo "=== train direct unet, ${EPOCHS} epochs, checkpoint every ${SAVE_EVERY} ==="
"$PY" scripts/train_i2i_survey.py \
  --checkpoint-dir "$CKPT" \
  --file-list "$TRAIN_LIST" \
  --rough-dir "$TRAIN_RAW_ROUGH" \
  --line-dir "$TRAIN_LINE_DIR" \
  --epochs "$EPOCHS" \
  --workers 0 \
  --model unet \
  --lr 8e-5 --pos-weight 3.0 \
  --bce-weight 0.8 --l1-weight 0.2 \
  --shape-weight 0.0 --ink-weight 0.0 \
  --edge-weight 0.5 \
  --save-every "$SAVE_EVERY" \
  --require-cuda

echo "=== infer + evaluate every saved checkpoint ==="
MODEL_NAMES=""
ep=$SAVE_EVERY
while [ "$ep" -le "$EPOCHS" ]; do
  epfmt=$(printf "%03d" "$ep")
  name="${TAG}_ep${epfmt}"
  MODEL_NAMES="$MODEL_NAMES $name"
  "$PY" scripts/inference_i2i_batch.py \
    --checkpoint "$CKPT/epoch${epfmt}.pth" \
    --file-list "$EVAL_LIST" \
    --rough-dir "$EVAL_RAW_ROUGH" \
    --output-dir "$OUT_ROOT/$name" \
    --autocontrast
  ln -sfn "$(pwd)/$OUT_ROOT/$name" "results/$name"
  ep=$((ep + SAVE_EVERY))
done

echo "=== fixed metrics (all checkpoints) ==="
"$PY" tools/evaluation/evaluate_fixed_outputs.py \
  --models $MODEL_NAMES \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$FIXED_CSV"

echo "=== stroke-stability metrics (all checkpoints) ==="
"$PY" tools/evaluation/evaluate_stroke_stability.py \
  --models $MODEL_NAMES \
  --sample-list "$EVAL_LIST" \
  --split test \
  --output-csv "$STAB_CSV"

echo "=== montage (subset: every other checkpoint to keep it readable) ==="
MONTAGE_ARGS=""
ep=$SAVE_EVERY
idx=0
while [ "$ep" -le "$EPOCHS" ]; do
  epfmt=$(printf "%03d" "$ep")
  name="${TAG}_ep${epfmt}"
  if [ $((idx % 2)) -eq 0 ]; then
    MONTAGE_ARGS="$MONTAGE_ARGS --model ep${epfmt}=$OUT_ROOT/$name"
  fi
  ep=$((ep + SAVE_EVERY))
  idx=$((idx + 1))
done
"$PY" tools/compare/make_multi_model_eval_compare.py \
  --sample-list "$EVAL_LIST" \
  --split test \
  $MONTAGE_ARGS \
  --output "$MONTAGE"

# CNN+GAN-family work runs on cleanup-refiner, but Monday's cron job
# (scripts/train_controlnet.py) only exists on diffusion-controlnet --
# switch back automatically since the user has a hard 00:00 JST deadline.
echo "=== switching back to diffusion-controlnet branch ==="
git checkout diffusion-controlnet

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "epochs=$EPOCHS"
  echo "save_every=$SAVE_EVERY"
  echo "fixed_csv=$FIXED_CSV"
  echo "stability_csv=$STAB_CSV"
  echo "montage=$MONTAGE"
} > "$DONE"

experiments/send_autoloop_notification.sh \
  "Lineart combined_koma direct unet fine-grid (5..${EPOCHS} step ${SAVE_EVERY}) trajectory complete" \
  "Review $MONTAGE, $FIXED_CSV, $STAB_CSV" || true

echo "done marker: $DONE"
echo "[$(date --iso-8601=seconds)] combined_koma direct unet fine-grid trajectory complete"
