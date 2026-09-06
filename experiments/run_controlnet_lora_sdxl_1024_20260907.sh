#!/bin/bash
# Track B main run: retrain the SDXL ControlNet LoRA at 1024, the base's
# native resolution (doc/initial_notice.md next-step #1).
#
# Every previous SDXL result in this project was trained AND inferred at
# 512 -- CLI defaults inherited from the SD1.5-era scripts -- against a
# 1024-native animagine-xl-3.1. This run puts SDXL on its own footing for
# the first time.
#
# Fitting 1024 in 12GB (measured 2026-09-06, see doc/work_log.md):
#   naive 1024        OOM inside the fp32 VAE encoder, before the UNet
#   cached 1024       8.05GiB peak, ~9.4s/step at batch 1 / grad-accum 4
#   (uncached 512     9.79GiB peak, ~9.9s/step -- the old footing)
# The difference is scripts/cache_sdxl_conditioning.py: the VAE and both
# text encoders are frozen functions of (tile, caption), so they are
# computed once into --cache-dir and never occupy the GPU during training.
# That buys back more memory than 4x the pixels costs, so 1024 is both
# cheaper in VRAM and no slower per step than the 512 runs it replaces.
#
# batch-size 1 / grad-accum 4 are kept identical to the 512 runs
# (run_controlnet_lora_sdxl_20260829.sh) so resolution stays the single
# changed variable against the historical baseline.
#
# 10 epochs = 21,168 steps ~= 55h. Launched Monday it lands Wednesday
# afternoon, inside the Mon-Thu batch window. save-steps 500 (~1.3h) keeps
# a day-3 crash from costing more than the last hour and a bit; the run is
# resumable in place with --resume-from-checkpoint latest.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/sh1/deepl/lineart/venv/bin/python

# anime = noob-sdxl-controlnet-lineart_anime + data/rough_lineart_coarse
# manga = noob-sdxl-controlnet-manga_line   + data/rough_manga_line
# anime, decided by results/resolution_sweep_20260906 + grey_source_cross_20260906
# (2026-09-06): lineart_anime is the only ControlNet of the two that tracks the
# conditioning at all -- manga_line comes out striped and blocky at 512 and
# 1024 alike, with a near-white page because it draws almost nothing.
VARIANT=${VARIANT:-anime}
EPOCHS=${EPOCHS:-10}
LORA_RANK=${LORA_RANK:-16}
LR=${LR:-1e-4}
RESOLUTION=${RESOLUTION:-1024}

SDXL_BASE=/home/sh1/.cache/huggingface/hub/models--cagliostrolab--animagine-xl-3.1/snapshots/483f0c322568ed13697ed01dd0be07204746d12b
CN_ANIME=/home/sh1/.cache/huggingface/hub/models--Eugeoter--noob-sdxl-controlnet-lineart_anime/snapshots/61ed2d40710b32a5a1c9873f7dec89ff0af9f2a4
CN_MANGA=/home/sh1/.cache/huggingface/hub/models--Eugeoter--noob-sdxl-controlnet-manga_line/snapshots/bc7619904de6489ba7e171cf27a80f0e12822943

case "$VARIANT" in
  anime) CN_INIT=$CN_ANIME; ROUGH_DIR=data/rough_lineart_coarse; DIAG_ROUGH=data/diag_rough_lineart_coarse ;;
  manga) CN_INIT=$CN_MANGA; ROUGH_DIR=data/rough_manga_line;     DIAG_ROUGH=data/diag_rough_manga_line ;;
  *) echo "ERROR: VARIANT must be anime or manga, got '$VARIANT'" >&2; exit 1 ;;
esac

TAG=controlnet_lora_sdxl_${VARIANT}_1024_20260907
FILE_LIST=data/train_list.txt
LINE_DIR=data/line
CAPTION_CSV=data/captions.csv
CAPTION="monochrome line art, manga panel, black and white"
# The cache holds VAE latents of data/line plus text embeddings -- neither
# depends on which conditioning preprocessor VARIANT selects, so both
# variants share one cache.
CACHE_DIR=data/cache_sdxl_${RESOLUTION}

LOG=logs/${TAG}.log
DONE=logs/${TAG}.done
CKPT=checkpoints/${TAG}

mkdir -p logs checkpoints
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

# This runs unattended for ~55h. Without this trap a failure -- a failed smoke
# test, an OOM on hour 30, a bad argument -- would exit quietly under
# `set -e` and send nothing, because the success notification only fires at
# the very end. Silence would then be indistinguishable from "still training"
# until someone opened the log days later.
on_failure() {
  local code=$?
  local line=$1
  echo "[$(date --iso-8601=seconds)] FAILED at line $line (exit $code)"
  echo "failed_at=$(date --iso-8601=seconds) line=$line exit=$code" > "logs/${TAG}.failed"
  /home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
    "SDXL 1024 fine-tune FAILED (exit $code)" \
    "line $line -- see logs/${TAG}.log" || true
}
trap 'on_failure $LINENO' ERR
rm -f "logs/${TAG}.failed"

echo "[$(date --iso-8601=seconds)] SDXL ControlNet LoRA ${RESOLUTION} start"
echo "variant=$VARIANT controlnet=$CN_INIT rough=$ROUGH_DIR epochs=$EPOCHS rank=$LORA_RANK lr=$LR"
echo "file count: $(wc -l < "$FILE_LIST")"

echo "=== conditioning cache (resumable; skips tiles already present) ==="
"$PY" scripts/cache_sdxl_conditioning.py \
  --file-list "$FILE_LIST" \
  --line-dir "$LINE_DIR" \
  --base-ckpt "$SDXL_BASE" \
  --caption-csv "$CAPTION_CSV" \
  --resolution "$RESOLUTION" \
  --batch-size 2 \
  --out-dir "$CACHE_DIR"

echo "=== smoke test (6 steps, throwaway output dir) ==="
rm -rf /tmp/sdxl_1024_smoke
"$PY" scripts/train_controlnet_sdxl.py \
  --file-list "$FILE_LIST" --rough-dir "$ROUGH_DIR" --line-dir "$LINE_DIR" \
  --base-ckpt "$SDXL_BASE" --controlnet-init "$CN_INIT" --cache-dir "$CACHE_DIR" \
  --controlnet-lora-rank "$LORA_RANK" --resolution "$RESOLUTION" \
  --output-dir /tmp/sdxl_1024_smoke \
  --max-train-steps 6 --log-steps 1 --save-steps 10000 \
  --batch-size 1 --grad-accum 4 --lr "$LR"
rm -rf /tmp/sdxl_1024_smoke
echo "smoke test passed"

echo "=== full fine-tune ($EPOCHS epochs) ==="
"$PY" scripts/train_controlnet_sdxl.py \
  --file-list "$FILE_LIST" --rough-dir "$ROUGH_DIR" --line-dir "$LINE_DIR" \
  --base-ckpt "$SDXL_BASE" --controlnet-init "$CN_INIT" --cache-dir "$CACHE_DIR" \
  --controlnet-lora-rank "$LORA_RANK" --resolution "$RESOLUTION" \
  --output-dir "$CKPT" --epochs "$EPOCHS" --lr "$LR" \
  --batch-size 1 --grad-accum 4 --save-steps 500 --log-steps 50 \
  --resume-from-checkpoint latest

echo "=== eval: 1024 inference across a cs ladder ==="
# Two things this ladder has to respect, both established 2026-09-06:
#  * "SDXL gets worse as cs rises" was a property of the 512-trained
#    anime LoRA alone, not of SDXL -- the bare ControlNet climbs to a broad
#    plateau at cs 2.0-3.0 (f1 0.2568/0.2582/0.2577) and only falls at 4.0.
#  * paper white appears abruptly between cs1.0 and cs2.0 (near_white 3.0%
#    -> 81.5%), and that interval is unsampled, hence 1.5.
# The bare ControlNet is re-run here at the same scales, not just cited from
# the sweep: it is the baseline this fine-tune has to beat, and running it in
# the same job removes any doubt that the two were measured differently.
for CS in 0.5 1.0 1.5 2.0 2.5 3.0; do
  OUT="results/${TAG}_eval/cs${CS}"
  mkdir -p "$OUT"
  "$PY" scripts/infer_controlnet_sdxl.py \
    --sample-list data/diag_valid5.txt --rough-dir "$DIAG_ROUGH" \
    --controlnet-dir "$CN_INIT" --controlnet-lora-dir "$CKPT/final" \
    --base-ckpt "$SDXL_BASE" --caption "$CAPTION" \
    --resolution "$RESOLUTION" --controlnet-conditioning-scale "$CS" \
    --cpu-offload --tag "${TAG}_cs${CS}" --output-dir "$OUT"
  touch "$OUT/.complete"

  BASE_OUT="results/${TAG}_eval/base_cs${CS}"
  if [ ! -f "$BASE_OUT/.complete" ]; then
    mkdir -p "$BASE_OUT"
    "$PY" scripts/infer_controlnet_sdxl.py \
      --sample-list data/diag_valid5.txt --rough-dir "$DIAG_ROUGH" \
      --controlnet-dir "$CN_INIT" \
      --base-ckpt "$SDXL_BASE" --caption "$CAPTION" \
      --resolution "$RESOLUTION" --controlnet-conditioning-scale "$CS" \
      --cpu-offload --tag "${TAG}_base_cs${CS}" --output-dir "$BASE_OUT"
    touch "$BASE_OUT/.complete"
  fi
done

echo "=== scoring: fine-tuned vs bare, on every axis ==="
"$PY" experiments/score_sdxl_1024_eval_20260907.py "results/${TAG}_eval" || true

{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "tag=$TAG"
  echo "variant=$VARIANT"
  echo "resolution=$RESOLUTION"
  echo "epochs=$EPOCHS"
  echo "lora_rank=$LORA_RANK"
  echo "lr=$LR"
  echo "controlnet_init=$CN_INIT"
  echo "rough_dir=$ROUGH_DIR"
  echo "cache_dir=$CACHE_DIR"
  echo "checkpoint=$CKPT/final"
  echo "eval_dir=results/${TAG}_eval"
} > "$DONE"

/home/sh1/deepl/lineart/experiments/send_autoloop_notification.sh \
  "SDXL ControlNet LoRA ${RESOLUTION} (${VARIANT}) complete" \
  "Review results/${TAG}_eval/" || true

echo "[$(date --iso-8601=seconds)] SDXL ControlNet LoRA ${RESOLUTION} complete"
