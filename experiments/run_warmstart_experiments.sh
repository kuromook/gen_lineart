#!/bin/bash
# plan1 / plan2 を std15 からの真warm-start(低LR)で学習→推論する。
# plan1: std15(660) + ako5 ink[2%,25%] = 3158枚 / plan2: + ako5 ink[2%,15%] = 2824枚
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
PY=./venv/bin/python

RESUME="checkpoints/std15/best.pth"
LR=3e-5
EPOCHS=50
INFER_SAMPLES="lineart_004_002 lineart_004_004 lineart_004_006 lineart_004_008 lineart_004_010 housei_002_19_12 housei_002_07_12 housei_002_06_15"

run_one () {
    local PLAN=$1 LIST=$2
    local CKPT="checkpoints/${PLAN}"
    local LOG="logs/train_${PLAN}.log"
    local RESDIR="results/${PLAN}"
    echo "====== ${PLAN} 学習開始: $(date) ======"
    $PY scripts/train.py \
        --file-list "$LIST" \
        --checkpoint-dir "$CKPT" \
        --resume "$RESUME" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --autocontrast 2>&1 | tee "$LOG"

    echo "--- ${PLAN} 推論開始: $(date) ---"
    mkdir -p "$RESDIR"
    for NAME in $INFER_SAMPLES; do
        if [[ $NAME == housei* ]]; then INPUT="dataset/pairs_480/train/rough/${NAME}.jpg"
        else INPUT="dataset/pairs_480/test/rough/${NAME}.jpg"; fi
        $PY scripts/inference.py --checkpoint "${CKPT}/best.pth" \
            --input "$INPUT" --output "${RESDIR}/${NAME}_out.png" --autocontrast
    done
    echo "====== ${PLAN} 完了: $(date) ======"
}

run_one warm1 dataset/pairs_480/valid_train_warm_plan1.txt
run_one warm2 dataset/pairs_480/valid_train_warm_plan2.txt
echo "全warm-start実験完了: $(date)"
