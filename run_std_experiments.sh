#!/bin/bash
# std 10 / 12 / 15 の3条件で順番に学習・推論を実行する

set -e
cd "$(dirname "$0")"

INFER_SAMPLES="lineart_004_002 lineart_004_004 lineart_004_006 lineart_004_008 lineart_004_010 housei_002_19_12 housei_002_07_12 housei_002_06_15"

for STD in 10 12 15; do
    echo "======================================"
    echo " STD >= ${STD}  開始: $(date)"
    echo "======================================"

    CKPT_DIR="checkpoints_std${STD}"
    LOG="train_std${STD}.log"
    RESULT_DIR="results/std${STD}"

    python train.py \
        --file-list "dataset_480/valid_train_std${STD}.txt" \
        --checkpoint-dir "${CKPT_DIR}" \
        --autocontrast \
        2>&1 | tee "${LOG}"

    echo "--- 推論開始: $(date) ---"
    mkdir -p "${RESULT_DIR}"
    for NAME in ${INFER_SAMPLES}; do
        EXT="jpg"
        # rough の場所を特定
        if [[ $NAME == housei* ]]; then
            INPUT="dataset_480/train/rough/${NAME}.${EXT}"
        else
            INPUT="dataset_480/test/rough/${NAME}.${EXT}"
        fi
        python inference.py \
            --checkpoint "${CKPT_DIR}/best.pth" \
            --input "${INPUT}" \
            --output "${RESULT_DIR}/${NAME}_out.png" \
            --autocontrast
    done

    echo "STD >= ${STD} 完了: $(date)"
    echo ""
done

echo "全実験完了"
