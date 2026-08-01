# Lineart Generator

ラフ画像（手書きスケッチ）から線画を生成する深層学習プロジェクト。

---

## 概要

UNet ベースのモデルでラフ画像を線画に変換します。  
BCE、形状系 loss、インク量 loss などを切り替えながら、ラフと線画の対応を学習します。

現在は過去の train/eval leakage を前提から外し、clean evaluation に基づいてベースモデルを作り直している段階です。旧 `shape1` や `std15` の高評価は採用基準として使いません。現状は `doc/CURRENT.md` を参照してください。

---

## 環境

- Ubuntu 22.04
- RTX 3060 (12GB) / CUDA 12.8
- Python 3.10

```bash
pip install torch torchvision pillow opencv-python
```

---

## データセット構成

```
dataset/
  raw/        # 元データ
  pairs/      # 旧 256x256 ペア
  pairs_256/  # 256x256 生成ペア
  pairs_480/  # 480x480 ペア（現在の主学習データ）
    train/
      rough/  # ラフ画像（入力）
      line/   # 線画画像（正解）
    test/
      rough/
      line/
    *.txt      # 学習・評価用ファイルリスト
```

生データからの抽出・保存・監査ルールは `doc/preprocess/EXTRACTION_RULES.md` にまとめています。新しい raw dataset は、同一座標抽出を行う前に alignment 診断と QC を通してください。

---

## 学習

```bash
python scripts/train.py \
  --file-list dataset/pairs_480/<train_list>.txt \
  --checkpoint-dir checkpoints/<experiment_name> \
  --autocontrast
```

**主なオプション:**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--file-list` | `dataset/pairs_480/valid_train.txt` | 学習ファイルリスト |
| `--checkpoint-dir` | `checkpoints/base` | チェックポイント保存先 |
| `--resume` | なし | 途中再開するチェックポイントパス |
| `--autocontrast` / `--no-autocontrast` | True | rough に autocontrast を適用 |

チェックポイントは 10 epoch ごとと best loss 更新時に保存されます。

---

## 推論

```bash
python scripts/inference.py \
  --checkpoint checkpoints/<experiment_name>/best.pth \
  --input path/to/rough.jpg \
  --output results/output.png \
  --autocontrast
```

---

## 現在の評価状態

旧実験結果は leakage を含む可能性があるため、トップレベルの採用判断から外しました。

現在の clean lineart004 比較:

| model | F1@2px | chamfer | ink_ratio |
|---|---:|---:|---:|
| `shape1_clean_split_bce_lineart004` | 0.2955 | 7.174 | 1.316 |
| `shape1_base_clean_unique_bce` | 0.2786 | 7.718 | 1.600 |

この比較では `shape1_base_clean_unique_bce` は改善していません。次の確認対象は `shape1_std15_clean_split_moredupes_bce` の epoch020 clean eval です。

最新の結果入口:

- `doc/CURRENT.md`
- `results/CURRENT.md`

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `lineart/unetgenerator.py` | UNetGenerator（ResBlock, DilatedConvBlock）|
| `lineart/losses.py` | edge / tolerant F1 / ink loss |
| `scripts/train.py` | 学習スクリプト |
| `scripts/inference.py` | 推論スクリプト |
| `tools/evaluation/audit_pair_dataset_integrity.py` | train/eval leakage と欠損の監査 |
| `tools/evaluation/evaluate_fixed_outputs.py` | 固定サンプル評価 |
| `tools/pair_extraction/` | raw pair 抽出・alignment・clean list 生成 |
| `experiments/` | 実験ランナー |
