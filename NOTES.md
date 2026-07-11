# Lineart Generator — 実験ノート

## 現在実行中（2026-05-31 開始）

```
nohup ./experiments/run_std_experiments.sh &  # PID 101554
```

std 10 / 12 / 15 の3条件でスクラッチから学習 → 推論を順番に実行。

| 条件 | ファイルリスト | 枚数 | チェックポイント | ログ |
|---|---|---|---|---|
| std>=10 | valid_train_std10.txt | 2,703 | checkpoints/std10/ | train_std10.log |
| std>=12 | valid_train_std12.txt | 1,537 | checkpoints/std12/ | train_std12.log |
| std>=15 | valid_train_std15.txt |   660 | checkpoints/std15/ | train_std15.log |

推論結果: `results/std10/` / `results/std12/` / `results/std15/`  
進捗確認: `tail -5 logs/train_std10.log`

---

## データ品質の知見

### housei系roughの問題
- housei_001（test）: 撮影指示シート等が混入、評価に不向き
- housei_002〜018（train）: ラフが全体的に薄い（original std 3〜6）
- `PIL.ImageOps.autocontrast` でlineart系と同等の濃さに補正できる
- autocontrast後のstdで品質フィルタリングを実施（housei全系列対象）

### 各系列のstd分布（autocontrast後）
- `lineart_*` / `orig_*`: std >= 11 で問題なし
- `housei_*`: std < 10 が過半数（54%）

---

## スクリプト変更履歴

### scripts/train.py
- `AUTOCONTRAST_ROUGH = True` フラグ追加
- CLI引数化: `--file-list` / `--checkpoint-dir` / `--resume` / `--autocontrast`

### scripts/inference.py
- `--autocontrast` フラグ追加

---

## 学習履歴

| ラウンド | epoch | チェックポイント | loss | 備考 |
|---|---|---|---|---|
| 第1 | 1〜100 | checkpoints/base/epoch100.pth | - | 初期学習 |
| 第2 | 101〜200 | checkpoints/base/best.pth | 0.1562 | pos_weight:5, EDGE_WEIGHT:1.0, CosineAnnealing |
| std実験 | 1〜200×3 | checkpoints/std*/best.pth | 実行中 | autocontrast, スクラッチから |

---

## 課題

- UNet（L1+BCE+EdgeLoss）の出力がblurry → pix2pix GANのfine-tuneを検討
- housei_001テストセットは評価に使わない（lineart_004を使う）
