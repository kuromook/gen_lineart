# Work Log - lineart generator

## 2026-07-11

### kuripデータセット取り込み・fine-tune開始

現在の目的は、新しい `~/dataset_kurip_v4.zip` の線画・下絵ペアを使って、
既存の実用モデル `shape1` を改善すること。

#### 入力データ

- zip: `~/dataset_kurip_v4.zip`
- manifest: `dataset_kurip_v4/manifest.json`
- 内容: 38ページ分の `*_sketch.jpg` / `*_line.jpg`
- 全ページサイズ: `4961x7016`
- sketch/line は同一サイズで、ページ対応済み
- housei より揺らぎは少ない想定

#### 実装・生成物

kurip専用の480pxタイル抽出スクリプトを追加した。

- 追加: `prepare_kurip_tiles.py`
- 同一ページ・同一座標から480pxタイルを抽出
- rough側は `ImageOps.autocontrast(cutoff=0)` を適用
- edge F1、chamfer、線密度、方向entropy、rough stdで候補を評価
- QC画像とCSVを生成
- `--save` 指定時のみ `dataset_480/train/{rough,line}` に保存

dry-runでは緩め条件で以下を確認した。

- raw candidates: 8,274
- accepted after dedup/page cap: 3,040
- QC:
  - `results/kurip_tiles_lenient_qc.png`
  - `results/kurip_tiles_lenient_qc_tail.png`
  - `results/kurip_tiles_lenient_qc_sample.png`

最終保存ではページ偏りを抑えるため `--max-per-page 60` を採用した。

- saved tiles: 2,280
- list: `dataset_480/valid_train_kurip.txt`
- csv: `results/kurip_tiles.csv`
- QC:
  - `results/kurip_tiles_qc.png`
  - `results/kurip_tiles_qc_tail.png`
  - `results/kurip_tiles_qc_sample.png`

既存の実用データリストと結合した。

- base: `dataset_480/valid_train_warm_regions.txt` = 1,078 pairs
- kurip: `dataset_480/valid_train_kurip.txt` = 2,280 pairs
- combined: `dataset_480/valid_train_warm_regions_kurip.txt` = 3,358 pairs
- missing rough/line files: 0

#### 学習

`shape1` からkurip追加データでfine-tuneを開始した。

- service: `lineart-kurip-ft.service`
- log: `train_kurip.log`
- checkpoint dir: `checkpoints_kurip`
- resume: `checkpoints_shape1/best.pth`
- file list: `dataset_480/valid_train_warm_regions_kurip.txt`
- epochs: 20
- batch size: 2
- batches/epoch: 1,679
- LR: `1e-5`
- `pos_weight`: 2.0
- `edge_weight`: 0.0
- `shape_weight`: 1.0
- `ink_weight`: 1.0
- autocontrast: enabled
- device: CUDA / RTX 3060

起動コマンド:

```bash
systemd-run --user --unit=lineart-kurip-ft --collect \
  --property=WorkingDirectory=/home/sh1/deepl/lineart \
  --property=StandardOutput=append:/home/sh1/deepl/lineart/train_kurip.log \
  --property=StandardError=append:/home/sh1/deepl/lineart/train_kurip.log \
  /home/sh1/deepl/lineart/venv/bin/python /home/sh1/deepl/lineart/train.py \
  --file-list dataset_480/valid_train_warm_regions_kurip.txt \
  --checkpoint-dir checkpoints_kurip \
  --resume checkpoints_shape1/best.pth \
  --epochs 20 \
  --lr 1e-5 \
  --pos-weight 2.0 \
  --edge-weight 0.0 \
  --shape-weight 1.0 \
  --ink-weight 1.0 \
  --autocontrast
```

現在状態:

- `systemctl --user is-active lineart-kurip-ft.service` は `active`
- `train_kurip.log` には学習設定と `epoch 1 〜 20` まで出力済み
- 2026-07-11 00時台時点では、まだ epoch 1 完了行は未出力
- 概算完了時間は開始から約2〜3.5時間、朝には完了見込み

確認コマンド:

```bash
systemctl --user status lineart-kurip-ft.service
tail -f train_kurip.log
ls -lh checkpoints_kurip/
```

#### 次の作業

1. 朝に `lineart-kurip-ft.service` と `train_kurip.log` を確認する
2. 完了していれば `checkpoints_kurip/best.pth` を採用候補として固定サンプル推論する
3. `shape1` と `kurip fine-tune` の比較画像を作る
4. 既存固定8サンプルに加えて、kurip由来サンプルでも目視比較する
5. 線の太り、余分な線、roughにない清書追加線の扱い、kuripへの過適合を確認する

### kuripペア不整合の確認とVLM選別

同一座標抽出で作ったkuripタイルは、`compare_kurip_balanced_dense_train_vs_shape1.png`
の確認でrough/lineの組が噛み合っていないことが判明した。汚れや網点の主因は、
下絵と線画のペア不整合を教師に混ぜたことと判断した。

追加した主なスクリプト:

- `diagnose_pair_alignment.py`: 新規データセットの同一座標前提が有効かを先に診断する
- `match_kurip_regions.py`: line 480px tileを基準にrough側の近傍translationを探索する
- `extract_kurip_matched_tiles.py`: マッチCSVからrough/lineタイルを保存する
- `vlm_review_kurip_matches.py`: Qwen3VL/Ollamaで候補ペアを二次判定する
- `make_kurip_matched_compare.py`: matched strict版の比較画像生成
- `make_kurip_vlm_accept_compare.py`: VLM accept版の比較画像生成

診断結果:

- `results/kurip_alignment_diagnosis.json`
- same-coordinateは `recommendation="needs_region_matching"`
- strict local match: 115 pairs
- relaxed local match: 896 candidates
- Qwen3VL review top500: accept 369 / reject 131

学習結果:

- `checkpoints_kurip_matched_strict_x4_noac/best.pth`
  - warm 1078 + strict 115 x4
  - fixed8では黒太りが強く、採用候補から外す
- `checkpoints_kurip_vlm_accept_top500_noac/best.pth`
  - warm 1078 + VLM accept 369
  - fixed8定量: F1 0.8710, ink_ratio 2.244
  - epoch010: F1 0.8602, ink_ratio 1.958
  - VLMでペア数は増えたが、fine-tuneではまだkurip線画側に寄りすぎる

現時点の結論:

- 新規データセットは、同一座標抽出の前に必ず `diagnose_pair_alignment.py` を通す
- kuripは同一座標ではなく、line tile固定でrough側を探索する
- VLMは候補ペアの意味的なrejectに有効
- 次の学習案は、VLM accept 369件を使いつつ kurip比率を15%前後に下げ、
  `lr=5e-5`, 10epoch程度で黒太りを抑える
