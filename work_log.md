# Work Log — lineart generator (480px branch)

---

## 2026-06-11

### warm_regions推論・定量評価

固定8サンプルでstd15、warm2、warm_regionsを比較した。

| model | F1@2px | chamfer | ink ratio | precision | recall |
|---|---:|---:|---:|---:|---:|
| std15 | 0.8848 | 1.146 | 2.188 | 0.8240 | 0.9895 |
| warm2 | 0.8396 | 1.381 | 2.239 | 0.7598 | 0.9685 |
| warm_regions | 0.8810 | 1.171 | 2.221 | 0.8176 | 0.9896 |

- warm_regionsはstd15に近く、目視ではわずかにガタつきが減った
- 正解線画への大きな改善はない
- 全モデルが正解の約2.2倍のインク量
- recall約0.99に対してprecisionが低く、主課題は欠落より太線・余分な線
- 既存`edge_loss`はOpenCV Canny前に予測をdetachしており、学習勾配に寄与しない

### ako5 pair抽出の成果

warm_regionsによる線画品質の改善は限定的だったが、今回の主要成果はモデル改善量
そのものではなく、状態の良くない原稿群から信頼できる下絵・線画pairを抽出できた
ことにある。

- ページ対応が不確実で、清書時の内容変更・削除・局所変形を含む原稿を対象にした
- 全ページ組合せ探索と局所領域マッチングにより、対応する人物・コマ領域を特定した
- region/tile単位の再評価と保守的な閾値により、別番号ページの誤pairを除外した
- 最終的に高信頼な480px pairを418件抽出し、学習データとして利用できた

線画改善が小さかったことはpair抽出の失敗を意味しない。低品質・不均一な生原稿から
学習可能な対応pairを自動抽出するパイプラインを確立できたことは大きな前進であり、
今後別原稿へ展開して高信頼データを増やすための基盤になる。

### shape1 fine-tune開始

太線化抑制の短期実験をGPU user systemdサービスで開始した。

- resume: `checkpoints_warm_regions/best.pth`
- data: warm_regions 1,078 pair
- epoch: 20、LR: `1e-5`
- `pos_weight`: 5.0から2.0へ低減
- 非微分Canny lossを無効化
- 2px位置ずれを許容する微分可能soft F1 lossを追加
- 総インク量lossを追加
- service: `lineart-shape1.service`
- log: `train_shape1.log`
- 完了後、固定8サンプルを`results/shape1`へ自動推論
- 2026-06-11 19:32時点: epoch 3/20、loss=0.3095、GPUで継続中

---

### ako5局所領域タイル抽出 完了

`extract_ako5_region_tiles.py` の全候補dry-runが遅かったため、以下を最適化した。

- 元ページ全体のsupport mask warpを廃止
  - 元画像四隅をタイル座標へ変換し、480px maskへ直接描画
  - 旧方式との差はsupport比で最大0.16%
- SciPy `distance_transform_edt` をOpenCV exact distance transformへ変更
  - 数値差は最大`5e-7`未満、単体ベンチマークで約12.6倍高速
- support、ink、std、edge数、entropyの安価な棄却をdistance transformより前へ移動
- 20領域ごとの進捗表示を追加
- 上位、下位、全順位均等sampleの3種類のQCモンタージュを追加

初回dry-run:

- 142領域、閾値通過2,117タイル、重複除去後1,151タイル
- tail QCで位置ずれ、内容変更、大きい黒ベタを含む低品質タイルを確認

保守的な採用閾値へ変更:

- `min_tile_score >= 3.0`
- `line_ink <= 0.15`

最終結果:

- 142領域、閾値通過758タイル、重複除去後418タイル
- 34ページペア、別番号ページペア0件
- score範囲: 3.000〜4.774、median 3.327
- edge F1範囲: 0.586〜0.831、median 0.643
- QCを確認し、初回warm-start用の保守的データとして採用

出力:

- `dataset_480/valid_train_ako5_regions.txt`（418枚）
- `dataset_480/train/rough/ako5r_*.jpg`
- `dataset_480/train/line/ako5r_*.jpg`
- `results/ako5_region_tiles.csv`
- `results/ako5_region_tiles_qc.png`
- `results/ako5_region_tiles_qc_tail.png`
- `results/ako5_region_tiles_qc_sample.png`

### warm_regions実験 開始

`build_warm_lists.py` にregion専用リスト生成を追加した。

- 元の高品質base: 非ako5 std15 660枚
- 新しいregionペア: 418枚
- 合計: 1,078枚、重複なし、旧`ako5_*.jpg`混入なし
- リスト: `dataset_480/valid_train_warm_regions.txt`

学習条件:

- resume: `checkpoints_warm_regions/epoch010.pth`
- checkpoint dir: `checkpoints_warm_regions`
- LR: `3e-5`
- epoch: 11〜50
- autocontrast有効
- epoch 10 loss: `0.2815`

通常のPTY学習をepoch 10完了後に停止し、セッション切断後も継続するよう
`nohup`をuser systemdサービス内で起動した。

現在のサービス:

```bash
systemctl --user status lineart-warm-regions.service
tail -f train_warm_regions.log
```

停止する場合:

```bash
systemctl --user stop lineart-warm-regions.service
```

### 次回再開地点

1. `systemctl --user status lineart-warm-regions.service` で学習状態を確認
2. `tail -50 train_warm_regions.log` で最新epoch/lossを確認
3. 完了後、`checkpoints_warm_regions/best.pth` で固定テスト画像を推論
4. `checkpoints_std15/best.pth`、`checkpoints_warm2/best.pth` と比較
5. regionペア追加が線画品質を改善したか目視評価する

---

## 2026-06-10

### 現在の最優先課題: ako5原稿ペアの特定と位置合わせ

モデル改善を一旦止め、ako5原稿から正しく対応する下絵・線画ページを特定して
位置合わせすることを優先する。

- manifest上のページ対応は信用せず、全下絵ページ×全線画ページから候補を検索する
- 当面扱う変換は平行移動・等方拡大縮小・微小回転
- 局所的・非線形な変形は、残差の悪いタイルを棄却して対応する
- ページ候補検索後、ページ全体を位置合わせし、最後にタイル単位で品質判定する

追加スクリプト: `match_ako5_pages.py`

- SIFT対応点 + RANSAC similarity transformで候補を粗検索
- SIFTで拾えない候補も、scale・微小回転のグリッド探索 + 位相相関で救済
- 上位候補をエッジのチャンファー距離で再ランキング
- inlier率、空間被覆率、scale、angle、1位と2位のmarginを記録
- 歩留まり優先のため、デフォルトでは全候補をアライメント後に再評価する

初期検証（先頭6下絵ページ）:

- 正しい同番号ペア: 5ページでTop-2以内、4ページでTop-1
- `page0004 -> page0004` は53位かつ内容も不一致で、誤ペアと判定可能
- 元解像度に適用できる変換行列をJSONへ保存する

### 局所領域ペア探索

ページ全体の一致を要求すると、清書時に変更・削除された部分の影響が大きい。
一方、480pxタイル単独では単純な曲線やコマ枠への偶然一致が多かったため、
中間サイズの局所領域から対応ペアを抽出する方式へ移行した。

追加スクリプト: `match_ako5_regions.py`

- 全下絵ページ×全線画ページでSIFT対応点を検索
- RANSACのinlierを順次取り除き、1ページ組から複数のsimilarity変換候補を抽出
- 変換候補を固定サイズ局所窓で評価し、対応する人物・コマ単位の領域を候補化
- edge F1、truncated symmetric chamfer、inlier空間被覆率、線方向entropyを記録
- 元解像度用の変換行列と切り出し座標をCSV/JSONへ保存
- CSVに目視判定用の`decision`、`notes`欄を用意

初期検証:

- 先頭15下絵×25線画の375組を探索し、上位50局所領域をQC画像で確認
- 上位50件はすべて同番号ページ内の部分一致で、別ページへの明確な誤一致はなし
- 人物・顔・衣服・コマ単位の対応領域が上位に抽出された
- ページ全体では不一致部分があっても、局所窓内の対応部分は抽出可能

全ページ探索（`max-dim=800`）:

- 全53×53 = 2,809ページ組を探索
- RANSAC・局所窓評価で513候補、重複除去後162候補
- 39下絵ページ・39線画ページから局所対応領域を抽出
- 162件中、同番号ページ内候補161件、別番号候補1件
- 唯一の別番号候補は最下位、score=1.44
- `score>=3.5`: 155件、別番号候補なし
- 上位80件をQC画像で確認し、人物・コマ単位の対応が概ね成立

出力:

- `results/ako5_region_matches.csv`
- `results/ako5_region_matches.json`
- `results/ako5_region_match_qc.png`

初回の自動採用候補には保守的に`score>=4.0`を使い、CSVの目視ラベルで
precisionを確認してから閾値を下げる。局所変形の判定は引き続き保留する。

### 次回再開地点: 局所領域から480px学習ペアを抽出

追加スクリプト: `extract_ako5_region_tiles.py`

- `results/ako5_region_matches.json` の `score>=4.0` 局所領域を入力にする
- 元解像度の線画側領域を、重なり付き480px窓で走査する
- `full_matrix` で下絵を線画座標へワープして学習ペア候補を生成する
- 各タイルを edge F1、symmetric chamfer、線密度、方向entropyで再評価する
- 重複タイルを除去し、CSVとQCモンタージュを生成する
- デフォルトはdry-runで、`--save`を付けるまで学習データには保存しない

現在の状態:

- スクリプト実装と `py_compile` は完了
- 全候補dry-runを開始したが処理が遅いため中断
- 学習データへの書き込みは未実施
- 遅い原因は、各480px候補ごとに元ページ全体サイズのsupport maskを
  `warpAffine`していること

次回やること:

1. `warp_tile()` のsupport生成を高速化する
   - ページ全体maskのwarpを廃止する
   - 逆変換したタイル四隅、または小さい480px maskだけで有効領域を判定する
2. `python extract_ako5_region_tiles.py` でdry-runを再実行する
3. `results/ako5_region_tiles_qc.png` を目視し、タイル採用閾値を調整する
4. QCが良好なら `--save` で高信頼ペアを保存する
5. 保存後に新しい学習リストでwarm-start実験を行う

---

## 2026-06-02

### std実験 完了・結論

3条件（std10 / std12 / std15）の epoch200 学習・推論がすべて完了。

| 条件 | 枚数 | best loss | 評価 |
|------|------|-----------|------|
| std>=10 | 2,703 | 0.2549 | 実用外 |
| std>=12 | 1,537 | 0.2583 | 実用外 |
| std>=15 | 660   | 0.1705 | **採用** |

比較画像: `results/compare_std_conditions.png`

**結論:** 右（std15）に行くほど品質が良く、std10/12 は実用範囲に入らない。
データが少なくても高品質サンプルのみで学習する方が有効。

### 採用モデル

`checkpoints_std15/best.pth`（loss=0.1705）

### 現状の課題

- std15 フィルタリングで660枚しかなく、データ量が不足
- augmentation（hflip/rotation/crop）は補助として有効だが根本解決にはならない
- **方針: 生データ追加を優先する**

### 次セッションでやること

1. 生データ追加（housei系など std>=15 相当の高品質ラフ画像を収集）
2. `data_check.py` 等で新データの std を確認してフィルタリング
3. `valid_train_std15.txt` を更新
4. `checkpoints_std15/best.pth` から resume して追加学習（または再学習）
5. augmentation 追加は任意（`train.py` の `SketchDataset` に paired hflip/rotation を追加）

---

## 過去の経緯（参考）

### GAN試行
- GAN1（checkpoints_gan/ep130）: 動作するが線がぼやける
- GAN2: LR_D=2e-5 + N_CRITIC=2 → epoch10からmode collapse → 廃棄

### UNet第2ラウンドまでの問題
- checkpoints/best.pth（loss=0.1562）: 出力がblurry、線が鮮明でない
- L1+BCE損失の限界、データ品質のばらつきが原因と判断
- → std閾値によるデータ品質フィルタリングを導入

### データ品質の知見
- housei系roughは全体的に薄い（autocontrast前 std 3〜6）
- autocontrastで補正しても薄いものは薄い → std>=15 が実用ライン
- housei_001（testセット）はラフでなく撮影指示シート等が混入 → 評価不適
- lineart_* / orig_* のroughは品質問題なし（std>=11）
