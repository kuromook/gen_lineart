# Work Log — lineart generator (480px branch)

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
