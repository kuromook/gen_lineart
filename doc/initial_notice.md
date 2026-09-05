# Track A: SD1.5路線 — 最良構成からの詰め

作成: 2026-09-06 JST / ブランチ: `controlnet-sd15-refine`
前身: `lineart-controlnet-realpairs`(クロスハッチ脱却を達成して終了)
起案: `doc/track_proposal_20260906.md`(このtrackにも同梱)
前身の全経緯: `doc/track_controlnet_realpairs_work_log.md`

このファイルは「現状・次の一手・運用ルール」だけを短く保つ。
実験の時系列は`doc/work_log.md`に追記すること(前trackでは引き継ぎ文書が
作業ログを兼ねて900行超まで肥大化し機能しなくなった。その反省)。

## 出発点

**到達済みの最良構成**:
- チェックポイント:
  `../lineart-controlnet-realpairs/checkpoints/controlnet_lora_manga_consistency_20260904/final`
- 推論: `--controlnet-conditioning-scale 3.5`
- 実測: gt_bsds_f1 **0.2337** / ink_ratio **0.0772** / line_width_p50 **3.34**
  (GT参照: ink_ratio 0.0353 / line_width_p50 3.72)

この構成は`scripts/train_controlnet_consistency.py`(x0推定をVAEデコードして
GT画像とのSobelエッジ一致度L1損失を補助項に追加)で学習したもの。
スクリプトはこのworktreeの`scripts/`に入っている。

**データ**: このworktreeの`data/`に**複製済み**(2026-09-06、ユーザー判断)。
前trackの`../lineart-controlnet-realpairs/data/`(約1GB)からの`cp -a`で、
ファイル一覧の一致を検証済み。ペアデータ一式(`train_list.txt` 8,467件、
`line/`、`rough_manga_line/`、`captions.csv`、`captions_no_manga_word.csv`、
`diag_valid5.txt`、`diag_rough_*`)が揃っている。
`data/`は`.gitignore`対象なのでコミットには乗らない。
以後は前trackのディレクトリを参照せず、この`data/`を使うこと
(前trackは終了済みで、いずれ整理される可能性がある)。

なお共通基盤側にはこれとは別に、より新しい8,798タイルのv3プール
(`../lineart/dataset/pairs_480/valid_train_combined_v3_20260830.txt`)がある。
ただし2026-08-30のユーザー指示で前trackにはv3を渡さない方針だったため、
ここにあるのはv2ベースのスナップショット。v3での再学習は別途判断すること。

## 課題

最良構成でも、GTの白背景・黒線に対し**背景がグレー・線もグレー寄り**。
cs4.0以上に上げると線自体が薄れて消える(cs5.0で剣がほぼ消失)ため、
csをさらに上げる方向では埋まらない。

## 次の一手(優先度順)

1. **consistency損失のハイパラスイープ**(最有力)。効果があることは確定
   したが`consistency_weight=0.1`・`consistency_max_timestep=200`の
   **1点しか試していない**。
2. **InnerControl方式への拡張**。[arxiv 2507.02321](https://arxiv.org/abs/2507.02321) /
   [github.com/ControlGenAI/InnerControl](https://github.com/ControlGenAI/InnerControl)。
   ControlNet++系の一致度損失が「最終デノイズステップのみ」に適用される
   限界を指摘し、全timestepの中間UNet特徴からの条件再構成に拡張して改善を
   報告している。実装済みのconsistency損失(`max_timestep=200`)は
   まさにその「最終ステップ付近のみ」版に相当するので直接の発展形。
3. **出力の二値性/コントラストへの介入**。グレー残差そのものへの対処。

## やってはいけないこと(前trackで検証済みの地雷)

- **ベースUNetにLoRAを足さない**。ベースラインより全csで劣ったうえ、
  **csレバー自体を壊した**(他モデルと違いcsを上げると単調悪化)。
  UNet凍結は維持する。
- **LoRA rankを上げない**。rank32は唯一cs上昇で単調悪化し最適域で最下位級。
- **ネガティブプロンプトで"hatching"等を禁止しない**。逆効果で、
  ハッチの代わりにベタ塗りへ逃げる(`line_width_p50` 14.85)。
- **エポックを増やさない**。2エポックと10エポックは誤差の範囲で同一。
- **キャプション語彙をいじらない**。頻度0.5%以上の63語を網羅スイープ済み、
  効果は誤差程度。

## 運用ルール

- 大きな学習・抽出は必ずスモークテストしてから本番投入
- バックグラウンド実行は`nohup ... & disown`+PPID=1確認で完全に切り離す
- 数値指標だけで判断しない、必ず目視モンタージュを作り、**その画像の
  正確なパスを数値と一緒に併記する**
- **モデル評価を`controlnet_conditioning_scale=1.0`だけで行わない**。
  cs1.0はハッチが支配的でモデル間の差が潰れる領域。最適域(cs2.5〜3.5)を
  含む複数csで評価すること
- **`orientation_entropy`単体で判断しない**。ハッチ網目と滑らかなベタ塗りの
  境界を区別できない。`line_width_p50`(GT≈3.7)と`ink_ratio`(GT≈0.035)を
  必ず併記する。実例: 再評価で`coarse_trained`はf1 0.2101と好成績に見えたが
  `line_width_p50`が40.92で実体はベタ塗りだった
- 実験の結論が出たら、その場で`results/`の生成物を要否判断する
