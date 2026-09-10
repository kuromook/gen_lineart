# Track A: SD1.5路線 — 最良構成からの詰め

作成: 2026-09-06 JST / ブランチ: `controlnet-sd15-refine`
前身: `lineart-controlnet-realpairs`(クロスハッチ脱却を達成して終了)
起案: `doc/track_proposal_20260906.md`(このtrackにも同梱)
前身の全経緯: `doc/track_controlnet_realpairs_work_log.md`

このファイルは「現状・次の一手・運用ルール」だけを短く保つ。
実験の時系列は`doc/work_log.md`に追記すること(前trackでは引き継ぎ文書が
作業ログを兼ねて900行超まで肥大化し機能しなくなった。その反省)。

## 出発点

**到達済みの最良構成**(2026-09-10更新、`consistency_weight`スイープの結果。
詳細は`doc/work_log.md`「2026-09-08/10 (Track A)」参照):
- チェックポイント:
  `checkpoints/controlnet_lora_manga_consistency_w0.2_20260908/final`
  (このworktree自身の`checkpoints/`。`consistency_weight=0.2`、
  `consistency_max_timestep=200`は旧最良点と同じ)
- 推論: `--controlnet-conditioning-scale 2.5`
- 実測: gt_bsds_f1 **0.2354** / near_white_frac **0.779** / ink_ratio **0.088**
  / line_width_p50 **3.17**
  (GT参照: near_white_frac 0.948 / ink_ratio 0.0353 / line_width_p50 3.72)
- ユーザーがモンタージュ目視で確認済み(2026-09-10、「もっとも線画がちかい」)

旧最良点(`consistency_weight=0.1`、`cs=3.5`、gt_bsds_f1 0.2337、
near_white_frac 0.400)は上位互換で置き換え。チェックポイント自体は
`../lineart-controlnet-realpairs/checkpoints/controlnet_lora_manga_consistency_20260904/final`
に残っており削除はしていない。

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

`consistency_weight=0.2`(cs2.5)への切り替えで背景の白さは大きく改善した
(near_white_frac 0.400→0.779)が、GT(0.948)にはまだ届いていない — **グレー
残差は縮小しただけで解決していない**。cs4.0以上に上げると線自体が薄れて
消える(cs5.0で剣がほぼ消失)ため、csをさらに上げる方向では埋まらない。

### Track B(SDXL)からの申し送り(2026-09-06)

`../lineart-controlnet-sdxl-fidelity`の測定から、この「グレー残差」に
直接効く材料が2つ来ている。詳細は共通基盤の
`../lineart/inbox/note_sdxl_resolution_findings_20260906.md`。

**1. グレー残差は目視でなく数値で追える。**
`gt_bsds_f1`は「GTの線の近くに線があるか」しか見ないので、下地が紙か
灰色かを一切区別しない。3軸が共通基盤の
`tools/evaluation/measure_lineart_profile.py`に`paper_profile()`として
取り込み済み(そのまま使える):

- `bg_mode` … 背景の最頻輝度 (GT 255)
- `near_white_frac` … 224以上の画素比率 (GT 94.8%、v3プール実測0.951)
- `midtone_frac` … 64〜192の画素比率 (GT 1.8%、v3プール実測0.014)

f1が高いのに線画でない実例2件: 灰色一色の出力がf1 0.2263(near_white
3.1%、midtone 84.8%)、ほぼ空白のページがf1 0.2121(白87.8%だが被写体が
描かれていない)。**`near_white_frac`単体でも読めない**——「きれいだから
白い」と「描けていないから白い」を区別できないので、`ink_ratio`・
`line_width_p50`・目視モンタージュと必ずセットで読むこと。

**2. グレーはControlNetチェックポイント固有の性質かもしれない。**
SDXL側では2×2交差(ControlNet × 前処理)で帰属が確定した:
`lineart_anime`は条件画像を替えても解像度を変えても灰色のまま
(near_white 0.031→0.030)、`manga_line`は常に白い。**つまり学習で
消そうとしていたものが、実はチェックポイントの持ち物だった可能性がある。**

→ **2026-09-08/10に実施済み**(SD1.5には`manga_line`ControlNetが存在しない
ため2×3グリッドに変更)。結果: 方向としてはSDXLと同じくControlNet起因寄り
(`cnLineart`はnear_white 0.010〜0.041、`cnAnime`は0.119〜0.252で行が
分離)だが、SDXLほど綺麗な分離ではなく前処理の副次効果も残る。加えて
SD1.5はどちらのControlNetもGT白紙には遠く及ばない(素のControlNetは
ink_ratioがGTの10倍以上)。詳細は`doc/work_log.md`「2026-09-08/10」参照。

**3. 解像度とcsは交互作用する。** SDXLでは、cs1.0のまま解像度を上げても
横ばい、cs2.0にして初めて単調改善した。紙の白さも1024かつcs2.0が揃って
初めて跳ねた(near_white 3.0%→81.5%)。1変数ずつ隔離するのは基本として
維持しつつ、**交互作用が疑われる軸は格子で当たること**。

## 次の一手(優先度順)

`consistency_weight`スイープは2026-09-08/10に完了し、`weight=0.2`・`cs=2.5`
が新最良点として採用済み(詳細: `doc/work_log.md`「2026-09-08/10」、
上の「出発点」セクション)。ただしnear_white_fracはGTの0.948に対しまだ
0.779どまりで、グレー残差自体は解決していない。次の一手:

1. **第2段consistency_weightスイープ**(0.2周辺をより密に、例えば
   0.1〜0.3を刻む)。今回の結果は0.5で一度下がる非単調な挙動を示しており、
   0.2近傍にもう少し情報がある可能性が高い。前回同様1変数隔離を維持。
2. **InnerControl方式への拡張**。[arxiv 2507.02321](https://arxiv.org/abs/2507.02321) /
   [github.com/ControlGenAI/InnerControl](https://github.com/ControlGenAI/InnerControl)。
   ControlNet++系の一致度損失が「最終デノイズステップのみ」に適用される
   限界を指摘し、全timestepの中間UNet特徴からの条件再構成に拡張して改善を
   報告している。実装済みのconsistency損失(`max_timestep=200`)は
   まさにその「最終ステップ付近のみ」版に相当するので直接の発展形。
3. **出力の二値性/コントラストへの介入**。グレー残差そのものへの対処。

### 共通基盤からの申し送り: 5枚問題は放置しないこと(2026-09-10)

本trackの`note_sd15_consistency_weight_result_20260910.md`は「より広い
タイル集合での再検証は今のところ予定していない」としているが、**予定して
ほしい**。新最良構成という重い結論が診断5枚(`data/diag_valid5.txt`)に
載ったままで、Track Bは同じ問題を自覚して既に対処を用意している。

**Track Bの292タイル・プロトコルをそのまま流用できる。** 検証済みの事実:

- 両trackの`data/train_list.txt`は**完全に同一の8,467行**(照合済み)。
  したがってTrack Bの「train_listとの重複0件」という検証はTrack Aにも
  そのまま成立する——**本trackでも重複0件を確認済み**(192/100とも)。
- リストは共通基盤に配置済み:
  `../lineart/dataset/pairs_480/holdout_lineart_family.txt` (192)、
  `../lineart/dataset/pairs_480/holdout_housei_100.txt` (100)。
  両trackが同じリストを使えば**測定値が直接比較可能**になる。
- 設計の要点(踏襲する価値がある): 単純な無作為抽出をせず2群に分ける。
  **群A 192**=診断5枚と同系統の出典(主問「5枚の結論は192枚でも成り立つか」)、
  **群B 100**=housei・別出典(副問「系統を越えて一般化するか」)。
  共通基盤のtest splitはhousei 1,596 + lineart_004 24という偏った構成なので、
  そこから無作為に300件引くと**規模拡大と出典変更が同時に起きて交絡する**。
  また診断5枚は群Aに含まれるので、**同一ラン内で過去値とのアンカー照合**が
  できる。
- 実装の雛形:
  `../lineart-controlnet-sdxl-fidelity/experiments/run_holdout_validation_20260912.sh`
  と`score_holdout_validation_20260912.py`。推論のみ・学習なし。各セルに
  `.complete`マーカーを置き中断から再開できる作りになっている。
  SDXL版なのでモデル読み込み部分はSD1.5用に差し替えが要る。
- コスト感: SDXLで292タイル×4構成×約25秒＝約8時間。SD1.5はより軽い。

**いつ回すか**: ラウンド2スイープ(2026-09-14(月)00:00にsystemd timerで
予約済み)の結果が出て最良点が確定してから、その確定構成で回すのが順序と
しては安い。ラウンド2で最良点が動く可能性があるため。ただし
**ラウンド2の結果もまた5枚に載る**ので、そこで確定と呼ばないこと。

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
  (数日規模のバッチが引数ミスで落ちるのが最も高くつく失敗)
- **長時間GPUジョブは月〜木の4日連続バッチが標準枠**(ユーザーは平日ほぼ
  在室しない)。週末は対話的な作業——レビュー・分析・短い実験・次の
  長時間バッチを決めること——に充てる。数日かかること自体は問題ない。
  チェックポイントは3日目のクラッシュで全損しない頻度で取る
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
