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

## ★Track Aの問いは終わった (2026-09-13)

Track Aの問いは「最良構成でもグレー背景になる問題をconsistency損失側の
チューニングで埋められるか」だった。**答えは、Track Bと同じ形で出た——
diffusionが前処理器(`manga_line`)を上回っていない。**

`consistency_weight`スイープ(round1: 0.02/0.05/0.1/0.2/0.5、round2:
0.15/0.25/0.3/0.4)は2026-09-13までに完了。診断5枚では新最良点
`w=0.2, cs2.5`(gt_bsds_f1 0.2354)が前処理器単体(0.2566)に-0.021で
負けていた。**5枚では楽観的すぎる数字だった**——Track Bの292タイル
プロトコルを流用し(housei群は除外、`holdout_lineart_family.txt`
192枚のみ)本trackでも再検証した結果:

| | gt_bsds_f1(192枚) | near_white_frac |
|---|---:|---:|
| 前処理器`manga_line`単体 | **0.2847** | **0.931**(GT 0.924とほぼ同水準) |
| w=0.2, cs2.5 | 0.2514(**-0.0333**) | 0.837 |
| w=0.4, cs2.5 | 0.2524(**-0.0323**) | 0.817 |

**決定的な再解釈**: 前処理器自体が既にGT水準の紙の白さ(near_white 0.931
≈ GT 0.924)を持っていた。**「グレー残差」はモデルが埋めるべき欠落では
なく、拡散モデルの生成過程が前処理器の白さを0.93→0.82まで劣化させて
いた**、というのが正確な構図。consistency損失のweightを上げるほど
near_white_fracが上がる(0.4→0.8台)のは、劣化を軽減しているだけで、
前処理器が最初から持っていた水準を超えたことは一度もない。

Track Bが「SDXLは条件画像を0.88でコピーしているだけ」と結論したのに対し、
Track Aは「条件画像から離れながらGTに近づく」機構的に異なる挙動を
示した点は本物の発見だが(`vs_condition_f1`が weight とともに下がりつつ
f1が上がる)、**それでもなお前処理器を超えられていない**という結果は
変わらない。

**このtrackは終了する。** 後継の投資先は既存の**Track C
(`../lineart-stroke-selection`、branch `stroke-selection`)**——同じ
「前処理器を超えられない」問いに対する答え(判別的ストローク選択、
削るだけで0.32→0.74)を出している。新規trackは起こさず、Track Cへの
申し送りを`../lineart/inbox/`経由で送る(下記)。

### 旧・次の一手(参考、この理由により凍結)

1. ~~第2段consistency_weightスイープ~~ → 実施済み(round2)、上記の
   通り前処理器超えには至らず
2. ~~InnerControl方式への拡張~~ → 保留。生成側の改善余地を掘る前に、
   Track Cの「選択」路線がすでに天井0.74超の実績を持つ以上、優先度は低い
3. ~~出力の二値性/コントラストへの介入~~ → 同上、保留

### 共通基盤からの申し送り(2026-09-10、経緯として保持): 5枚問題は放置しないこと

本trackの`note_sd15_consistency_weight_result_20260910.md`は「より広い
タイル集合での再検証は今のところ予定していない」としていたが、**その後
実施した**(上記参照)。Track Bは同じ問題を自覚して既に対処を用意していた。

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

**ユーザー判断(2026-09-13): houseiは含めない。** Track Aの学習データ
(`data/train_list.txt` 8,467件)自体が型A系統(clippairskoma、
`doc/pool_inventory.md`の分類)であり、housei(型C: ベタ・描き文字・背景)は
学習にも診断5枚にも一度も混ざっていない。292タイル検証をやる際も
**`holdout_lineart_family.txt`(192枚)のみを先行して回す**こと。houseiは
切り離し、型C(ベタ塗り)を主題にする別trackの課題として扱う
(Track B終了報告の「群B(ベタ塗り)を目標にするかの判断」参照、未決)。

**いつ回すか**: consistency_weightスイープはround1(0.02/0.05/0.1/0.2/0.5)・
round2(0.15/0.25/0.3/0.4)とも完了済み(2026-09-13、予定を前倒しして実施)。
新最良点は`w=0.4, cs=2.5`(`near_white_frac` 0.813)。ただし
**前処理器`manga_line`単体(0.2566)にround1・round2の全27セルが依然
負けている**ことが共通基盤の汚染チェックで判明したため、192タイル検証は
この確定構成で回すのが順序としては安いが、**それでもなお5枚→192枚への
一般化を確認する位置づけ**であり、192枚の結果が出てもそれ自体を新たな
「確定」と呼ばないこと(次のハイパラ変更が来れば再び載せ替わる)。

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
