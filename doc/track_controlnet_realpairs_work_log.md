# 作業ログ: ControlNet 実ペアLoRAファインチューン ハルシネーション究明

このtrackの時系列作業ログ。2026-09-06に`inbox/initial_notice.md`から
分離した(それまでは引き継ぎ文書が実質の作業ログを兼ねて900行超まで
肥大化していた)。以降、実験の経過はこのファイルに追記し、
`inbox/initial_notice.md`は「現状・次の一手・運用ルール」だけの
短い引き継ぎ文書として維持する。

2026-08-26以前のより古い経緯は共通基盤側 `../lineart/doc/work_log.md`
の「2026-08-24/25」「2026-08-26」エントリにある(共通基盤側は
2026-08-30が最終更新で、それ以降のこのtrackの実験は記録されていない)。

---

## 検証済み: 推論時 `controlnet_conditioning_scale` 調整(2026-08-26)

学術文献調査(ControlNet++, InnerControl等)を踏まえ、再学習不要な最初の
切り分けとして`infer_controlnet.py --controlnet-conditioning-scale`を
0.50〜2.00で振ってdiag_valid5の5枚を推論し評価。

- 数値: `roundtrip_ssim`は0.25→0.42まで単調増加、`ink_ratio`は6.9倍→4.7倍
  まで低下(いずれもscaleを上げるほど改善方向)。しかし`gt_bsds_f1`は
  0.125〜0.135のレンジでほぼ横ばい(scaleと無相関)。
  詳細: `results/cond_scale_sweep/sweep_metrics.csv`
- 目視: `results/cond_scale_sweep/montage.png`(5サンプル×[rough, GT,
  cs0.50〜2.00]のコンタクトシート)。scaleを上げるとハルシネーションの
  *性質*が変わる — 低scaleでは無秩序なクロスハッチ/スクリブル、
  高scaleでは整った平行線状のハッチングに寄っていく。だが**どちらも
  GTの実際の構造を再現していない**。roundtrip_ssim改善はモデルが
  入力の局所的なエッジ方向にはある程度追従するようになった結果と
  見られるが、生成語彙自体(密ハッチのテクスチャ事前分布)が
  「クリーンな線画」ではなく「ハッチング」に偏っているため、
  GTとの一致度は改善しない。
- **結論**: 推論時のconditioning_scale調整は根本原因の対策にはならない
  (棄却)。文献調査アイデア#2(ControlNet++ cycle-consistency損失)や
  #3(InnerControl中間特徴一貫性)のような、学習時に条件忠実度を
  直接最適化する手法の方が本質的に効きそうという傍証になった。

## 重要な発見: `lineart_anime`前処理によるラフ内容の大幅な欠落(2026-08-26)

idea#1の検証中に、評価スクリプトが**生ラフ**を条件画像として推論している一方
学習は`rough_lineart_anime`(`lineart_anime`プリプロセッサ適用済み)を条件と
していることに気づいた(train/inference不一致)。診断5枚を前処理し直して
再推論すると`gt_bsds_f1`が平均0.129→0.148に改善(5/5で改善)。

しかしさらに調査したところ、より根本的な問題が判明した:
**`lineart_anime`プリプロセッサ(`controlnet_aux.LineartAnimeDetector`)は、
完成・彩色済みのアニメ絵から線画を抽出するために学習されたモデルであり、
「ラフの中から作画意図のある線を選び出す」タスク向けではない**。診断5枚で
生ラフのインク画素数に対する前処理後インク画素数の比率(保持率)を測定した
ところ、42.0%〜55.5%(一律に約半分)しか保持していなかった。線の濃淡に
よらずほぼ一律に半分が失われることから、「たまたま薄い線が消える」という
より、このディテクタ自体がラフ入力に向いていないことを示唆している。

**含意**: 学習時点でモデルは一度も「本物のラフ」を条件として見ておらず、
常に情報量が目減りした版だけを見ている。GTの線画には条件画像にない情報
(間引かれた約半分)が含まれるため、モデルが「条件にない内容を埋める」
ことを学習中に常態的に強いられ、これが密なハッチングによる汎用的な
穴埋めを学習してしまった一因である可能性が高い。rank/epoch/scaleを
振るだけでは解決しない可能性がある。

**方針決定(2026-08-26)**: ベースConditioning戦略(ラフ→条件画像の変換
方法)自体を見直す方向で次を進める。公開`control_v11p_sd15s2_lineart_anime`
を差し替える方向(条件エンコーダごと学習し直す等、より大きな設計変更)は
いったん保留。

## 検証済み: `lineart_coarse`前処理での再学習(2026-08-27完了)

`lineart_anime`前処理の情報欠落問題(上記)を受け、ラフ→条件画像の
変換を`LineartDetector(coarse=True)`("lineart_coarse")に差し替えて
再学習した。他の変数(rank16, lr=1e-4, 10エポック, データ8,467枚)は
`controlnet_lora_realpairs_20260824`と同一に固定(1変数のみ変更)。

- 前処理: `tools/preprocess_lineart_coarse_condition.py`で学習データ
  全体を`data/rough_lineart_coarse/`に生成(8,467/8,467、エラー・
  スキップなし)。データセット全体でのインク密度は中央値11.1%
  (`lineart_anime`版は2.6%、GTターゲットは4.3%)——`lineart_coarse`は
  むしろGTより密になる(当たり線等も含むため)。ほぼ空白のタイルは
  0%(`lineart_anime`版は8.7%)。
- 学習: `experiments/run_controlnet_lora_coarse_20260827.sh`
  (`checkpoints/controlnet_lora_coarse_20260827/final`、10,580ステップ、
  正常完走・エラーなし。所要時間約13.4時間——`lineart_anime`版の
  約9.2時間より遅い。原因未調査、優先度低)。
- 評価(diag5、両モデルとも学習時と同じ前処理形式の条件画像で推論、
  スクリプト: `results/coarse_vs_anime_comparison/eval_compare.py`):
  `gt_bsds_f1`平均が0.1476→0.1538(+4.2%、5サンプル中3で悪化・2で
  改善、ばらつき大)。
- 目視(`results/coarse_vs_anime_comparison/montage.png`): クロスハッチへの
  ハルシネーション自体は**解消されていない**。ハッチングの性質が
  タイトなグリッド状から、やや方向性のあるジェスチャル的なストローク寄りに
  変化した程度で、一部サンプル(008_023, 001_014)ではより単純な
  トーン/シルエットに寄る傾向も見られたが、GT構造への忠実な再現には
  至っていない。
- **結論**: 「ラフの情報を半分捨てている」問題は実在し補正の価値はあった
  (数値・保持率とも改善)が、**根本原因はこれだけでは説明できない**。
  クロスハッチ/ハッチングへの過学習という支配的な失敗モードは、
  条件画像の情報量を増やしただけでは解消しない。文献調査アイデア
  #2(ControlNet++ cycle-consistency)や#3(InnerControl中間特徴一貫性)、
  あるいはLoRA適用層・rank・正則化(prior-preservation)など、学習目的
  関数側に手を入れる方向の検証が必要。

## 進行中: 代替モデル調査からの2実験を無人連結実行中(2026-08-27夜開始)

`lineart_coarse`再学習でも根本原因(クロスハッチ/グレー塗りハルシネーション)
は解消しなかったため、代替モデルを調査(サブエージェント)。候補:
1. SDXL化(Animagine XL 3.1 + SDXL線画ControlNet) — ユーザーが翌日に
   自分で着手する予定、今回は対象外。
2. ControlNetを`lllyasviel/control_v11p_sd15_lineart`(非anime版)に差し替え
   (条件画像は`data/rough_lineart_coarse`をそのまま流用——同じ
   `LineartDetector`系のためペアが自然)。
3. 前処理を`p1atdev/MangaLineExtraction-hf`(マンガ構造線抽出CNN、
   SIGGRAPH2017由来)に差し替え(ControlNet/base_ckptは元のまま)。

**運用上の注意**: MangaLineExtraction-hfのカスタムコードは共有
`../lineart/venv`のtransformers(5.14.1)と非互換(`transformers.onnx`が
削除済み、かつ`PreTrainedModel`内部APIも変更されており動かない)。
共有venvには手を入れず、`/tmp/.../scratchpad/mle_venv`という隔離venv
(torch 2.7.1+cu118, transformers==4.46.0, opencv-python-headless)を
作って前処理専用に使っている——**このtmp venvはセッション/マシン
再起動で消える可能性があるため、再現time is必要になったら同じ手順で
作り直すこと**(このファイルに手順を残す: pip install torch==2.7.1
--index-url https://download.pytorch.org/whl/cu118 transformers==4.46.0
opencv-python-headless、Hugging Face `trust_remote_code=True`でロード)。
出力はwhite-bg/dark-lineの通常配色なので反転して黒背景+白線に変換
済み(`tools/preprocess_manga_line_extraction_condition.py`)。

**実行中**: `experiments/run_overnight_20260827.sh`(nohup&disown、
ユーザーの明示的な承認不要指示のもと無人実行中)。
`bash experiments/run_controlnet_lora_lineartsd15_20260827.sh`→
`bash experiments/preprocess_manga_line_full_20260827.sh`→
`bash experiments/run_controlnet_lora_manga_20260827.sh`の順で、
前段が失敗したら`set -e`で連結が止まる設計。各段ともスモークテスト
済み(手動事前検証も実施)。ログ: `logs/overnight_20260827.log`
(全体)、`logs/controlnet_lora_lineartsd15_20260827.log`、
`logs/preprocess_manga_line_full_20260827.log`、
`logs/controlnet_lora_manga_20260827.log`。完了マーカーはそれぞれ
`logs/*.done`。想定所要時間: 合計18〜28時間程度
(学習1本あたり9〜14時間×2 + 前処理約38分)。

## 完了: 4モデル比較(2026-08-27夜〜08-29未明、無人実行)

前段の「代替モデル調査からの2実験を無人連結実行中」の続き。#2
(`control_v11p_sd15_lineart`)・#3(`manga_line`前処理)ともに学習・評価
完了(エラーなし)。既存2モデル(anime版・coarse版)と合わせて4モデルを
diag5で比較した。

**数値**(`results/four_model_comparison/metrics.csv`、
`results/four_model_comparison/eval_compare.py`で再現可能):

| model | gt_bsds_f1 | precision | recall | area_ink% |
|---|---|---|---|---|
| anime_trained(元祖) | 0.1476 | 0.0874 | 0.5070 | 54.68 |
| coarse_trained | 0.1538 | 0.0954 | 0.4578 | 69.53 |
| **sd15_lineart_trained(#2)** | **0.1558**(最良) | 0.0952 | 0.4733 | 63.17 |
| manga_trained(#3) | 0.1293(**最悪**) | 0.0773(**最悪**) | 0.4249(**最悪**) | 63.43 |

**目視**(`results/four_model_comparison/montage.png`、5サンプル×
[raw rough, GT, anime, coarse, sd15_lineart, manga_line]): **4モデル
全てが、支配的な密ハッチングへのハルシネーションから脱却できていない**。
sd15_lineart版はanime/coarse版と見た目の傾向が近く、大きな質的差は
見えない。manga_line版は一部サンプル(008_023)でハッチングの代わりに
渦巻き状のテクスチャに逃げる等、性質は変わるが根本改善ではない。

**結論**:
- 候補#2(`control_v11p_sd15_lineart`へのControlNet差し替え)は数値上
  4モデル中最良(anime比+5.6% f1、precisionも改善)だが、改善幅は小さく
  目視でも質的なブレークスルーは確認できない。
- 候補#3(`manga_line`前処理への差し替え)は、生ラフの情報保持率が
  最も高かったにもかかわらず、**4モデル中最悪の結果**(f1・precision・
  recallすべて最低)。「前処理段階でラフの情報をより多く残す」という
  介入だけでは、むしろ悪化しうることを示す反例になった。
- 総合すると、この track で試した「ベースモデル・ControlNet世代・
  前処理の差し替え」という4通りの組み合わせはいずれも根本原因(密
  ハッチングへの過学習)を解消しなかった。次に検討する価値が高いのは、
  文献調査(2026-08-26)で見つけた学習目的関数側の介入
  (ControlNet++のcycle-consistency損失、InnerControlの中間特徴一貫性)、
  または候補#1(SDXL化、ユーザーが別途着手予定)。

**保存済み成果物**: `checkpoints/controlnet_lora_lineartsd15_20260827/final`
(5.7MB)、`checkpoints/controlnet_lora_manga_20260827/final`(5.7MB)、
`data/rough_lineart_coarse/`・`data/rough_manga_line/`(各8,467枚、
再学習時の再利用のため保持)。隔離venv
(`/tmp/.../scratchpad/mle_venv`)はセッション終了で消える可能性がある点
に注意(再作成手順は上記の運用上の注意を参照)。

### 追記(2026-08-29): `gt_bsds_f1`だけでは「ハッチ埋め脱却」を測れていなかった

ユーザーからモンタージュの目視で「manga_lineの方がマシに見える」との
指摘。再度見比べると、008_023(渦巻き状の一貫したストローク)・
001_012(三角形の構図に沿ったまとまった線)・001_014(涙滴/うろこ状の
模様)など、manga_line版は他3モデルに比べてハルシネーションの中身が
「雑然とした断片の寄せ集め」ではなく「一貫した流れのある線」に見える
サンプルが複数あった。

これを`tile_region_manifest_480.py`の`orientation_entropy`
(ストローク方向のヒストグラムのエントロピー——少数角度への偏り=硬い
クロスハッチ的、多方向への分散=脱却的、という既存の検証済み指標)で
定量化したところ(`results/four_model_comparison/hatch_score.py`):

| model | orientation_entropy(平均) | 順位 |
|---|---|---|
| **manga_trained** | **0.7728** | **1位(最もハッチ的でない)** |
| anime_trained | 0.7553 | 2位 |
| sd15_lineart_trained | 0.7399 | 3位 |
| coarse_trained | 0.7391 | 4位(最もハッチ的) |
| (参考: GT自体) | 0.7117 | — |

**`gt_bsds_f1`のランキングと完全に逆転している**(`gt_bsds_f1`では
manga_trainedが最下位、sd15_lineart_trainedが最良だった)。

**含意**: `gt_bsds_f1`/precision/recallは「GTとの位置一致度」しか
測っておらず、ハルシネーションの中身が硬いクロスハッチか否かという
このtrackの現在の主目的(ハッチ埋め脱却)には無関係。今後は
**`gt_bsds_f1`(内容の正確さ)と`orientation_entropy`(ハッチ埋め脱却度)
を必ず並記して評価する**方針に変更(ユーザー合意済み、2026-08-29)。
この2軸はトレードオフの関係にある可能性があり、「ハッチ埋め脱却」を
優先するなら`manga_line`前処理系が現時点の最有力候補、「GT一致度」を
優先するなら`sd15_lineart`(候補#2)が最有力、という整理になる。
ただし`orientation_entropy`は「局所的には依然ハッチだが画像内で複数の
異なる角度のハッチ領域が混在している」場合にも高くなり得るため
(=ハッチ脱却の完全な証明ではなく方向性の指標)、目視確認を省略しない
こと。

## 完了: 候補#1 SDXL化 + 5モデル比較(2026-08-29開始〜08-29 23:32完了)

文献調査(2026-08-26)の候補#1。ベース: `cagliostrolab/animagine-xl-3.1`
(アニメ特化SDXL、diffusers形式でHFキャッシュに保存済み
`~/.cache/huggingface/hub/models--cagliostrolab--animagine-xl-3.1/
snapshots/483f0c322568ed13697ed01dd0be07204746d12b`)。ControlNet:
`Eugeoter/noob-sdxl-controlnet-lineart_anime`(diffusers ControlNetModel
形式、fp16 variant、同キャッシュ内)——学習時のベースは`Laxhar/sdxl_noob`
(非diffusers形式の生学習チェックポイント集、追いかける価値なしと判断)
だが、SD1.5の候補#2差し替え時と同様、同系統アニメチェックポイント間の
ControlNet可搬性に賭けている。条件画像は`data/rough_lineart_coarse`を
流用。

**新規スクリプト**: `scripts/train_controlnet_sdxl.py`・
`scripts/infer_controlnet_sdxl.py`(共有リポジトリにSDXL版が存在しない
ため、diffusersの`examples/controlnet/train_controlnet_sdxl.py`を参照
してtrack内に新規作成。2テキストエンコーダ・add_time_ids対応)。

**判明した重要な制約**: SDXLのControlNetはSD1.5版よりはるかに大きい
(fp32で約5GB、SD1.5は約1.4GB)。当初SD1.5と同じ「ControlNetをfp32で
保持」する設計のままだと12GB VRAMでOOMした。**対策**: ControlNetの
凍結ベースをfp16でロードし、LoRA差分パラメータだけ学習後にfp32へ
upcastする(diffusers標準の`cast_training_params`パターン)ことで解消。
さらに、1ステップあたりの所要時間がSD1.5版の約3倍(~14.5s/step)、かつ
VRAM制約でbatch_size=1(SD1.5は2)にせざるを得ずステップ数も2倍になる
ため、**同じ10エポックだと合計約85時間(3.5日)**かかる計算になった。

**方針決定(2026-08-29、ユーザー合意)**: まずは**2エポック
(約17時間、SD1.5系の10エポック学習量とサンプル通過数ベースでほぼ同等)**
で本番投入(`experiments/run_controlnet_lora_sdxl_20260829.sh`、
無人実行中)。この結果が有望なら、**平日の不在期間(2026-09-01月曜以降
を想定)に10エポック本番(約85時間)を別途投入してよい**とユーザーから
了承済み——3.5日かかること自体は、その期間を割り当てれば問題ないとの
判断。

**学習完了**: 07:47開始〜23:32完了(約15.7時間、当初想定17時間より
やや速い。実測ペース約13.03〜13.34s/step、4,232ステップ全て正常完走、
エラーなし)。`checkpoints/controlnet_lora_sdxl_20260829/final`。

**5モデル比較**(anime・coarse・sd15_lineart・manga・sdxlをdiag5で
比較。数値: `results/five_model_comparison/eval_compare.py`・
`hatch_score.py`、目視: `results/five_model_comparison/make_montage.py`
→`montage.png`):

| model | gt_bsds_f1 | precision | recall | area_ink% | orientation_entropy |
|---|---|---|---|---|---|
| anime_trained(元祖) | 0.1476 | 0.0874 | 0.5070 | 54.68 | 0.7553 |
| coarse_trained | 0.1538 | 0.0954 | 0.4578 | 69.53 | 0.7391(最悪) |
| sd15_lineart_trained(#2) | **0.1558**(最良) | 0.0952 | 0.4733 | 63.17 | 0.7399 |
| manga_trained(#3) | 0.1293 | 0.0773 | 0.4249 | 63.43 | **0.7728**(最良) |
| **sdxl_trained(#1・2epoch)** | **0.0945(5モデル中最悪)** | 0.0929 | **0.1284(5モデル中最悪)** | 70.41 | **0.6366(5モデル中最悪)** |
| (参考: GT自体) | — | — | — | — | 0.7117 |

**結論**: SDXL化(候補#1、2エポック)は**両軸とも5モデル中最悪**という
明確なネガティブ結果。目視(`montage.png`)で確認すると、他4モデルに
共通する「密なクロスハッチへのハルシネーション」とは異なる、**別種の
失敗モード**が見える——多くのサンプルでほぼ空白のフラットな灰色領域
+疎らな平行斜線ストロークのみ(lineart_003_018, lineart_001_012等)、
かつ1サンプル(lineart_008_023)では下絵の刀の輪郭とは無関係な
コンテンツ(ゲームUI風のアイコン列)が生成されており、**条件付け
自体がほぼ機能していない**ように見える。`orientation_entropy`が
最低値なのは、伝統的な密ハッチではなく「少数方向への疎らな
平行ストローク」もこの指標では低エントロピーになるため。

**含意・次の検討**: 2エポック(SD1.5系10エポックとサンプル通過数を
揃えたつもり)では明らかに学習不足、または元のSDXL ControlNet
(`Eugeoter/noob-sdxl-controlnet-lineart_anime`、`Laxhar/sdxl_noob`
ベースで学習されたもの)とbase_ckpt(`animagine-xl-3.1`)の組み合わせに
互換性上の問題がある可能性がある(SD1.5候補#2のControlNet差し替えでは
問題なかったが、SDXLはアーキテクチャ差分がより大きい)。ユーザーが
別途検討中の10エポック本番(2026-09-01月曜以降想定)を実施する場合は、
まず**低エポック数でも条件追従が機能しているか(rough構造をある程度
なぞれているか)を先に目視で確認**してから投入判断すべき——今回の
結果だけでは「エポック不足」と「アーキテクチャ非互換」のどちらが
支配的要因か切り分けられていない。

## 完了: 候補#1追試 SDXL化(manga_line差し替え)+ 6モデル比較(2026-08-30開始〜20:53完了)

ユーザーの指摘(2026-08-30)がきっかけ:「animagine-xl 3.1 base x noob
sdxl controlnet lineart anime がこちらの期待するモデルではないように
思える。以前のanime modelとsketchモデルでもマスクにあからさまな差が
あった」。調査の結果、`Eugeoter/noob-sdxl-controlnet-*`シリーズは
`controlnet_aux`前処理器の出力名と完全一致する命名規則
(canny・depth・normal・lineart_anime・softedge_hed・**manga_line**・
lineart_realistic・scribble_hed・scribble_pidinet・tile)——つまり
`lineart_anime`ControlNetは`LineartAnimeDetector`前処理を前提にして
学習されている可能性が高く、前回投入したのは`lineart_coarse`前処理
画像だった、という**条件付けフォーマット不一致**の疑いが濃厚と判断。

**対策(ユーザー選択: manga_lineペアに切り替えて再学習)**:
ControlNetを`Eugeoter/noob-sdxl-controlnet-manga_line`に、条件画像を
`data/rough_manga_line`(SD1.5候補#3で使用済み、`orientation_entropy`
軸で最良だったペア)に差し替え。base_ckpt(`animagine-xl-3.1`)・
LoRA rank・lr等は前回と完全に同一に固定し、ControlNet初期化+条件付け
フォーマットの1変数のみを分離検証(`experiments/
run_controlnet_lora_sdxl_manga_20260830.sh`)。

**学習完了**: 4,232ステップ全て正常完走、エラーなし(既知の無害な
fp16フォールバック警告のみ)。`checkpoints/
controlnet_lora_sdxl_manga_20260830/final`。

**6モデル比較**(anime・coarse・sd15_lineart・manga・sdxl・sdxl_mangaを
diag5で比較。数値: `results/six_model_comparison/eval_compare.py`・
`hatch_score.py`→`metrics.csv`、目視: `results/six_model_comparison/
make_montage.py`→`montage.png`):

| model | gt_bsds_f1 | precision | recall | area_ink% | orientation_entropy |
|---|---|---|---|---|---|
| anime_trained(元祖) | 0.1476 | 0.0874 | 0.5070 | 54.68 | 0.7553 |
| coarse_trained | 0.1538 | 0.0954 | 0.4578 | 69.53 | 0.7391(最悪) |
| sd15_lineart_trained(#2) | **0.1558**(最良) | 0.0952 | 0.4733 | 63.17 | 0.7399 |
| manga_trained(#3) | 0.1293 | 0.0773 | 0.4249 | 63.43 | **0.7728**(最良) |
| sdxl_trained(#1・lineart_anime版) | 0.0945(6モデル中最悪) | 0.0929 | 0.1284(6モデル中最悪) | 70.41 | 0.6366(6モデル中最悪) |
| **sdxl_manga_trained(#1追試・manga_line版)** | 0.1226 | **0.0978**(最良) | 0.2009 | 42.47 | 0.6982 |
| (参考: GT自体) | — | — | — | — | 0.7117 |

**結論**: 条件付けフォーマットをname-matchedなmanga_lineに揃えたところ、
`sdxl_trained`比で両軸とも明確に改善(`gt_bsds_f1` 0.0945→0.1226、
`orientation_entropy` 0.6366→0.6982、GT値0.7117にかなり接近)。
目視(`montage.png`)でも、前回見られた「ほぼ空白のフラット領域」
「下絵と無関係なコンテンツ生成(ゲームUI風アイコン)」という
条件付けが機能していない失敗モードは**解消**——manga_line版は
下絵の構図をある程度認識して反応している。

一方で、**新しい失敗モードが出現**: 密なクロスハッチでも空白でもなく、
太い黒塗り/白抜けの**幾何学的ブロック状パターン**(ステンドグラス風・
鋭角な多角形分割)が全サンプルで支配的に生成されている
(`lineart_003_018`, `lineart_008_014`など顕著)。GTの繊細な線画とも、
SD1.5系4モデル共通の「クロスハッチ密集」とも異なる、SDXL固有の
第3の失敗モード。`orientation_entropy`が改善したのは、この多角形
エッジが様々な角度を持つため(疎な平行ストロークより角度多様性が
高い)であり、必ずしも「良い線画に近づいた」ことを意味しない——
数値だけでは見抜けなかった点、目視併用の方針が改めて有効だった。

**含意**: SDXL化(2エポック)は、ControlNet条件付けフォーマットの
一致によって「条件無視」問題は改善できたが、依然として6モデル中
`gt_bsds_f1`は下から2番目・`orientation_entropy`もGTに最も近いのは
manga_trained(SD1.5系)のまま。SDXL移行は現時点でSD1.5系を上回る
根拠がなく、**10エポック本番(約85時間)の投入は保留が妥当**——
2エポックでも「太い幾何学ブロック」という別の望ましくない挙動が
支配的であり、エポック数を伸ばすだけでこれが解消する保証がない。
むしろ`animagine-xl-3.1`ベース自体がクリーンな線画生成に不向きな
可能性、あるいはLoRA rank16がSDXLの表現力に対して不十分な可能性を
含め、SD1.5系での未検証仮説(下記)を先に潰す方が投資対効果が高いと
判断。

## 検証済み・棄却: 仮説1(`clip_pairs`タイル側のコンテンツ構成偏り、2026-09-02)

`measure_lineart_profile.py`で`data/line`内の`koma_ref`(既存5プール
合算、ako5ver2/fitness/gakuen/hamlabi/housei、1489枚——過去のwork_log
記録`combined_koma_20260729`と同数で一致確認)と`clip_pairs`
(`clippairskoma`プレフィックス、6978枚、学習データの82%を占める主力)
を分割・比較。GPU不要、CPU処理のみで約7分。

| metric(中央値) | koma_ref | clip_pairs | 乖離 | 参考: 過去の実破綻例(psd_line拡張) |
|---|---:|---:|---:|---:|
| blank_cell_fraction | 0.688 | 0.609 | -11% | -91%(0.688→0.063) |
| grid_ink_cv | 1.880 | 1.682 | -10% | -64%(1.878→0.670) |
| background_ratio | 0.945 | 0.957 | +1% | -42%(0.945→0.548) |
| long_component_ratio | 0.814 | 0.846 | +4%(改善) | -45%(0.819→0.447) |
| components_per_1k_ink_px | 10.85 | 13.08 | +21% | +108%(10.91→22.64) |
| deep_black_ratio | 0.0020 | 0.0292 | 14.6倍 | (未計測) |
| line_width_p50 | 3.82px | 2.74px | -28%(細い) | 変化なし |

生データ: `results/data_bias_check_20260902/lineart_profile_koma_ref_vs_clip_pairs.csv`

**判定**: 「テクスチャ崩壊」を示す複合構造軸(blank_cell_fraction・
grid_ink_cv・background_ratio・long_component_ratio・
components_per_1k_ink_px)は、2026-08-21の実際の破綻ケース
(psd_lineプール拡張、これらが-91%/-64%/-42%/-45%/+108%動いた)と
比べて全て軽微な乖離にとどまる。目視スポットチェック
(6枚ずつのコンタクトシート)でもclip_pairsは細く綺麗な線画で、
密集テクスチャの兆候なし。二値化強度(deep_black_ratio)と線幅は
有意に違うが、方向は「clip_pairsの方が細く硬い線」であり、これが
ハッチ密集ハルシネーションを誘発する説明にはならない。**仮説1は
棄却** — データ側の構図偏りがクロスハッチの主因である可能性は低い。

## 検証済み・棄却: 仮説2(LoRA rank16の表現力不足、2026-09-02開始〜09-03 06:09完了)

`manga_trained`(候補#3、rank16)を基準に、base_ckpt/ControlNet初期化/
`data/rough_manga_line`条件/エポック数(10)/lrを完全固定し、LoRA rankの
みrank16→rank32に変更して再学習(`experiments/
run_controlnet_lora_manga_rank32_20260902.sh`、10,580ステップ、
約10.25時間、正常完走・エラーなし)。`lora_alpha == rank`
(`train_controlnet.py`)のため実効LoRAスケールは両rankで約1x、追加の
スケール補正は不要。

**7モデル比較**(既存6モデル+`manga_rank32_trained`をdiag5で比較。
数値: `results/seven_model_comparison/eval_compare.py`・
`hatch_score.py`→`metrics.csv`、目視: `results/seven_model_comparison/
make_montage.py`→`montage.png`):

| model | gt_bsds_f1 | precision | recall | area_ink% | orientation_entropy |
|---|---|---|---|---|---|
| anime_trained(元祖) | 0.1476 | 0.0874 | 0.5070 | 54.68 | 0.7553 |
| coarse_trained | 0.1538 | 0.0954 | 0.4578 | 69.53 | 0.7391 |
| sd15_lineart_trained(#2) | **0.1558**(最良) | 0.0952 | 0.4733 | 63.17 | 0.7399 |
| manga_trained(#3・rank16) | 0.1293 | 0.0773 | 0.4249 | 63.43 | **0.7728**(最良) |
| **manga_rank32_trained(#3・rank32)** | 0.1411 | 0.0816 | **0.6039**(最良) | 61.16 | 0.7697(僅差2位) |
| sdxl_trained(#1・lineart_anime版) | 0.0945(最悪) | 0.0929 | 0.1284(最悪) | 70.41 | 0.6366(最悪) |
| sdxl_manga_trained(#1追試・manga_line版) | 0.1226 | 0.0978 | 0.2009 | 42.47 | 0.6982 |
| (参考: GT自体) | — | — | — | — | 0.7117 |

**結論**: rank16→32でrank16版比 `gt_bsds_f1` +9.1%(0.1293→0.1411)、
recall +42%(0.4249→0.6039)と位置一致度は明確に改善した一方、
このtrackの主目的である`orientation_entropy`は**ほぼ横ばい**
(0.7728→0.7697、誤差の範囲でむしろ僅かに悪化)。目視
(`montage.png`)でも密なクロスハッチ自体は解消されておらず、
サンプルによってはrank16版より格子状ハッチがより密・均一に見える
(`lineart_008_023`、`lineart_001_012`)。**仮説2も棄却** —
容量(rank)不足はクロスハッチハルシネーションの主因ではなく、
むしろ「クロスハッチをより精密に配置する」方向に効いてしまっている
可能性がある。

## 検証済み・棄却: 仮説3(エポック数、2026-09-03開始〜22:19完了)

`manga_trained`(rank16、10エポック)を基準に、rank/lr/データ/条件付けを
完全固定してエポック数のみ10→2に変更(`experiments/
run_controlnet_lora_manga_epoch2_20260903.sh`、2,116ステップ、約1.85
時間、正常完走・エラーなし)。診断ログ(rank16/rank32いずれの学習でも
diffusion lossがstep約250=1エポック未満で既にノイジーな収束水準に
達していた)から、「学習不足」より「早期にハッチへ収束済み」の疑いを
検証する目的。

**8モデル比較**(既存7モデル+`manga_epoch2_trained`をdiag5で比較。
数値: `results/eight_model_comparison/eval_compare.py`・
`hatch_score.py`→`metrics.csv`、目視: `results/eight_model_comparison/
make_montage.py`→`montage.png`):

| model | gt_bsds_f1 | precision | recall | area_ink% | orientation_entropy |
|---|---|---|---|---|---|
| manga_trained(rank16・10ep) | 0.1293 | 0.0773 | 0.4249 | 63.43 | 0.7728 |
| manga_rank32_trained(rank32・10ep) | 0.1411 | 0.0816 | 0.6039 | 61.16 | 0.7697 |
| **manga_epoch2_trained(rank16・2ep)** | 0.1297(ほぼ同一) | 0.0789 | 0.3875 | 48.59 | **0.7756**(僅差最良) |
| (参考: GT自体) | — | — | — | — | 0.7117 |

**結論**: `gt_bsds_f1`・`orientation_entropy`とも10エポック版と
誤差の範囲でほぼ同一(f1 0.1293→0.1297、entropy 0.7728→0.7756)。
目視(`montage.png`のmanga_ep2列)でも、密なハッチ/スクリブルへの
ハルシネーションという支配的な失敗モードは10エポック版と質的に
変わらない——`lineart_008_014`ではやや塊状の濃淡に寄るが、GTの
繊細な線画構造には至っていない。**仮説3も棄却** — 診断ログの
示唆通り、モデルは1エポック未満で既に定常的な(ハッチに逃げる)挙動に
収束しており、エポック数を10→2に減らしても増やしても
(仮説2のrank32検証時と同じ10エポックで確認済み)結果は変わらない。

## 追加確認: GT訓練データ全体(`data/line`, 8467枚)にクロスハッチ/中間グレーは
   ほぼ皆無(2026-09-04)

ユーザーからの指摘: モンタージュを見ると生成物には10〜30%程度の中間グレーが
多く、より濃い塗りは現れない——これは実際のマンガ原稿が濃いグレーを避ける
傾向と一致する。また網トーン・ハッチングはマンガ表現として実在の技法であり、
学習データ側にそもそも含まれている可能性がある。2026-09-02の仮説1検証は
`koma_ref`と`clip_pairs`という**2つのプール間の相対比較**であり、GT全体の
**絶対統計**(密な塗り・網トーンがそもそも存在するか)は未確認だったため、
`data/line`全8467枚に対して直接集計した
(`results/gt_corpus_hatch_check_20260904/measure_full_corpus.py`)。

| 指標 | 結果 |
|---|---|
| タイル全体ink_ratio(<128、8467枚) | 平均3.95%・中央値3.66%・**最大値9.95%**(全タイル中) |
| ink_ratio > 30%のタイル | **0枚 / 8467枚** |
| ink_ratio > 50%のタイル | **0枚 / 8467枚** |
| 局所8x8グリッドセル(60x60px、541,888セル) | 84.1%が[0,10%)、96.2%が[0,20%)未満、**50%超は0.10%**、**70%超は0.014%**、最大セル95.78% |
| deep_black_ratio(<30、純黒、タイル全体) | 中央値2.49%・最大9.72% |

局所ink密度が最も高かった上位12セル(コーパス全体から)を目視スポット
チェック(`results/gt_corpus_hatch_check_20260904/top_dense_cells_spotcheck.png`)
——**全て黒髪・黒い服・影・黒猫などの単純な黒塗りオブジェクトで、
網トーン/クロスハッチパターンは1件も確認できなかった**。10〜30%の
平坦な中間グレー(screentone相当)に該当する例もゼロ。

**結論**: このtrackの`data/line`は網トーン/ハッチングレイヤーを含まない
「線のみ」抽出データセットである可能性が高い。モデル出力に支配的な密な
クロスハッチ・中間グレーは、GTに存在する技法の忠実な再現では**なく**、
学習データにほぼ存在しない領域への純粋なハルシネーションと確認された。
2026-09-02の仮説1棄却(プール間相対比較)を、より強い絶対的根拠で補強する
結果——「データ側の技法の偏りが原因」という説明は事実上完全に排除できる。

生データ: `results/gt_corpus_hatch_check_20260904/per_tile.csv`

## 検証済み・棄却: 仮説4(x0-GT Sobelエッジ一致度損失、2026-09-04
   10:03開始〜20:59:43完了)

`manga_trained`(rank16、10エポック)を基準に、rank/lr/データ/条件付けを
完全固定し、学習目的関数のみ変更——予測x0(VAEデコード後)とGT画像との
Sobelエッジ一致度L1損失を補助項として追加(`consistency_weight=0.1`、
`consistency_max_timestep=200`、低timestepのみ適用)。10,580ステップ、
約10.88時間で正常完走・エラーなし。外部`manga_line`前処理器経由の
roundtrip(venv非互換のため断念)の代替として、GTペアがある強みを
活かした設計(`scripts/train_controlnet_consistency.py`)。

**9モデル比較**(既存8モデル+`manga_consistency_trained`をdiag5で比較。
数値: `results/nine_model_comparison/eval_compare.py`・
`hatch_score.py`→`metrics.csv`、目視: `results/nine_model_comparison/
make_montage.py`→`montage.png`):

| model | gt_bsds_f1 | precision | recall | area_ink% | orientation_entropy |
|---|---|---|---|---|---|
| manga_trained(rank16・10ep・損失介入なし) | 0.1293 | 0.0773 | 0.4249 | 63.43 | 0.7728 |
| manga_epoch2_trained(rank16・2ep) | 0.1297 | 0.0789 | 0.3875 | 48.59 | 0.7756 |
| **manga_consistency_trained(rank16・10ep・Sobel一致度損失)** | 0.1411(改善) | 0.0858 | 0.4306 | 57.41 | **0.7758**(改善なし) |
| (参考: GT自体) | — | — | — | — | 0.7117 |

**結論**: `gt_bsds_f1`はmanga_trained比で改善(0.1293→0.1411、
manga_rank32_trainedと同水準)したが、これは主にprecision/recallの
バランス変化であり、肝心の`orientation_entropy`は0.7728→0.7758と
**誤差の範囲で不変**(GTの0.7117とのギャップは全く縮まらず、むしろ
9モデル中で最もGTから遠い側に位置する)。目視(`montage.png`の
manga_cons列)でも、5サンプル全てで他の学習済みモデルと質的に
区別できない密なクロスハッチ/スクリブルへのハルシネーションが
支配的で、Sobel一致度損失は低timestep(<200)のx0推定に対してのみ
作用するため、線の「配置」を多少押し戻す効果はあってもハッチへの
定常収束という支配的な失敗モードそのものは抑制できていない。
**仮説4も棄却** — データ(仮説1)・LoRA容量(仮説2)・エポック数
(仮説3)に続き、単純な補助損失項の追加という軽量な目的関数側介入でも
再現できず、4仮説すべてが棄却された。

## 未検証の仮説(次に試すべき候補・2026-09-04時点)

1. ~~`clip_pairs`タイル側のコンテンツ構成偏り~~ → 棄却済み(2026-09-02)
2. ~~LoRA rank16の表現力不足~~ → 棄却済み(2026-09-03、rank32でも
   `orientation_entropy`は横ばい)
3. ~~8,467枚×10エポックでも実は学習不足の可能性~~ → 棄却済み
   (2026-09-03、2エポックでも10エポックと誤差の範囲で同一)
4. ~~学習目的関数側の介入(x0-GT Sobelエッジ一致度損失)~~ → 棄却済み
   (2026-09-04、`orientation_entropy`は誤差の範囲で不変)

**4仮説すべて棄却**(データ・容量・エポック数・軽量な補助損失、いずれも
`orientation_entropy`をGT側に動かせなかった)。加えて2026-09-04の
GT全コーパス絶対統計確認により「データ側に実在する技法の再現」という
説明も排除済み。次に検討すべき方向性(未着手、ユーザーとの相談が必要):

- Sobel一致度損失の重み・timestep範囲を大きく変える(現状
  `consistency_weight=0.1`・`consistency_max_timestep=200`は
  1点しか試していない——過小の可能性)よりも、根本的に異なるアプローチ
  (例: 推論時サンプリング側の介入、CFGスケール調整、ControlNetの
  conditioning_scale再検証との組み合わせ、あるいはベースSD1.5モデル
  自体が持つ「密な陰影表現」への強いプライアを疑う)を検討する価値が
  ある。
- 4仮説すべてが「学習側の1変数」に閉じていた点に注意——ベースモデルの
  事前分布(SD1.5がアニメ塗り靴/陰影表現に強く偏っている可能性)や
  推論時の設定(現状の`controlnet_conditioning_scale`・CFG値)を疑う
  仮説はまだ検証されていない。

**文献調査(2026-09-05)**: ControlNet++(arxiv 2404.07987)は
denoisingロスのみでは条件忠実度を直接罰していないと定式化、
InnerControl/"Heeding the Inner Voice"(arxiv 2507.02321、
github.com/ControlGenAI/InnerControl)はControlNet++の一致度損失が
**最終ステップのみ**にしか適用されない限界を指摘し全timestepでの
中間特徴一致度への拡張を提案——実装済みの仮説4損失
(`consistency_max_timestep=200`、低timestepのみ)はこの限界を
そのまま持つ簡易版だった可能性。加えて、`data/captions.csv`全8467件
(学習時)・全推論スクリプトの`--caption`引数のいずれも
`"...manga panel..."`を固定で含んでおり、SD1.5事前学習データ中で
「manga」概念自体がハッチ/トーンと強く結びついている可能性
(コミュニティ記事: lilting.ch)を踏まえ、base modelの事前分布が
LoRAの補正より優先されているのではという仮説が浮上。

**簡易確認(2026-09-05・交絡あり)**: `manga_trained`の既存チェック
ポイントに対し、**推論キャプションのみ**"manga panel"を除去して
再生成(`results/caption_ablation_20260905/no_manga_word/`、
比較元`results/controlnet_lora_manga_20260827_eval/`)。結果:
`orientation_entropy`は0.7728→0.7925と**むしろ悪化**、目視でも
明確な改善なし。ただし学習自体は"manga panel"入りキャプションで
行われているため、これは推論時のみを変えた交絡ありの簡易テストに
過ぎず、「学習データ側からmanga語を除いて最初から再学習する」という
仮説5本来の公平な検証にはなっていない——**結論保留**、本格検証には
`captions.csv`のcaption列から"manga panel"を除去した上での再学習
(数時間規模)が必要。

## 検証済み・棄却: 仮説5("manga panel"キャプション除去、2026-09-05
   01:12開始〜10:20:44完了)

上記の交絡ありテストを受け、`data/captions.csv`全8467件のcaption列から
一律の末尾"`, manga panel`"を除去した`data/captions_no_manga_word.csv`
を作成(全件が完全に同一パターンで終わることを事前確認済み)、
学習・推論キャプションの両方から一貫して"manga panel"を排除して
`manga_trained`と同一条件(rank16・10エポック・lr1e-4・同一
base_ckpt/ControlNet-init/前処理)で最初から再学習
(`experiments/run_controlnet_lora_manga_nomangaword_20260905.sh`、
10,580ステップ、約9.15時間、正常完走・エラーなし)。

**10モデル比較**(既存9モデル+`manga_nomangaword_trained`をdiag5で
比較。数値: `results/ten_model_comparison/eval_compare.py`・
`hatch_score.py`→`metrics.csv`、目視: `results/ten_model_comparison/
make_montage.py`→`montage.png`):

| model | gt_bsds_f1 | precision | recall | area_ink% | orientation_entropy |
|---|---|---|---|---|---|
| manga_trained("manga panel"あり) | 0.1293 | 0.0773 | 0.4249 | 63.43 | 0.7728 |
| manga_consistency_trained | 0.1411 | 0.0858 | 0.4306 | 57.41 | 0.7758 |
| **manga_nomangaword_trained("manga panel"完全除去)** | 0.1417(最良水準) | 0.0839 | 0.4707 | 55.18 | **0.7860**(**10モデル中最悪**) |
| (参考: GT自体) | — | — | — | — | 0.7117 |

**結論**: `gt_bsds_f1`は10モデル中で最良水準(0.1417)まで改善したが、
`orientation_entropy`はGTから最も遠い0.7860に**悪化**——「manga」語を
キャプションから排除するほどクロスハッチが強まるという、当初の仮説とは
**逆方向**の結果になった。目視(`montage.png`のmanga_noman列)でも、
`lineart_008_023`では他モデルより渦を巻くような複雑なハッチパターンが
むしろ増加している。**仮説5も棄却** — 「manga」という単語が
base modelのハッチ/トーン事前分布を呼び出しているという説明は
支持されず、むしろキャプションの具体性を下げたことで、テキスト条件が
弱まりControlNetの空間条件への依存度が(意図とは逆に)下がった、
または別の交絡が生じた可能性がある。これで**5仮説すべてが棄却**され、
「学習パイプライン側の1変数を1つずつ変える」というアプローチ自体を
見直す必要がある段階に達した。

## 検証済み・棄却: 網羅的キャプション語アブレーション(2026-09-05
   10:41開始〜11:13:48完了、推論のみ・再学習なし)

「manga panel」除去が逆効果だった結果を受け、"comic"以外の語も含めて
`data/captions.csv`のtags列で頻度0.5%以上(8467件中42件以上)の
**63タグ全て**を個別に調査(`results/word_ablation_20260905/
candidate_tags.py`)。`manga_nomangaword_trained`チェックポイントを
固定し、各タグをベースキャプション("monochrome line art, clean
linework, black and white")に前置してdiag5を生成(再学習なし、
パイプライン1回ロードで63語×5枚を約32分、
`results/word_ablation_20260905/generate_sweep.py`)。

**結果**(`results/word_ablation_20260905/scores.csv`、目視:
`results/word_ablation_20260905/montage_extremes.png`):

| | mean orientation_entropy | baseline差分 |
|---|---|---|
| GT参照 | 0.7117 | — |
| baseline(語追加なし) | 0.7860 | — |
| 最もGTに近づいた語: `blood` | 0.7570 | **-0.0291** |
| `long_hair` | 0.7628 | -0.0232 |
| `multiple_girls` | 0.7634 | -0.0226 |
| `black_border` | 0.7637 | -0.0224 |
| ちなみに`comic` | 0.7884 | +0.0023(ほぼ無風) |
| ちなみに`silent_comic` | 0.7832 | -0.0029(ほぼ無風) |
| ちなみに`lineart` | 0.7863 | +0.0003(ほぼ無風) |
| 最も悪化させた語: `text_focus` | 0.7958 | +0.0098 |

**結論**: 63語全てのdelta幅は-0.029〜+0.010(baseline
0.7860を中心に±3%程度)に収まり、GTとbaselineの間の絶対的な
ギャップ(0.7860-0.7117=**0.0743**)と比べると**どの単語も
1桁小さい**——単語1つの寄与ではこのギャップは到底説明できない。
当初怪しいと目星をつけた"comic"(+0.0023)・"silent_comic"
(-0.0029)はいずれもほぼゼロ影響。目視(`montage_extremes.png`)でも、
最大改善語`blood`ですら密なクロスハッチはそのまま残っており(唯一
`text_focus`だけは意味的に当然の副作用として文字風の落書きが
出現しただけ)、質的な違いは見られない。**「特定のキャプション語が
base modelのハッチ事前分布を呼び出している」という仮説の系統は
全体として棄却** — キャプション語彙(仮説5とその発展形)は
原因ではなく、原因はテキスト条件付け以外の場所(ControlNetの空間
条件付けとbase modelの拡散過程そのものの綱引き、あるいは
アーキテクチャ/サンプリング側)にあると考えるべき段階に達した。

**訂正・方法論上の重要な補足(2026-09-05)**: ユーザーが
`montage_extremes.png`の目視で「text_focus(entropy最悪+0.0098)は
むしろクロスハッチが少なく見える、グレートーンへの逃避では」と指摘。
検証のため`measure_lineart_profile.py`の追加軸を計算
(`results/word_ablation_20260905/hatch_vs_blob_profile.py`)。

| source | orientation_entropy | line_width_p50 | ink_ratio |
|---|---|---|---|
| GT参照 | 0.7117 | **3.72**(細) | 0.035 |
| baseline | 0.7860 | 3.99 | 0.333 |
| text_focus(entropy最悪) | 0.7958 | **13.25**(3.3倍太) | 0.342 |
| blood(entropy最良と誤報) | 0.7570 | **11.82**(3倍太) | 0.407 |

`line_width_p50`(ストローク幅中央値、GT基準で細いほど良い)を見ると、
指摘通り**text_focusは実際には一枚の滑らかな塗りへの逃避**であり、
`orientation_entropy`はその滑らかな曲線境界を「多方向=ハッチ的」と
誤認して過大評価していた。さらに重要な点として、**前述のスイープで
「最良」と報告した`blood`も同じ症状(line_width_p50=11.82)**——
ハッチが減ったのではなく、たまたま単純な形状の塗りに逃げて
角度分布が偏っただけで、text_focusと同じ「塗り逃避」失敗モードだった
可能性が高い。**教訓**: `orientation_entropy`は「ハッチ網目」と
「滑らかな塗りの境界」を区別できない——以降はこれ単体でなく
`line_width_p50`(太ければ塗り逃避)・`ink_ratio`(GTの0.035近辺からの
乖離)を併記して判断すること。なお、いずれの語もink_ratioはGTの
10倍前後(0.32〜0.41)であり、この訂正は前述の「キャプション語彙は
原因ではない」という結論そのものは変えない。

## ★ブレイクスルー: `controlnet_conditioning_scale`が主因だった
   (推論のみ施策スイープ、2026-09-05)

仮説1〜5がすべて棄却され、キャプション語彙の系統も棄却された後、
`train_controlnet.py`を読み直して判明した構造的事実が突破口になった:

```python
unet.requires_grad_(False)           # ベースUNetは全実験を通じて凍結
controlnet.add_adapter(lora_config)  # LoRAはControlNetにしか付かない
```

**このtrackの全実験でベースUNet(実際にデノイズする本体)は一度も
学習されていない**。ならばUNet側のハッチ事前分布は学習で潰せない——
代わりに**空間条件(ControlNet)側の影響力を事前分布に打ち勝つ強さまで
上げればよい**、という発想で推論のみのスイープを実施
(`results/inference_only_sweep_20260905/`、再学習なし・計25バリアント・
約20分。ホストは`controlnet_lora_manga_consistency_20260904/final`)。

**結果**(`scores.csv`、目視: `montage.png`・`montage_cs_ladder.png`):

| variant | orientation_entropy | line_width_p50 | ink_ratio | components/1k | **gt_bsds_f1** |
|---|---|---|---|---|---|
| GT参照 | 0.7117 | 3.72 | 0.0353 | 21.01 | — |
| **baseline (cs1.0, 全実験の既定値)** | 0.7758 | 3.94 | 0.2992 | 13.39 | 0.1411 |
| cs2.0 | 0.7982 | 4.32 | 0.1258 | 22.27 | 0.2166 |
| **cs2.5** | 0.7952 | 3.55 | 0.1056 | 33.36 | **0.2336** |
| cs2.5_cfg5.0 | 0.7923 | 4.48 | 0.1019 | 34.62 | **0.2343**(最良) |
| **cs3.5** | 0.7744 | 3.34 | 0.0772 | 57.08 | **0.2337** |
| cs4.0 | 0.7752 | 3.17 | 0.0693 | 70.42 | 0.2266 |
| cs5.0(破綻) | 0.7853 | 3.94 | 0.0634 | 82.06 | 0.2165 |

**`gt_bsds_f1`が0.1411→0.2337(+66%)**——これまで学習した全10モデルの
範囲(0.0945〜0.1558)を大きく超える、このtrack最高値。`ink_ratio`は
0.2992→0.0772とGT(0.0353)方向に3.9倍改善し、`line_width_p50`は3.34と
GTより細い(=塗り逃避ではない)。目視(`montage_cs_ladder.png`)でも、
cs1.0で背景を埋め尽くしていたクロスハッチが**cs2.0〜2.5で消え、背景が
白に戻り線が clean になる**という質的転換が明確に見える。

**上限**: cs4.0以降は線自体が薄れて消えはじめ(cs5.0では剣がほぼ消失)、
`components_per_1k_ink_px`が82まで暴走(GT 21)=線の分断。F1も
cs2.5〜3.5でピークを打って以降低下。**最適域は cs2.5〜3.5**。

**他の軸**: CFG単独(5.0〜10.0)もink_ratioを下げるが`line_width_p50`が
5.4〜6.3と太くなる(塗り寄り)ため単独では劣る。ネガティブプロンプトは
**逆効果**——特に`neg_hatching`("hatching, cross-hatching"を除外)は
line_width_p50=14.85とベタ塗りへ逃げ、ink_ratioも0.4027と最悪。
「ハッチを禁止すると代わりに黒く塗りつぶす」という挙動。新しい
positiveプロンプト(coloring book等)も単独では効果薄。

**重要な含意**: このtrackは**全10モデルの評価を一貫してcs=1.0
(infer_controlnet.pyの既定値)で行ってきた**。仮説1〜5の比較結果は
すべて「ハッチで潰れた条件下での比較」だった可能性があり、
モデル間の実力差を過小評価していた恐れがある。今後の評価は
cs2.5〜3.5でも併せて行うべき。

## 準備完了・月曜起動待ち: 仮説6(UNet側LoRA併用、2026-09-05実装済み)

上記の「ベースUNetが全実験で凍結されていた」という発見の、もう一方の
帰結。推論側(conditioning_scale)は上のブレイクスルーで対処できたが、
**学習側でUNetのテキスト条件応答そのものを動かす**手はまだ未検証。
`scripts/train_controlnet_unet_lora.py`を実装(共通基盤の
`train_controlnet.py` + `train_domain_lora.py`のUNet LoRAパターンを合流。
ControlNet LoRAとUNet LoRAを同一の対ペアデータで同時学習)。

**検証済み(2026-09-05)**:
- `--unet-lora-rank 0`でプレーン版と**全6ステップのlossが完全一致**
  (0.0092/0.0271/0.0378/0.0564/0.0190/0.0501)——A/B土台として健全
- `--unet-lora-rank 16`でスモークテスト通過。ControlNet 1,462,272 +
  UNet 3,188,736 の2アダプタが学習・保存される。OOMなし
  (12GB GPUに収まる。UNetは学習時fp32必須のため`accelerator.prepare`
  経由に分岐——fp16のままだと勾配unscaleで壊れる)
- 保存したUNet LoRAが`infer_controlnet.py`の**既存**`--lora-dir`で
  そのままロードでき、ControlNet LoRAとの併用推論が通ることを確認
  (推論スクリプトの変更は不要)
- 実測 **5.2s/step**(プレーン版3.1s/stepの約1.65倍)→ 10エポック
  =10,580ステップで**約15.3時間**

**月曜(2026-09-07)日中に起動**:
`experiments/run_controlnet_lora_manga_unetlora_20260907.sh`
(10エポック・rank16/16・lr1e-4、他はmanga_trainedと同一の単一変数分離)。
エポック数は`ten_model_comparison`との厳密な比較可能性を優先して10を
維持(ユーザー判断)——朝9時開始で深夜0時頃完了、結果確認は火曜。
評価は上のブレイクスルーを反映し、**cs=1.0(過去との比較用)と
cs=3.0(実用最適域)の両方**で自動実行される。


## 完了: 仮説6(UNet側LoRA併用)= 失敗、および全11モデルのcs再評価
   (2026-09-05 14:05開始 → 09-06 06:07完了)

GPUが空いていたため月曜を待たず09-05に前倒しで起動。学習
(`controlnet_lora_manga_unetlora_20260905`、10エポック=10,580ステップ、
5.12s/step、約15.1時間)は09-06 05:09に正常完走、続けて連結ジョブが
全11モデル × cs{1.0, 2.0, 2.5, 3.0, 3.5} × diag5 = 275枚を生成して
06:07に完了(`experiments/run_cs_reeval_20260906.sh`、
結果 `results/cs_reeval_20260906/`)。

**アンカー検証**: cs=1.0列の`gt_bsds_f1`が既存
`results/ten_model_comparison/metrics.csv`と**全10モデルで完全一致
(max |delta| = 0.0000)**。モデル→チェックポイント/条件画像/キャプションの
対応付けが正しいことが確認できたので、以降の数値は信頼できる。
(過去と同じ`infer_controlnet.py`/`infer_controlnet_sdxl.py`をそのまま
呼ぶ実装にした狙い通り。)

| model | cs1.0(過去の全比較) | 最良 | ink_ratio | line_w_p50 | orient_ent |
|---|---|---|---|---|---|
| **manga_consistency**(仮説4) | 0.1411 | **0.2337**(cs3.5) | **0.0772** | **3.34** | 0.7744 |
| manga_nomangaword(仮説5) | 0.1417 | 0.2263(cs2.5) | 0.1309 | 4.10 | 0.7992 |
| manga_trained | 0.1293 | 0.2195(cs3.0) | 0.1191 | 3.94 | 0.7961 |
| anime_trained | 0.1476 | 0.2160(cs3.5) | 0.0702 | 3.55 | 0.7650 |
| manga_epoch2(仮説3) | 0.1297 | 0.2146(cs3.5) | 0.1878 | 5.25 | 0.7806 |
| coarse_trained | 0.1538 | 0.2101(cs3.5)※ | 0.5490 | **40.92** | 0.7781 |
| sd15_lineart | **0.1558**(旧1位) | 0.2099(cs3.5) | 0.0982 | 4.32 | 0.8016 |
| sdxl_manga | 0.1226 | 0.1522(cs2.0) | 0.1347 | 6.01 | 0.7692 |
| **manga_unetlora**(仮説6) | 0.1203 | 0.1479(cs2.0) | 0.3310 | 4.53 | 0.7961 |
| **manga_rank32**(仮説2) | 0.1411 | 0.1411(**cs1.0**) | 0.3906 | 3.34 | 0.7697 |
| sdxl_trained | 0.0945 | 0.0945(**cs1.0**)※ | 0.2491 | 20.64 | 0.6366 |
| (GT参照) | — | — | 0.0353 | 3.72 | 0.7117 |

※ `coarse_trained`と`sdxl_trained`のf1は**信用してはいけない**——
`line_width_p50`がGT(3.72)の11倍・5.5倍で、これは線ではなくベタ塗りへの
逃避。2026-09-05に追加した判定ルール(entropy単体で判断せず
line_width_p50/ink_ratioを併記する)が実際に機能した事例。

### 結論

1. **仮説4(x0-GT Sobel一致度損失)の棄却を撤回する**。cs=1.0では
   全モデルが0.13〜0.15に団子で埋もれていたが、最適域では**全11モデル中
   1位**(f1 0.2337)、かつ`ink_ratio` 0.0772・`line_width_p50` 3.34と
   GT(0.0353 / 3.72)に最も近い。直接のベースライン`manga_trained`
   (0.2195 / 0.1191 / 3.94)を3軸すべてで上回る。当時の棄却根拠は
   「`orientation_entropy`が誤差の範囲で不変」だったが、その指標自体が
   後日(2026-09-05)ハッチ網目と塗りの境界を区別できないと判明した
   ものだった。**consistency損失は効いている**(効果は小さいが一貫)。
2. **仮説6(UNet側LoRA)は失敗**。ベースの`manga_trained`より全csで劣り
   (cs1.0で0.1203 < 0.1293、最良でも0.1479 < 0.2195)、`ink_ratio`も
   0.3310と悪い。さらに他のmanga系と異なり**csを上げると単調に悪化**
   ——UNet側を可学習にしたことでcsレバー自体を壊している。目視
   (`montage_best.png`のmanga_unetlora列)でも砂状のノイズが出る。
   凍結UNetという制約は、少なくとも「LoRAを足す」形では外すべきでない。
3. **仮説2(rank32)の棄却はより強く確定**。manga系で唯一cs上昇に対し
   単調悪化(0.1411→0.1232)し、最適域では最下位級。LoRA容量を
   増やすと、条件付けを強めても追随できなくなる。
4. **csレバーは前処理・アーキテクチャを跨いで一般化する**(11モデル中
   8モデルで改善、SD1.5系はすべて改善)。ただし最適csはモデル依存:
   manga系は2.5〜3.5、rank32とsdxl_trainedは1.0(上げると悪化)、
   unetloraとsdxl_mangaは2.0。
5. **順位が大きく入れ替わった**。cs1.0では`sd15_lineart`(0.1558)が1位で
   `manga_trained`(0.1293)は下位だったが、最適域では
   `manga_consistency`が1位、`sd15_lineart`は7位に後退。**過去の
   仮説1〜5の比較はすべてハルシネーション支配領域での比較だった**
   という懸念は、少なくとも仮説4については現実のものだった。

目視: `results/cs_reeval_20260906/montage_best.png`(各モデル最良csでの
全5サンプル)、`montage_by_scale.png`(モデル×スケール)。
最良の`manga_consistency` cs3.5でも、GTの白背景・黒線に対して
**背景がグレー・線もグレー寄り**という差は残っている(次の課題)。

## ★このtrackの目的達成 — クロスハッチ脱却を完了とし、2つの後継trackへ分割
   (2026-09-06、ユーザー判断)

`montage_best.png`をユーザーが目視し、**4列目`manga_consistency` cs3.5が
最良**であることを確認。**このtrackの課題である「クロスハッチからの脱却」は
達成したと判断**し、ここで一区切りとする。

**到達点**: `checkpoints/controlnet_lora_manga_consistency_20260904/final`
を `--controlnet-conditioning-scale 3.5` で推論する構成。
gt_bsds_f1 0.1411(開始時点の既定cs=1.0)→ **0.2337**、
ink_ratio 0.2992 → **0.0772**(GT 0.0353)、
line_width_p50 3.94 → **3.34**(GT 3.72)。目視で背景のクロスハッチが消え、
白背景にクリーンな線が出る。

**主因の総括**: 原因は学習側(データ・容量・エポック・損失・キャプション)
ではなく、**推論時の`controlnet_conditioning_scale`が既定値1.0のまま
だったこと**だった。cs=1.0はハルシネーションが支配的な領域で、そこでの
比較では全モデルがf1 0.13〜0.15に潰れ、モデル間の差もconsistency損失の
効果も見えなくなっていた。仮説1〜6を順に潰していく過程で
`train_controlnet.py`を読み直し「ベースUNetが全実験を通じて凍結されて
いる=UNet側のハッチ事前分布は学習で潰せない」と気づいたことが、
「ならば空間条件側を事前分布に打ち勝つ強さまで上げればよい」という
発想につながって解決に至った。

**後続の2課題は別trackに分割する**(ユーザー判断、2026-09-06):

1. **SD1.5路線**: 上記の最良構成からさらに詰める。残差はGTの白背景・黒線に
   対して**背景がグレー・線もグレー寄り**な点。cs4.0以上に上げると線自体が
   薄れて消える(cs5.0で剣がほぼ消失)ためcsだけでは埋まらない。
2. **SDXL路線**: SDXLは**クロスハッチを出さない**
   (`sdxl_trained`の`orientation_entropy` 0.6366は全11モデル中最小で、
   GTの0.7117よりさらに低い=方向が揃っている)が、**下絵との乖離が激しい**
   (f1 0.0945で最下位、かつSD1.5系と違いcsを上げると悪化する=csレバーが
   効かない)。クロスハッチとは**別種の問題**として切り離して扱う。

起案は共通基盤側 `../lineart/doc/track_proposal_20260906.md` に置いた。
