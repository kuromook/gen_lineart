# アーキテクチャ選択の記録

更新: 2026-08-05 JST

このファイルは「**どのモデルアーキテクチャを何のために試したか、その数式・コードは何か、そして実際に目視で確認したか**(数値指標だけで判断していないか)」を一目で確認するための場所です。`doc/work_log.md`はセッションごとの経緯を追う物語的な記録、このファイルは「今何が存在し、何が証明されたか」を項目ごとに整理したものです。

**このファイルを作った理由:** 数値指標は改善しているのに、montageで目視すると、そのアーキテクチャが本来解決しようとしていた目的に全く適っていなかった、というケースが何度か起きました(各項目の「目視評価」を参照 — 例: adversarial loss を外した実験は数値上悪化したが、montageレビューで理由が判明した。GAN有無の差は実は「変わらない曖昧な模様に対する濃さ・自信度の再調整」に過ぎず、数値が示唆していたような構造的な違いではなかった)。**どの項目についても、その他メモ・目視評価の記述を読まずに数値だけを信用しないこと。**

**ブランチについて:** 下記のCNN+GAN系(「cleanup」/atari系列)は`cleanup-refiner`ブランチ上にあります(`lineart/model_zoo.py`, `scripts/train_i2i_survey.py`)。Direction 4(diffusion)関連は`diffusion-controlnet`ブランチ(ControlNetによるrough→line条件付き変換、2026-08-04時点で一旦保留 — 10倍長時間ランでも幻覚が解消せず)と、そこから分岐した新しい`diffusion`ブランチ(2026-08-04〜、変換タスクは一旦棚上げし、rough/line各ドメイン単体の生成品質そのものを見る方向)の2つに分かれています — アーキテクチャ的にCNN+GAN系とは無関係なので意図的に分けています。このファイルは横断的なまとめなので関連ブランチ間で同期させておくべきです。もし内容がズレていたら、片方だけを信用せず手動でマージしてください。

**データについて:** 特に断りがない限り、「koma」とついている項目はすべて`combined_koma_20260729`(5ソース計1489タイル、現在のクリーン/整合済みデータ)で学習・評価しており、比較には8タイルの`eval_clean_lineart004_8.txt`(クリーンeval set)を使っています。このデータより前の項目(BCEベースライン、ResNet-GAN atari、Direction 7の線幅損失スイープ)は古い世代のデータを使っています(各項目に記載)。

---

## 評価指標について(簡易説明)

以下で使われている指標の意味です。詳細な実装は`tools/evaluation/evaluate_fixed_outputs.py`と`tools/evaluation/evaluate_halo_outputs.py`を参照。いずれも予測画像を閾値128でink/背景に二値化してから計算します。

### 基本指標(`evaluate_fixed_outputs.py`)

| 指標 | 意味 |
|---|---|
| `precision_2px` | 予測したinkピクセルのうち、2px以内にGTのinkピクセルがある割合。「余計な線を引いていないか」 |
| `recall_2px` | GTのinkピクセルのうち、2px以内に予測inkピクセルがある割合。「本来の線をどれだけ拾えているか」 |
| `F1@2px` | precision_2pxとrecall_2pxの調和平均。総合的な線の一致度(2pxのズレは許容)。**この文書内で最も頻出する指標** |
| `chamfer_px` | 予測→最寄りGT、GT→最寄り予測、双方向の平均距離(px、20pxで打ち切り)。小さいほど空間的に近い |
| `pred_ink` / `target_ink` | 予測/GTそれぞれの画面に占めるinkピクセルの割合(密度) |
| `ink_ratio` | `pred_ink / target_ink`。1.0がGTと同密度、1より大きいとinkの出しすぎ(over-ink)、小さいとinkの出し足りなさ(under-ink) |

### halo(にじみ・曖昧さ)指標(`evaluate_halo_outputs.py`)

GTの線から「core(線そのもの、距離0-2px)」「halo_band(線のすぐ外側、2-9px)」「far_bg(線から離れた背景、9px超)」の3領域に分け、各領域での予測inkの強さを見ます。「faint(かすれ)」は予測ink値が0.03〜0.35の中途半端な範囲にあるピクセル(=はっきり線でも背景でもない曖昧な状態)。

| 指標 | 意味 |
|---|---|
| `core_ink_mean` | GTの線の真上での予測ink強度の平均。**高いほど「本物の線をどれだけ自信を持って濃く描けているか」** |
| `halo_band_ink_mean` | 線のすぐ外側(2-9px)での予測ink強度の平均。理想的にはほぼ0(そこにinkがあってはいけない) |
| `halo_band_faint_ratio` | 線のすぐ外側のうち「かすれ」状態にあるピクセルの割合。**高いほど線の輪郭がぼやけている(marbled/soft) — 今夜の一連の検証で最も注目した指標** |
| `far_bg_ink_mean` / `far_bg_faint_ratio` | 線から離れた背景での同様の指標。高いと関係ない場所にまで薄いinkが滲んでいる |
| `halo_to_core` | `halo_band_ink_mean / core_ink_mean`。1に近いほど「線の周りのにじみが線本体とほぼ同じ濃さ」=輪郭が不明瞭。小さいほど線とそれ以外がくっきり分かれている |

### 線画らしさプロファイル指標 -- GT/ペア不要(`measure_lineart_profile.py`)

`diffusion`ブランチで導入(2026-08-05)。上記2つはいずれもGT線画とのペアが前提だが、domain-only LoRA(rough/lineを無条件・無ペアで生成)にはGTペアが存在しないため、**単一画像だけから計算できる**「線画らしさ」の多軸プロファイルとして新規に作成した。実装は`tools/evaluation/measure_lineart_profile.py`(モジュールdocstringに設計意図と失敗事例を詳述)。想定用途: 実線画タイル(リファレンス分布)と生成サンプルの両方に同じ指標をかけ、軸ごとにどこがリファレンス分布から外れているかを見る — 単一スコアに潰さないのは、このプロジェクトでこれまで単一指標最適化が別の軸での崩壊を隠してきた反省(本ファイル冒頭「このファイルを作った理由」と同じ教訓)による。

いずれも閾値128(このプロジェクト全体で使うink/背景の標準閾値)を基準に計算する。**1点、設計時に閾値校正のミスがあり修正済み**: 初版は「confident-black」をink非依存の`<30`で独自定義し、それ以外(30-220)を一律"midtone"と呼んでいたが、実データ調査で実線画のink画素(`<128`)自体の中央値が73であることが判明 -- 大半の本物のinkが「30という恣意的な閾値」のせいで誤って"midtone"側にカウントされ、`faint_of_drawn_ratio`が実線画で異常に高く(0.9台)出るというバグを生んでいた。修正後はinkの標準閾値(128)を唯一のアンカーとして使う。

| 指標 | 意味 |
|---|---|
| `ink_ratio` | 全画素に占めるink(`<128`)画素の割合 |
| `deep_black_ratio` | 全画素に占める`<30`(ほぼ純黒)画素の割合。inkの中でもどれだけ濃いかの参考診断(locality計算のアンカーには使わない) |
| `background_ratio` | 全画素に占める`>220`(確信ある白)画素の割合 |
| `faint_of_drawn_ratio` | 「描かれた領域」(ink+faint)のうちfaint(128-220、曖昧なグレー)が占める割合 |
| `faint_near_ink_ratio` | faint画素のうち、最寄りのink画素から3px以内にあるものの割合。**高いほど「曖昧さが本物のストロークに薄く貼り付いたアンチエイリアス縁」、低いほど「inkから浮いて拡散した曖昧さ」= soft/marbled失敗モードの兆候** |
| `faint_mean_dist_to_ink` | faint画素から最寄りink画素までの平均距離(px) |
| `long_component_ratio` / `components_per_1k_ink_px` | skeleton化後の連結成分長分布。`evaluate_stroke_stability.py`と同一実装、ストローク連続性(2026-08-01の「ぐらつき」調査で導入) |
| `line_width_p50` / `width_consistency`(p95/p50) | 距離変換ベースの線幅と、その画像内でのばらつき。ばらつきが大きいほど太さが不均一(blobby) |
| `long_line_ratio` / `orientation_entropy` | `tile_region_manifest_480.py`から流用。長い直線の比率とエッジ方向のエントロピー |

2026-08-05時点の初回結果(koma_ref参照1489タイル vs. domain LoRA line生成17枚、中央値): ストローク連続性はほぼ同等(`long_component_ratio` 0.81 vs 0.78)。線幅はLoRAが明確に太い(3.82px vs 5.73px)。最も差が出たのは`faint_near_ink_ratio`(参照0.90 vs LoRA 0.66) -- LoRAのfaint画素はinkから浮いて拡散している割合が本物より高く、これが次に注視すべき軸。CSV: `results/lineart_profile_koma_ref_vs_domain_lora_line_20260805.csv`。詳細: `doc/work_log.md`(2026-08-05のエントリ)。

---

## 土台となるモデル(`cleanup`/atari系列全体の入力)

### BCEベースライン(下限値の基準)

- **目的**: 素のBCEだけでどこまで到達できるか、というリファレンス
- **アーキテクチャ**: 単純なU-Net、rough(1ch)→line(1ch)の直接回帰、aux/hintなし
- **数式**: `L = BCEWithLogits(pred, target)`
- **実装のコード**: `lineart/unetgenerator.py::UNetGenerator`、`scripts/train.py`/`train_i2i_survey.py`の`--model unet`
- **目視評価**: 済み — 以降のあらゆる方向性が脱却しようとした「ソフトで何も言い切らない」テクスチャの基準として確立
- **montageの場所**: `results/compare_shape1_clean_split_bce_milddup800_ft10_lr1e5*.png`(`doc/model_results_summary.md`参照)
- **その他メモ**: F1@2px 0.301、chamfer 7.08。目標値ではなく下限値。

### ResNet-GAN atari生成器

- **目的**: roughだけから「構造のヒント」(atari)を生成し、以降の全cleanup/refinerモデルの条件付けに使う
- **アーキテクチャ**: ResNetのエンコーダ-残差-デコーダ(`ResnetGenerator`)+ PatchGAN adversarial loss
- **数式**: `L_G = L_recon + λ_adv·L_adv(D(rough,pred))`; `ResnetGenerator`: 7×7 head → stride-2 down ×2 → `ResnetBlock`(conv-IN-ReLU-conv-IN+skip)×6 → transpose-conv up ×2 → 7×7 tail
- **実装のコード**: `lineart/model_zoo.py::ResnetGenerator`、`PatchDiscriminator`；チェックポイント`checkpoints/model_resnet_binft_e3_resnet_gan_advsharp_binft/best.pth`
- **目視評価**: 済み、以降の比較で繰り返し確認
- **montageの場所**: (以下の全cleanup/refinerモデルの`--aux-dir`入力として使われており、koma系montageで単体評価はしていない)
- **その他メモ**: **一族全体のsoft/marbled天井の根本原因(2026-08-01判明)**。この生成器自体の生出力が既にハーフトーン状にソフト(ピクセル直接確認で検証済み — `doc/work_log.md`の2026-08-01のエントリ参照)。以下の全モデルはこの出力周りの小さな補正しか行わないため、このソフトさをそのまま引き継ぐ。

---

## Direction 5: 浅い残差クリーンアップ (`--model cleanup` / `cleanupdark`)

Direction 1(multi-scale PatchGAN)・2(feature matching)・3(structure/edge loss)は単独では検証されず、loss項として束ねて使われている — 詳細は末尾の「Direction 1/2/3/7: 単独検証なし」を参照。

### cleanup + MSGAN(**採用中/本番候補**)

- **目的**: ゼロから塗り直すのではなくatariヒントを「補正」する方向にネットワークを誘導し、BCEベースラインのソフトなテクスチャから脱却する
- **アーキテクチャ**: `ResidualCleanupGenerator`: 2ch入力(rough+aux) → conv 1層 + (conv-IN-ReLU)ブロック×5 → conv 1層 → `tanh`で範囲を絞った補正を、aux入力自体のlogitに加算
- **数式**: `out = logit(1 − aux) + tanh(net(rough,aux))·4.0`；loss(`combined_koma_lucy_mild_msgan_20260729`): `L = 0.75·BCE(pos_w=5) + 0.03·L1 + 0.08·tolerant_f1 + 0.14·ink_loss + 0.10·binary_conf + 0.04·structure_pyramid + 0.03·adv(multiscale) + 0.08·feature_match`
- **実装のコード**: `lineart/model_zoo.py::ResidualCleanupGenerator`；`scripts/train_i2i_survey.py`；`experiments/run_combined_koma_lucy_mild_msgan_20260729.sh`
- **目視評価**: 済み — 何夜にもわたりmontageレビュー済み、現時点で採用している「この一族の中では一番マシ」な候補
- **montageの場所**: `results/compare_combined_koma_lucy_mild_msgan_20260729.png`
- **その他メモ**: F1@2px 0.4175、chamfer 4.680、ink_ratio 1.557。**どのレビューでも依然としてsoft/marbled** — 「採用」は「この一族の中で最良」という意味であり「解決した」わけではない。

### cleanupdark(一方向補正)

- **目的**: 双方向補正ではなく一方向(inkを足すだけ)の補正にすることで、部分消去/部分追加を混ぜたグレーをどこにでも塗れてしまうソフトな双方向tanhブレンディングを避けられるか検証
- **アーキテクチャ**: `DarkenOnlyCleanupGenerator`: 同じ骨格だが`sigmoid`で範囲を絞り、ゼロ/負バイアス初期化でaux baselineに近い状態から開始し、inkを足すことしかできない(消せない)
- **数式**: `out = logit(1 − aux) + sigmoid(net(rough,aux))·1.5`
- **実装のコード**: `lineart/model_zoo.py::DarkenOnlyCleanupGenerator`、`--model cleanupdark`；`experiments/run_combined_koma_cleanupdark_20260730.sh`(loss recipeはmsganと同じ)
- **目視評価**: 済み
- **montageの場所**: `results/compare_combined_koma_cleanupdark_20260730.png`
- **その他メモ**: F1@2px 0.409(msganの0.4175より悪い)、chamferも悪化。**不採用** — 一方向制約は正味の改善にならなかった。

### no-adversarial-loss ablation(loss設計仮説の検証)

- **目的**: adversarial/feature-match/structure lossを外した場合(アーキテクチャは固定)に、よりシャープでmarbledでない出力が得られるか — つまりmarblingは*loss設計*由来のアーティファクトなのかを検証
- **アーキテクチャ**: 上のmsganと同じ`ResidualCleanupGenerator`アーキテクチャ、**GANを一切使わない**
- **数式**: `out`の式はmsganと同一；loss: `L = 0.8·BCE(pos_w=3) + 0.2·L1 + 0.5·edge_loss(Canny)`、adversarial/feature-match/shape/ink/binary/structureの各項なし
- **実装のコード**: `experiments/run_combined_koma_lucy_mild_noadv_20260801.sh`；今回のセッションで`scripts/train_i2i_survey.py`に`--edge-weight`フラグを追加(以前から存在したが未使用だった`lineart/losses.py::edge_loss`を接続)
- **目視評価**: **済み — このドキュメントを作る直接のきっかけになった事例。** 数値上は悪化(F1 0.20 vs 0.42、深刻なunder-ink)し、*かつ*見た目も依然としてsoft/marbled — ただ薄くなっただけで、loss設計仮説が予測したようなシャープな結果にはならなかった。追加で画素相関を確認したところ(このoutputとmsgan outputの相関0.94〜0.95)、GAN有無は「どこにinkを置くか」をほとんど変えておらず、変化しない曖昧な空間パターンに対する全体的な濃さ・自信度の再調整に過ぎないことが判明
- **montageの場所**: `results/compare_combined_koma_lucy_mild_noadv_20260801.png`
- **その他メモ**: **仮説は否定された。** 根本原因は代わりに(上の「ResNet-GAN atari生成器」項目にある)atari生成器自体のソフトな出力が、このアーキテクチャの小さな`tanh`補正上限(`max_delta=4.0`)を通して伝播していることに求められる — この構造上、lossを何に変えてもソフトなアンカーから大きくは逸脱できない。`doc/work_log.md`「2026-08-01: No-Adversarial-Loss Ablation」参照。

---

## Direction 6: 自信度/太さ二頭出力 (`--model dualhead`)

- **目的**: 「ここに線があるか」(自信度/スケルトン)と「どれだけ太い/濃いか」(ink)を分離し、中間グレーの曖昧さに直接対処する — 1つの出力テンソルに両方を背負わせているのがボトルネックかもしれないという仮説
- **アーキテクチャ**: `DualHeadRefinerGenerator`: 共有トランク(cleanupと同じconv構成)→2つのhead: `ink_head`(cleanupと同じ範囲を絞った補正)+ `skeleton_head`(自信度)、`final = aux_logits + ink_delta + gain·sigmoid(skeleton_logits)`として合成
- **数式**: v1(バグあり): `final = ink_head(trunk(x))`、**auxへのアンカーが一切ない**。v2(修正済み): `final = logit(1−aux) + tanh(ink_head(feat))·4.0 + 1.5·sigmoid(skeleton_head(feat))`；lossに`+ skeleton_weight·BCE(skeleton_logits, GT_skeleton)`を追加
- **実装のコード**: `lineart/model_zoo.py::DualHeadRefinerGenerator`；`experiments/run_combined_koma_dualhead_20260731.sh`(v1)。v2は専用の実験スクリプトなし、`.done`マーカーなしのその場限りの再実行で、(2026-08-01まで)保存されたmontageもなかった — `results/combined_koma_dualhead_v2_20260731/`配下のタイルごとの出力PNGだけが残っていた
- **目視評価**: 両方とも済み、ただし**v2のmontage/metricsは、このドキュメントを書くまでディスク上に実際には存在していなかった** — `doc/work_log.md`/`doc/model_directions.md`の数値自体は正しかったが、それを描画したものが今までなかった。2026-08-01に、残っていたタイルごとのPNGから再生成: `tools/compare/make_multi_model_eval_compare.py` + `tools/evaluation/evaluate_fixed_outputs.py`で、F1@2px 0.4096 / ink_ratio 2.609が過去の記述と正確に一致することを確認
- **montageの場所**: v1: `results/compare_combined_koma_dualhead_20260731.png`。v2(2026-08-01に再生成): `results/compare_combined_koma_dualhead_v2_20260731.png`、metrics: `results/fixed_output_metrics_combined_koma_dualhead_v2_20260731_compare.csv`
- **その他メモ**: v1: アンカーなしバグ → 慢性的にunder-ink、F1@2px 0.198。v2(アンカーあり): F1@2px 0.4096、一族中最良のchamfer 4.530だが、ink_ratio 2.609(over-ink、montageでmsgan/v1より明らかに濃い)、1サンプルで目立つ黒い塊のアーティファクトあり。**不採用** — 同じsoft/marbled一族のまま、質的な飛躍なし。**この項目自体が、このドキュメントが防ごうとしている問題の実例**: 本物の数値結果が何ヶ月も目視可能な成果物なしで放置されていた。

---

## Direction 8: HED風マルチスケールside output (`--model hed`)

- **目的**: HED(エッジ検出)の「デコーダの中間スケールを直接教師あり学習する」手法を借用し、マルチスケール教師信号が長い線・短い線の両方に役立つか検証
- **アーキテクチャ**: `HedUNetGenerator`: フルの`UNetGenerator` 2chエンコーダ/デコーダ + デコーダの1/4・1/2解像度特徴を読む1×1convの「side output」head 2つ、それぞれダウンサンプルしたGTで教師あり；最終出力はcleanupと同じパターンでauxを中心とした範囲を絞った補正
- **数式**: `final = logit(1−aux) + tanh(out_conv(d1))·4.0`；side loss: `L += side_weight · mean(BCE(side3, downsample(GT)), BCE(side2, downsample(GT)))`；side headは最終出力そのものには影響せず、共有デコーダへの勾配だけを形作る
- **実装のコード**: `lineart/model_zoo.py::HedUNetGenerator`；`experiments/run_combined_koma_hed_20260731.sh`(v1、バグあり)、`hed_v2`(3エポックのまま再実行)、`hed_v3`(10エポック)と再試行
- **目視評価**: 3回とも済み
- **montageの場所**: v3(最終、最も情報量が多い): `results/compare_combined_koma_hed_v3_20260731.png`
- **その他メモ**: v1: dualhead v1と同じアンカーなしバグ → F1@2px 0.190。v2(アンカーあり、3エポック): F1@2px 0.235、まだ大きく不足。v3(アンカーあり、10エポック — このプロジェクトの標準的なfrom-scratch-unet予算): F1@2px 0.328、まだ採用中の~0.41天井には届かず、別の天井ではなくその同じ天井に*収束していく*途中と判断。**不採用**、これ以上エポックを重ねる価値なし。

---

## Direction 9: ボトルネック自己注意機構 (`--model attn`)

- **目的**: 遠く離れたピクセル間をO(1)ホップで結ぶ経路をrefinerに与える(例: 顔の片側の髪の房と反対側のストロークを整合させる)。U-Netのローカルなconv受容野だけに頼らずに済むようにする
- **アーキテクチャ**: `AttentionUNetGenerator`: フルの`UNetGenerator` 2chエンコーダ/デコーダ + 512チャンネルのボトルネック(480px入力に対し60×60)に1つの`SelfAttention2d`ブロック；`gamma`をゼロ初期化して最初は何もしないようにする；出力はcleanupと同じパターンでauxを中心とした範囲を絞った補正
- **数式**: `attn(x) = x + γ·softmax(QKᵀ/√d)·V`(γは0から開始)；`final = logit(1−aux) + tanh(out_conv(d1))·4.0`、loss recipeは上のmsganと同じ(GANあり)
- **実装のコード**: `lineart/model_zoo.py::SelfAttention2d`, `AttentionUNetGenerator`；`experiments/run_combined_koma_attn_20260731.sh`
- **目視評価**: 済み
- **montageの場所**: `results/compare_combined_koma_attn_20260731.png`
- **その他メモ**: 最初から残差アンカーの教訓を適用して構築(v1のバグなし)。F1@2px 0.348、一族中最もバランスの良いink_ratio(1.016)だが、まだ~0.41の天井には届かず、montageでも**attention特有の長距離一貫性のメリットは見られなかった**。不採用。**これでDirection 5/6/8/9のサーベイが終了** — 全て同じ~0.40-0.42のF1@2px soft/marbled天井以下に収束。

---

## Direction 4: Diffusion / ControlNet (`diffusion-controlnet`ブランチ、別系統)

- **目的**: CNN+GAN一族(Direction 5/6/8/9)が全て同じ天井に収束したあと、「線画がどういうものか」を実際に知っている全く別の生成パラダイムを試す
- **アーキテクチャ**: SD1.5系UNet(`AOM3A1B_orangemixs.safetensors`、凍結)+ 学習済みControlNetアダプタ(約3.61億パラメータ、roughタイルで条件付け)；標準的なlatent diffusionのnoise-prediction学習
- **数式**: `ε̂ = UNet(z_t, t, τ, down/mid_residuals)`、`down/mid_residuals = ControlNet(z_t, t, τ, rough)`；`L = MSE(ε̂, ε)`；全タイル共通の固定caption `τ`(後にタイルごとのWD14自動タグに置き換え済みだが、まだ実学習には未使用)
- **実装のコード**: `scripts/train_controlnet.py`、`scripts/infer_controlnet.py`、`scripts/tag_wd14.py`(いずれも`diffusion-controlnet`ブランチにのみ存在)
- **目視評価**: 済み
- **montageの場所**: `results/compare_controlnet_koma_direction4_20260731.png`、スイープ版`results/compare_controlnet_koma_direction4_20260731_sweep.png`、10倍長時間ラン版`results/compare_controlnet_koma_direction4_longrun_20260803.png`
- **その他メモ**: **プロジェクト全体で初めてsoft/marbled天井を視覚的に突破した結果** — わずか10エポック(1860ステップ)で、くっきりとした自信のある完全二値のアニメ風inkを達成。ただし入力ラフの具体的な内容とは緩くしか対応しない幻覚が発生する(一部タイルで表情/ポーズが違う)ため、見た目は良くてもF1@2pxはCNN+GANベースラインより悪化(0.20 vs 0.42)。conditioning-scale/guidance-scaleのスイープでも幻覚は解消せず(スイープ版montage参照) — 学習不足(ControlNetの既知の「sudden convergence phenomenon」)と診断、推論設定で直せる問題ではない。**2026-08-03 00:00 JSTにcrontabで10倍(18,600ステップ)の長時間ランを実行、18:50完了**。**結果: 学習不足仮説は否定された** — F1@2pxは0.202→0.216、ink_ratioは6.36→5.93と、10倍の学習量でもほぼ動かず(`results/fixed_output_metrics_controlnet_koma_direction4_longrun_20260803_compare.csv`)。ユーザーがmontageを目視確認し、幻覚(roughの具体的な内容と無関係な、もっともらしい別内容を自信満々に生成する挙動)は解消していないと判断。sudden convergence phenomenonでは説明がつかず、ステップ数を伸ばす方向はこれ以上の投資に値しないと判断し、この軸は保留。次に試すなら学習量ではなく別の変数(データ量・conditioning方式・per-tile caption等)が候補。

---

## 単段直接回帰(`notebooks/gen_lineart.ipynb`のオリジナルアーキテクチャを現在のクリーンデータで再現)

- **目的**: これまでの全ての項目で手を付けていなかった唯一のアーキテクチャ変数を検証: Direction 5/6/8/9(およびno-adversarial-lossアブレーション)は全てatari+範囲補正の2段構成を固定していた。今回の一連の調査のきっかけとなった、leak修正以前の時代のnotebookモデルは、atariを介さない**単段**の直接rough→line回帰で、より見た目がシャープだったとされている
- **アーキテクチャ**: `UNetGenerator`、aux/atari入力なし(`in_channels=1`) — notebookのモデルとアーキテクチャ的に同一(64→128→256→512チャンネル、`ResBlock`+dilated conv、concat skip connection)
- **数式**: `pred_logits = UNet(rough)`(残差アンカーなし)；lossはnoadvと同一: `L = 0.8·BCE(pos_w=3) + 0.2·L1 + 0.5·edge_loss`、GANなし
- **実装のコード**: `lineart/unetgenerator.py::UNetGenerator`、`--model unet`で`--aux-dir`を指定しない；`experiments/run_combined_koma_direct_unet_20260801.sh`(3エポック)、`run_combined_koma_direct_unet_100ep_20260801.sh`(100エポック)、`run_combined_koma_direct_unet_dense_28ep_20260801.sh`(密データ5373タイル・28エポック、総勾配ステップ数は100エポック版と同じになるよう調整)、`run_combined_koma_direct_unet_200ep_20260801.sh`(1489タイル・200エポック、per-tile-exposure仮説の検証)、`run_combined_koma_direct_unet_finegrid_20260802.sh`(1489タイル・60エポック・5刻みチェックポイント、エポック軌跡の再現性検証)
- **目視評価**: 済み(3エポック版・100エポック版・密データ28エポック版・200エポック版・fine-grid軌跡版とも)
- **montageの場所**: `results/lessons/compare_combined_koma_direct_unet_20260801.png`(3エポック)；`results/lessons/compare_combined_koma_direct_unet_100ep_20260801.png`(100エポック)；`results/lessons/compare_combined_koma_direct_unet_dense_28ep_20260801.png`(密データ28エポック)；`results/lessons/compare_combined_koma_direct_unet_200ep_20260801.png`(200エポック)；`results/lessons/compare_combined_koma_direct_unet_200ep_20260801_trajectory.png`(200epランのエポック軌跡、20刻み)；`results/lessons/compare_combined_koma_direct_unet_finegrid_20260802_trajectory.png`(独立再学習、5刻み)
- **その他メモ**: 3エポック版はほぼ真っ白な出力(F1@2px 0.012)だが、学習lossは終了時点でも収束の兆しなく順調に低下中(0.377→0.292→0.273)だったため、公平な検証として100エポック版(約8時間、lossは最終的に0.377→0.137付近で緩やかに収束)を実行。**100エポック版の結果は今夜で最も興味深い**: F1@2px 0.187、chamfer 10.19(今夜最悪)、ink_ratio 0.924(GTとほぼ同量)。ユーザー確認: **これはleak修正以前のnotebook時代に見ていた出力そのもの**。montageを見ると、**cleanup/atari一族とは全く異なる失敗モード**であることが分かる — 出力は既存のどのモデルよりも明らかにcrispで完全な二値の黒線であり、soft/marbledではない(**このアーキテクチャは十分学習すればsoft/marbled天井を脱却できるという仮説を裏付ける**)。ただし線が細部で不安定(ぐらぐらしていて安定したストロークとして繋がりきらない)ため断片化して見える。roughとdirect_unet出力を並べて詳細比較したところ、**完全な無関係ノイズではなく、roughの描き込みが多い領域とdirect_unet出力が濃い領域は大まかに対応している**(例: 004_008行ではroughの垂直・斜めストロークの位置に対応する濃い線が出ている)— ただし具体的な形状(例: 004_009行の目の形)までは再現できていない。**正しい特徴づけは「無関係なノイズ」ではなく「roughをなぞろうとする意思はあるが、安定したストロークとして繋げられずぶれ・断片化として現れている」**(ユーザー表現: 自信のなさが「ボケ」ではなく「ぐらつき」として出ている)。この意味で、Direction 4(diffusion)の幻覚(入力と無関係な、もっともらしい別内容を自信満々に生成)より、素材(rough)との関係性という点ではむしろ健全な失敗モードと言える。ink_ratio(≈GT並みの量)とF1/chamfer(位置がまだ不正確)の組み合わせは「線の総量は合わせられたが、正確な位置に安定して線を置くところまでは学習できていない」ことを示している。**結論: 単段直接回帰はcrispさの面でatari一族に勝り、roughへの対応の意思もdiffusion系より明確だが、まだ安定したストロークとしては実用にならない。**

**データ量を増やす検証(2026-08-01、密データ28エポック版): むしろ悪化、データ量の問題ではなさそうという結果。** 既存5ソースを`--duplicate-overlap`/`--max-per-region`を緩めて密に再タイル化(1489→5373タイル、ただし同じ元ページからの重複クロップで内容の多様性は増えていない)、総勾配ステップ数を100エポック版と揃えた28エポックで学習。結果: F1@2px 0.150(100エポック版の0.187より悪化)、chamfer 10.95(同10.19より悪化)、ink_ratio 0.596(同0.924よりunder-ink化)。montageでも100エポック版よりさらに薄く/ソフトになっている。**タイル数を3.6倍にした分、1タイルあたりの露出回数は100回→28回に減っている**(総ステップ数は同じでも、タイル1枚を見る絶対回数が減る)。これがそのまま結果に出たと考えられ、**「ぐらつき」は総学習量やデータの多様性ではなく、同じ具体例への反復露出で安定化する現象らしい**という仮説を示唆している。だとすると次に試すべきは「データを増やす」ではなく「既存1489タイルでもっと長く(200エポック以上等)学習する」方向。dataset_4thの530タイル(真に新規の内容、重複クロップではない)はこの結果によって完全に否定されたわけではないが、同じ露出回数の理屈が当てはまるなら、対応するエポック数の伸長なしに追加しても同様に薄まる可能性が高い。

**200エポック版の結果(2026-08-02): 上記の「反復露出で安定化する」仮説は否定された。100エポックより明確に悪化。** 同じ1489タイルで反復回数を100→200に倍増したが、F1@2px 0.162(100エポック版0.187より悪化)、chamfer 10.80(同10.19より悪化)、ink_ratio 0.628(同0.924よりunder-ink化、密データ28エポック版0.596に近い水準まで後退)。新設のstroke-stability指標(`tools/evaluation/evaluate_stroke_stability.py`)でも同様: `long_component_ratio`(長い連続ストロークが占める割合、GT=0.819)は100エポック版0.625→200エポック版0.507とGTから遠ざかり、`components_per_1k_ink_px`(インク量あたりの断片数、GT=20.9)も100エポック版17.2→200エポック版24.9と悪化(=より断片化)。montage(`results/lessons/compare_combined_koma_direct_unet_200ep_20260801.png`)を目視しても、200エポック列は100エポック/密データ28エポック列と同系統の細く毛羽立ったスクリブルのままで、むしろ100エポック版より繊細で断片的に見え、GTへ近づいた印象はない。**結論: 28エポック(密データ)・100エポック・200エポックの3点を通して見ると、100エポック/1489タイルの地点が局所的な最適点であり、そこからタイル数を増やして反復を減らしても(28ep)、同じタイルで反復を増やしても(200ep)、どちらの方向にも悪化する。** 「反復露出を増やせば安定化する」という単純な仮説は誤りで、100エポック付近がこの小さいデータ規模(1489タイル)でのオーバーフィット境界に近いと考えられる。次に試すべきはエポック数のさらなる調整ではなく、(a) 明示的な連続性/平滑化損失項の追加、(b) 真に新規なデータでの拡張(未使用のdataset_4th 530タイル、または密リタイルで追加された分のQC見直し)、(c) aux-anchor系列のsoft/marbled天井への回帰、のいずれか。詳細は`doc/work_log.md` 2026-08-02の該当エントリを参照。

**エポック軌跡の読み直し(2026-08-02夕方): 「エポック40が最良点」という当初の読みは訂正、大枠の結論は維持。** 200エポック版ランが5エポックごとに保存していたチェックポイント(20/40/60/.../200)を1本の学習の軌跡として評価し直したところ、`long_component_ratio`/`components_per_1k_ink_px`がエポック40で明確なピークを打ち、そこから200まで単調に悪化するように見えた。しかし**別の独立した学習(`run_combined_koma_direct_unet_finegrid_20260802.sh`、5〜60エポックを5刻みで評価)で再現性を確認したところ、鋭いエポック40ピークは再現しなかった** — 指標によって最良エポックがばらけ(components/1kは40、long_component_ratioとchamferは50、F1は55)、目視でもep025〜055にかけて連続的に濃くなっていくだけで明確な折り返し点は見えない。加えて、この再学習ではエポック10・15で出力がほぼ完全に消える異常な谷(ink_ratio 0.001/0.010)が見つかった — エポック5では多少構造があったのに一旦崩壊し20から回復するという非単調な挙動で、3エポックスモークテストの「ほぼ真っ白」出力と同系統の学習初期不安定性と考えられる。**修正した結論**: 評価セットが8タイルしかなく、単一の最適エポックを一点に絞り込めるほどの精度はない。ただし「エポック30〜60あたりに妥当なプラトーがあり、そこからさらに100〜200まで伸ばすと明確に悪化する」という大枠は2回の独立実験で一致しており維持する。「エポック40が唯一無二の最良点」という言い方は撤回。詳細は`doc/work_log.md` 2026-08-02(夕方)の該当エントリを参照。

**白黒化とrough忠実さのトレードオフ(2026-08-02夜): ユーザーの目視直感が正しく、既存指標が食い違いを検知できていなかった。** ユーザーの読み: montageを見る限り線画らしい白黒(確信度・binarization)はep55あたりで既に十分だが、roughへの忠実さはep15あたりから崩壊し始めそこから悪化し続ける — つまり両者は別軸で、片方を伸ばすともう片方が犠牲になるトレードオフ関係にあるのではないか。**まず`tools/evaluation/evaluate_rough_fidelity.py`(新規、`score_pair_agreement.py`のCannyエッジ一致度ロジックをGT線ではなくroughと予測出力の対応度に転用)で定量検証したが、支持されなかった**: `edge_f1`(rough対応度)はep10-20でほぼゼロ(この時点で出力自体にCannyエッジが立たないほど平坦)からep40-160で0.36〜0.41まで上昇し、200で緩やかに下降するだけで、GT一致度指標とほぼ同じ「上昇→プラトー→緩やかな下降」の形。ep15からの連続悪化は見られなかった。**しかしクロップして並べた目視比較(`results/lessons/crop_compare_finegrid_20260802.png`、3サンプル×rough/ep05/ep15/ep35/ep55/GT、240x240px領域を3倍拡大)で、ユーザーの直感が正しいことが分かった** — ep05とep15はほぼ同じ見た目で、ぼやけてはいるがroughの具体的なストローク経路をかなり正確になぞっている(roughのソフトコピーに近い)。ep35/ep55になると黒く確信度は上がるが、**roughの特定のストローク経路とは違う場所に別の「もっともらしい線画テクスチャ」が生成される**ようになり、具体的な線の通り道の一致度は下がっていく。`edge_f1`がこれを検知できなかった理由: 許容誤差(3px)ベースのCanny位置一致は「roughのエッジ密集領域の近くにインクがあるか」しか見ておらず、「同じ具体的な曲線をなぞっているか」は判定できない — インク量が増えるほどたまたま近くにヒットする確率も上がり、スコアが下支えされてしまう(100エポック版のGT比較で既に指摘した「004_009の目の形は再現できていない」という盲点と同種)。**修正した理解**: 白黒化(確信度)とrough忠実さは同じ軸の両端ではなく、実質的にトレードオフの2軸。低エポックは「不正確だが位置に忠実」、高エポックは「確信度は高いが位置は近似的」。このアーキテクチャに二段階分解(まず精密だが弱いトレースを出し、それを別段階で確信度強化・binarizeする)を組み込む発想は、既存のaux+bounded-correction `cleanup`アーキテクチャと構造的に似ている(ただしcleanup系列にはatari生成器由来の別のsoft/marbled天井問題がある)。未実装・未検証、次の候補案として記録。

---

## データ分割によるablation(アーキテクチャではないが「なぜ自信がないのか」に直結)

これまでの全ての項目が対処しようとしている症状(自信のなさ)について、*アーキテクチャ以外*の説明を排除/確認するために実行したので、ここに含める。

### agreement-halo 再検証(3回)

- **目的**: rough/line対応度の高いタイルと低いタイルを*混ぜて*学習すると、モデルが自信を持てなくなる(ヘッジする)方向に働くか — 別々に学習した場合と比較
- **「アーキテクチャ」(実際はデータ分割)**: 上のmsganと同じ`cleanup`アーキテクチャ/loss；学習ファイルリストだけが違う(対応度スコアの上位/下位240または450タイル)
- **数式**: msganと同じ；対応度スコアの算出方法はテストごとに異なる(その他メモ参照)
- **実装のコード**: `tools/evaluation/score_pair_agreement.py`(汎用的なCanny+距離変換スコア)、`tools/evaluation/split_koma_by_tile_edge_f1.py`(パイプライン自身が計算するアライメント後`edge_f1`、より精密)；`experiments/run_agreement_halo_survey.sh`(2026-07-20、旧データ)、`run_combined_koma_agreement_halo_20260801.sh`、`run_combined_koma_tile_edge_f1_halo_20260801.sh`
- **目視評価**: 3回とも済み
- **montageの場所**: `results/compare_agreement_halo_e2.png`(2026-07-20)、`results/compare_combined_koma_agreement_halo_20260801.png`、`results/compare_combined_koma_tile_edge_f1_halo_20260801.png`
- **その他メモ**: **3回のテストで単調に効果が縮小し、ほぼゼロに収束**: 2026-07-20(データにまだ未解決の座標整合性問題あり)F1差0.155(high 0.436 vs low 0.281)；2026-08-01 汎用スコア・整合済みデータでは差0.009；2026-08-01 パイプライン純正のアライメント後`edge_f1`では差が**逆転**(-0.008、lowの方がわずかに高い)。**結論: 当初の効果の大半は、その後修正された座標整合性問題の副産物であり、内容的な対応の薄さそのものではなかった。** 新しい根拠なしに再検討しないこと。

---

## Direction 1/2/3/7: 単独検証なし

- **1: Multi-Scale PatchGAN** — 単独では未検証。採用中のmsganレシピに`--multiscale-gan`として直接組み込み済み(`lineart/model_zoo.py::MultiScalePatchDiscriminator`、スケール`(1.0, 0.5)`) — 上記の「msgan」を含む項目は全てこれを含んでいる。
- **2: Feature Matching Loss** — 同様、msganレシピに`--feature-match-weight 0.08`として組み込み済み(discriminatorの中間活性を使う`feature_matching_loss`)。Direction 1から単独で切り離したことはない。
- **3: Line Perceptual / Structure Feature Loss** — 2種類の実装があるが、どちらもloss項として組み込まれており単独検証はしていない: (a) `structure_pyramid_loss`(Sobel/DoG風の2スケール比較)、msganレシピで`--structure-weight 0.04`として使用；(b) `edge_loss`(Cannyベースのl1、`lineart/losses.py`)、2026-08-01のnoadv/direct-unetアブレーションで`--edge-weight 0.5`として使用。両者を比較したことも、「structure lossなし」と比較したこともない。
- **7: Soft Morphology / Line-Width Loss** — `soft_width_loss`は存在し、**koma以前の**サーベイ(`run_line_refiner_survey.sh`、`run_badrough_inkwidth_survey.sh`、`run_badrough_lucy_thin_*_survey.sh`)で`--width-weight`として使用済み — 現在の`combined_koma_20260729`データでは再実行しておらず、`doc/model_directions.md`に「Direction 7」の見出しでの結果記録もない。現データでの状況: **未検証**。

---

## このファイルを正直に保つために

- 数値だけを根拠に項目を追加しないこと。montageをまだ見ていない(またはまだ存在しない)場合は目視評価に「未レビュー」と書き、採用/不採用の判定はしないこと。
- 数値結果と目視レビューが食い違う場合は目視レビューを優先し、その食い違い自体をその他メモに一文書き残すこと(no-adversarial-loss ablationが典型例)。
- チェックポイントが更新された場合(v1→v2など)、上書きせず両方の項目を残すこと — 前バージョンの失敗モードこそが、修正を生んだ教訓であることが多い。
- 1項目の要約で足りない場合は、日付を頼りに`doc/work_log.md`を横断参照すること。
