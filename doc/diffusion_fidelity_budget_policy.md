# Diffusion Domain LoRA 忠実度バジェット・ポリシー

制定: 2026-08-07 JST (`diffusion`ブランチ)

## 位置づけ

このファイルは、domain-only LoRA(rough/lineをそれぞれ無条件・無ペアで生成する
現在の`diffusion`ブランチのアプローチ)の生成物を「良い」と判断する基準を単独で
まとめたもの。`doc/work_log.md`の該当エントリ(2026-08-06/07、base-checkpoint
比較の一連の実験)から派生した、今後の評価に継続して使うためのポリシー。

## ルール

生成物のうち、**LoRA(=我々が用意した学習タイル)由来が7割、baseチェックポイント
自身の描き癖由来が3割以内**なら「制御できている範囲」とみなす。

## 根拠

実際の制作工程(下絵 → 線画)でも、以下のような逸脱が現実に起きている。

- 下絵を描かずに線を描く
- 下絵を無視して(別の)線を描く
- 下絵に従って描いたが、より良くしたいがために描き直す

これは決して「劣化」ではなく、通常の作画プロセスに内在する逸脱である。我々が使って
いる下絵・線画ペアを比較すると、この種の不一致がおおむね3割程度で発生していると
いう分析がある。この現実の逸脱率を「制作上正常な範囲」の基準として流用し、LoRA
生成物の base 側への逸脱がこれと同程度(3割以内)なら許容、それを大きく超えるなら
「baseモデル自身の絵作りに引きずられすぎ」と判断する。

## 現時点でオープンな問題

3割という数値は今のところ決め打ちであり、次の点は未確定・今後精査していく:

- **一律の閾値として全指標に3割を適用してよいか**、それとも指標ごとに異なる許容幅を
  設定すべきか(例: 構図/内容に関わる指標はより厳しく、線幅のような表現上の揺らぎは
  もっと緩く、など)。
- **どの指標(軸)が3割ルールになじむか**。単一スコアではなく
  `tools/evaluation/measure_lineart_profile.py`の各軸ごとに判定する方針。

## 軸別の例外(2026-08-07 制定)

一律3割ではなく、以下の軸は個別の扱いとする。

- **`deep_black_ratio`(黒ベタ量)**: 3割ルールの合否判定から**除外**する。指標として
  計測・出力は継続するが、乖離が大きくても問題視しない。根拠: 我々が学習させている
  タイル(下絵/線画)は基本的に「ベタが入る前」の状態であり、real referenceの
  `deep_black_ratio`が低いのはそのため。base checkpoint側はおそらく逆(ベタを含む
  仕上げ済みイラストで学習されている)で、生成物にベタが出ること自体は実害がないと
  判断する。
- **`line_width_p50`(線幅)**: 現状程度(`domain_lora_line_sd15base_captiontags_20260807`
  + motif prompt サンプリングでの実測値、中央値7.39px、real 3.82px比93%乖離)までは
  容認する。ただし**これを上限とする** — これより太くなる方向へはさらに悪化とみなす。
  根拠: 線が太いのはクローズアップ構図(motif promptで`close-up`を明示した影響)に
  起因すると考えられ、現状程度の太さはむしろ好ましいという判断。
- **`components_per_1k_ink_px`(ストローク断片化密度)**: 許容幅を3割ではなく**5割**
  まで緩める。2026-08-07時点の実測(per-image caption + motif prompt版、real比
  40.8%乖離)はこの緩めた枠内。

これら以外の軸(`background_ratio` / `blank_cell_fraction` / `grid_ink_cv` /
`width_consistency` など、構図・疎密のコントラストに関わるもの)は、当面デフォルトの
3割ルールのまま。

**2026-08-07(later)追記**: `blank_cell_fraction` / `grid_ink_cv` /
`width_consistency` の3つは、ユーザー判断により**優先して追わない指標**と位置づけ
直した。線質(line_width_p50)・黒ベタ(deep_black_ratio)・断片化密度
(components_per_1k_ink_px)については上記の通り具体的な扱いが決まった一方、この
3つは「今のところ優先したい指標ではなさそうだ」という評価。3割ルールの対象からは
外さないが、判断の主軸としては扱わない。

## 現状の適用例(2026-08-06/07 base-checkpoint / capacity比較)

line-domain LoRAを3変種(AOM3A1B・rank16attn / SD1.5素ベース・rank16attn / SD1.5素
ベース・rank32+conv層拡張)で比較した際、各軸の相対乖離率
(`|生成値 - real中央値| / real中央値`)を計算した結果、軸が2つのグループに
はっきり分かれた。

**常に3割枠内(ストローク形状系) — 3変種とも合格:**

| metric | AOM3A1B | SD1.5素ベース | SD1.5+rank32+conv |
|---|---:|---:|---:|
| `faint_near_ink_ratio` | 26.3% | 28.6% | 8.4% |
| `long_component_ratio` | 4.1% | 7.6% | 13.1% |
| `components_per_1k_ink_px` | 22.8% | 11.3% | 11.6% |
| `long_line_ratio` | 3.7% | 5.3% | 13.7% |
| `orientation_entropy` | 8.2% | 4.3% | 12.0% |

**一貫して3割枠外(インク配置・量系) — capacity増強でむしろ悪化:**

| metric | AOM3A1B | SD1.5素ベース | SD1.5+rank32+conv |
|---|---:|---:|---:|
| `background_ratio` | 44.0% | 33.1% | 49.4%(悪化) |
| `blank_cell_fraction` | 95.5% | 84.1% | 100%(全サンプルで余白セルゼロ) |
| `grid_ink_cv` | 67.2% | 56.1% | 77.0%(悪化) |
| `width_consistency` | 18.1%(枠内) | 84.1% | 86.1%(悪化) |
| `ink_ratio` / `deep_black_ratio` | 676% / 6650% | 555% / 5450% | 998% / 17650%(大幅悪化) |

**注意点**: `ink_ratio`・`deep_black_ratio`はreal分布の中央値がほぼゼロに近いため、
単純な相対%では分母がゼロに近づき乖離率が発散する。これらの軸には絶対差、あるいは
real分布のIQR幅に対する差など、別の正規化方法が必要 — 3割ルールをそのまま適用
できない軸がある、という具体例。

**解釈**: ストローク形状(線の続き方・向きの多様性)は元々3変種とも合格範囲内で、
「内容・構図の逸脱」問題とは無関係だった。rank32+conv層への容量拡張は、1489枚と
いう比較的小さいデータ規模に対しては、我々のタイルへ忠実に寄せる方向ではなく、
SD1.5が元々持つ別の強い分布(少年漫画バトルもの、視覚的に確認)により強くアクセス
させる方向に働き、問題の軸(インク配置)をむしろ悪化させた。2026-08-07時点でこの
3変種の中で最も良いのは rank16・attn-onlyのSD1.5素ベース版
(`domain_lora_line_sd15base_20260806`)。「容量を上げれば忠実度が上がる」という
単純な仮説は否定された。

この一覧は「3割ルールを機械的に適用する前段階の整理」であり、公式な合否判定と
してまだ確定していない。今後の isolation 実験の結果が出るたびに、この表形式で
軸別に評価していく。

## 2026-08-07(later): Per-Image Caption + Motif Prompt -- 現時点のベスト

rank16・attn-only・SD1.5素ベースは固定したまま、学習キャプションを1489枚共通の
固定文から`scripts/tag_wd14.py`(WD14タガー)によるper-image danbooruタグ+スタイル
接尾辞に変更(`domain_lora_line_sd15base_captiontags_20260807`)。ただし学習後、
サンプリング時に元の汎用キャプションのままだと**ほぼ変化なし**(統計上ほぼ同一) --
学習時にモチーフ情報を学ばせても、生成時に指定しなければ引き出せないため。そこで
サンプリング時にも明示的なモチーフ(`"1girl, solo, close-up, white_background,
simple_background, ..."`)を指定したところ、目視でこれまでの line-domain 全variant
中最もreal referenceに近い結果になった。

軸別の変化(固定caption版 → per-image caption + motif prompt版):

| metric | SD1.5素ベース(固定caption) | +per-image caption+motif prompt | 判定 |
|---|---:|---:|---|
| `background_ratio` | 33.1%乖離 | **29.4%乖離** | 初めて3割枠内 |
| `faint_near_ink_ratio` | 28.6%乖離 | **1.5%乖離** | ほぼreal一致 |
| `grid_ink_cv` | 56.1%乖離 | 52.4%乖離 | 微改善(枠外のまま) |
| `width_consistency` | 84.1%乖離 | 80.0%乖離 | 微改善(枠外のまま) |
| `blank_cell_fraction` | 84.1%乖離 | 79.5%乖離 | 微改善(枠外のまま) |
| `components_per_1k_ink_px` | 11.3%乖離 | 40.8%乖離 | 5割枠(上記例外)内 |
| `line_width_p50` | 43.4%乖離 | 93.4%乖離 | 例外上限として容認 |
| `deep_black_ratio` | 5450%乖離 | 9240%乖離 | 判定除外(上記例外) |

「軸別例外」を適用した上での結論: 構図・内容の一致度(背景の見え方、faint画素の
局所性)は明確に改善し、`background_ratio`は初めて3割枠内に入った。黒ベタ・線幅は
例外ルールにより問題視しない。残る主要な未解決軸は `blank_cell_fraction` /
`grid_ink_cv` / `width_consistency`(余白の絶対量とストロークの太さの均一性)で、
これらは依然として大幅に枠外(この後、上記の通り優先指標から外れた)。

## 2026-08-07(later): LoRA推論時scaleの掃引 -- 品質と形状のトレードオフ

`cross_attention_kwargs={"scale": X}`(`scripts/sample_domain_lora.py`に
`--lora-scale`として追加、再学習不要)で、per-image caption + motif prompt
チェックポイントに対し scale 1.0/1.3/1.4/1.5/1.6 を掃引。

**重要な教訓**: scaleを上げるほど `blank_cell_fraction` / `grid_ink_cv` /
`components_per_1k_ink_px` などの構造指標は軒並みreal分布に近づく(scale1.6で
どれも3割枠内)。しかし目視では逆に、scale1.4以上で顔や体の形状が視認できない
抽象的な殴り書きに崩れていった。指標が「余白と密度のコントラストがあるか」しか
見ておらず、崩れて自信なく疎らに線を置いただけの状態でも同じような統計になって
しまうため。**このプロジェクトが繰り返し踏んできた「指標と目視の乖離」パターンの
典型例**として記録。scaleというノブ単体では品質(線質)と形状保持の両立ができず、
LoRAの重みそのものへの介入が必要という結論に至った。

## 2026-08-07(later): Rare Trigger Token -- 失敗と成功

「意味を持たないtrigger token」でLoRAの重みそのものに介入するアプローチ。

**1回目(失敗)**: スタイル接尾辞全体を`"sks style, monochrome, black and white"`に
置換。scale1.0ではLoRAの影響が弱く、baseモデル(SD1.5)自身の強いprior --
モノクロの実写人物"写真"(線画ではない)-- が表に出てしまった。scale1.3/1.4では
写真調と漫画線画調が同一画像内で混在し、これまでで最悪の結果に。**教訓**:
`"manga panel, monochrome line art"`はbaseモデルの余計な癖を運ぶノイズだった
だけでなく、「これは線画/漫画である」という必須のドメイン指定情報でもあった。
失敗した出力・チェックポイントは削除(`results/**`・`checkpoints/*`・`logs/`は
gitignore対象、git履歴への影響なし)。

**2回目(成功)**: ドメイン指定語は残し、スタイル記述部分のみ
`"sks style, monochrome line art, manga panel, black and white"`に変更。
scale1.0/1.3/1.4のいずれでも写真化は起きず。scale1.4で
`line_width_p50`が5.73px(real比50.0%乖離)まで細くなり、これは
per-image-caption版で設定した上限(7.39px、93.4%乖離)を大きく下回りながら、
ストローク形状系の指標(`faint_near_ink_ratio` 13.0%乖離、`long_component_ratio`
7.8%乖離、`long_line_ratio` 11.4%乖離、`orientation_entropy` 6.3%乖離)は全て
3割枠内を維持し、かつ**目視でも形状の崩壊が見られなかった**。「スタイル記述部分
だけを無意味なtokenに置き換える」ことで、品質と形状保持のトレードオフの限界点
自体を後ろにずらせた、と解釈している。

## 2026-08-07/08: Rough Domainへの適用 -- 同じ構成をそのまま転用、改善幅はlineより大きい

line domainで確立した isolation chain の結果(base checkpoint / capacityは
アーキテクチャ一般の知見として再検証せず転用、per-image caption + motif prompt +
LoRA scale + ドメイン語を残したrare trigger token の4点をそのまま組み合わせ)を
rough domainに直接適用。データは2026-08-06に確認済みの2,014枚クリーンプール、
caption接尾辞は`"sks style, pencil rough sketch, monochrome"`(ドメイン語
"pencil rough sketch, monochrome"は保持)。

結果、2026-08-05/06から未解決だった「平行ハッチへの一様な収束」がほぼ解消。
元の崩壊状態(`domain_lora_rough_20260804`)との比較:

| metric | rough_ref | 元のrough(崩壊状態) | sksv2 scale1.4 |
|---|---:|---:|---:|
| `width_consistency` | 2.70 | 7.3%乖離 | **3.4%乖離** |
| `orientation_entropy` | 0.79 | 8.4%乖離 | **0.7%乖離(ほぼ一致)** |
| `long_component_ratio` | 0.44 | 65.1%乖離 | **11.3%乖離(3割枠内)** |
| `background_ratio` | 0.90 | 44.2%乖離 | **32.2%乖離(枠に肉薄)** |
| `components_per_1k_ink_px` | 37.2 | 59.3%乖離 | **32.7%乖離(5割枠内)** |
| `grid_ink_cv` | 2.03 | 80.6%乖離 | 60.7%乖離(改善) |
| `blank_cell_fraction` | 0.72 | 100%乖離(余白ゼロ) | 86.9%乖離(改善だが依然大) |
| `faint_near_ink_ratio` | 0.44 | 83.7%乖離 | 45.1%乖離(改善) |

全軸でreal方向に改善し、line domainより改善幅が大きい(元の崩壊がより深刻だった
ため)。ユーザー目視確認: 「下絵の雰囲気はだいぶ出てる」。`blank_cell_fraction` /
`grid_ink_cv`が依然として最大のギャップで、line domainと同じ傾向(かつ既に
優先指標から外れている軸)。

- checkpoint: `checkpoints/domain_lora_rough_sd15base_sksv2_20260807/final`
- caption: per-image WD14タグ + `"sks style, pencil rough sketch, monochrome"`
- サンプリング: motif prompt(`"1girl, solo, close-up, sketch, ..."`)+
  `--lora-scale 1.4`

## 現在の採用構成(2026-08-08時点)

**line domain**:
- checkpoint: `checkpoints/domain_lora_line_sd15base_sksv2_20260807/final`
- caption: per-image WD14タグ + `"sks style, monochrome line art, manga panel,
  black and white"`
- サンプリング: motif prompt(`"1girl, solo, close-up, white_background,
  simple_background, ..."`)+ `--lora-scale 1.4`
- montage: `results/domain_lora_line_sd15base_sksv2_20260807_scale14/
  contact_sheet_domain_lora_line_sd15base_sksv2_20260807_scale14.png`

**rough domain**:
- checkpoint: `checkpoints/domain_lora_rough_sd15base_sksv2_20260807/final`
- caption: per-image WD14タグ + `"sks style, pencil rough sketch, monochrome"`
- サンプリング: motif prompt(`"1girl, solo, close-up, sketch, ..."`)+
  `--lora-scale 1.4`
- montage: `results/domain_lora_rough_sd15base_sksv2_20260807_scale14/
  contact_sheet_domain_lora_rough_sd15base_sksv2_20260807_scale14.png`

両ドメインとも base: SD1.5素(`v1-5-pruned-emaonly.safetensors`)、rank16・
attn-only。両ドメインでユーザー確認済み。次は
SDEdit系変換の再開検討(`doc/work_log.md`のNext Actions参照)。

**2026-08-21試行(不採用)**: line domainの学習プールに、未使用だった
`psd_line`抽出タイル2022枚を追加(1489→3511枚、同一レシピ)して再学習した
ところ、`background_ratio`/`long_component_ratio`/`components_per_1k_ink_px`
/`grid_ink_cv`/`blank_cell_fraction`など主要な構造指標がほぼ全て悪化し、
目視でも縦方向のハッチング/ストライプへの重度崩壊を確認(2026-08-05/06の
rough domain「平行ハッチ崩壊」と同系統)。**上記の採用構成のまま変更なし**。
詳細: `doc/work_log.md`(2026-08-21後半のエントリ)。

## 関連

- `doc/architecture_decisions.md`: `measure_lineart_profile.py`の指標グロッサリ
  (`grid_ink_cv`/`blank_cell_fraction`を含む)。
- `doc/work_log.md`: この方針が生まれた経緯の詳細な議論(2026-08-06/07エントリ)。
