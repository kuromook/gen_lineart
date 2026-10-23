# Track M work log

## 2026-10-24 -- the tree was opened

Cut from `foundation` at `e9d680b` by user approval, after the proposal
(`doc/track_m_colour_manuscripts_proposal_20261024.md`) was written on the
foundation side. Nobody has worked in this tree yet; this entry records what it
was handed.

**The author answered three things on approval, and the first one changes the
order of the candidates**:

1. the structure is **a base layer that indicates the area, with 濃淡 layers on
   top of it** (with some exceptions), and **the base is what to go after**,
   because finding areas is the current target
2. **single illustrations go in** -- a colour-painted manga page is likely less
   tractable than a single illustration, which is made more carefully
3. open the tree

Consequence of (1): the area division is **already a layer**, not something to
recover from composited pixels. Candidate 1 of the proposal (quantise the
composite, take connected components) is handicapped by exactly the 濃淡 that
sits on top; candidate 2 (read the layer structure) becomes the main line. **The
reordering comes from the author's process knowledge and not from a measurement,
and the registration has to say so.** What must not follow from it is deciding
which layer is the base **by its name** -- that is the 2026-10-23 pattern (four
times in one day) and the condition has to be measured and registered.

Verified at birth, not assumed:

- `git check-ignore -q data/a.png` passes in this tree -- the rule arrived with
  `f7372e6` and this branch was cut after it. 10 of the 14 trees on the shelf
  did not have it on 2026-10-24
- `doc/CURRENT.md` was replaced with a pointer the same day, before any session
  read it (audit check 8; Track J found its copy 14 days and 3,682 lines behind)

---

## 2026-10-24: 最初に座ったセッション —— 器具の確認(結果ではない)

登録は `doc/track_m_registration_20261024.md`(測る前に書いた)。
器具は `tools/colour/m1_page_cut_calibration.py` と
`tools/colour/m2_partition_metrics.py`。**腕は1つも採点していない。**

### 1. 既知セルは再現した

`inbox/work_color_mode_pages.csv`(1,348行)から抽出側の作品単位の判定を手元で
再構成し、**289作品の `測定ページ` / `カラーページ` / `判定` が mismatch 0**。
**全カラー 28 / 混在 47 / 全モノクロ 214** をこの木から再現したので、2本の CSV は
同じ走査であり、通達に書かれた規則(色画素率 >= 0.02)が実際に走った規則である。
受領の点検: 6列に欠けなし、`(年, 作品)` が両 CSV で完全一致、測定ページは
5が256作品 / 4以下が33作品(合計1,348)。CSV は **UTF-8 BOM**。

### 2. ページ側の棒は較正した。谷は 0 にしかない

色画素率の分布: **ちょうど 0 が 1,173ページ(87.0%)**、0 < x < 0.0001 が 1、
0.0001〜0.02 が 27、0.02以上が 147。**0.001 から 0.05 のあいだに谷はない**
(0.0053, 0.0054, 0.0075, 0.0086, … と連続する)。
ゆえに **2% の棒は谷の上に立っていない**。棒を 0.001 → 0.20 に動かすとカラー側は
171 → 82ページ、0.001 と 0.02 のあいだでは 171 → 147。
**棒の任意性が触るのは最大28ページ(走査の 2.1%)で、それ以上ではない。**

**登録した選別は `色画素率 > 0`。** この在庫が支持する棒はこれだけである。

### 3. 画素側の棒は較正できない —— そして、できてもこの木の問いには足りない

**彩度 >= 24 はこの在庫からは較正できない。** 在庫にあるのは彩度24で切った**後**の
割合だけで、ページあたりの彩度ヒストグラムが無い(依頼1)。
**借りてきた数として使わない。**

それと別に、2つ測った:

- **平均彩度は第2の軸ではない。** 色画素率との spearman **0.9646**
  (色画素率 > 0 の175ページ)、全1,348ページで pearson 0.9247。在庫の軸は1本
- **色画素率は「塗り切られているか」を順位づけない。** 全カラー28作品の70ページで
  色画素率は **0.0274 〜 0.9726(36倍)**。`2017 illustration`(79p、全カラー)は
  0.0645〜0.3554、`2018 illust201710`(10p、全カラー)は 0.0274〜0.1470。
  逆に 0.5 を超える40ページのうち15ページは**混在**作品にある

**合成の彩度は面の数を知らない。** これは作者の知識(面はすでに層である)と同じ方向を
指すが、**独立な支持ではない**——同じ対象(合成画素)を測っているから。

### 4. 面の分割に対する region-F1 を書いた(借りた規則のまま)

借りた `region_metrics.match` は**二値マスクの連結成分**を面にする。カラーの面の分割は
余白以外を覆うので、二値にすると面同士が接して1つの塊になり、**連結成分は面を返さない**。
ゆえに `m2_partition_metrics.py` に**ラベル地図どうしの照合**を書いた。規則は同一
(8近傍・面積の床・IoU 降順の貪欲1対1・調和平均)。器具確認は期待値を先に書いて全て PASS:

| | 内容 | 結果 |
|---|---|---|
| IC-M1 | 二値から作ったラベル地図で借りた `match` と4量が完全一致(IC5 2配置 + 乱数6配置) | 8/8 PASS |
| IC-M2 | ラベル置換で不変(matched 2/3、region-F1 0.6667 の非自明セル) | PASS |
| IC-M3 | 自分自身で 1.0(matched 3/3)、背景だけで 0.0 | PASS |
| IC-M4 | 100px の正方形を 33px ずらすと IoU 0.5038 で照合、34px で 0.4925 で不照合 | PASS |

借りた側の自己点検(`region_metrics.py` の assert セルと IC5)も**この木で走らせて PASS**。
`MIN_REGION_PX = 500` と `MATCH_IOU = 0.5` は **Track J/L がモノクロ合成の都合で
決めた値なので継承しない**。既定値として置き、カラーの面の面積分布を見てから登録し直す。

### 5. 塞がっているもの

**この木にはカラー原稿のラスタが1枚も無い。** `../lineart/data/incoming_manuscript/` は
`ako6` / `ako7` のベタ・トーン・グラデ(モノクロ)と part_layers の5ページだけで、
**どれもカラーではない**。ゆえに**自明な腕 A0 と帰無値 N1/N2/N3 は配達を待つ**
(登録は済み、`doc/track_m_registration_20261024.md` §3.2。GPU は要らない)。

**既存の層一覧の列を数えて確認した**: `part_layers/*_layers.json` と
`inbox/page_layers_priority65.csv`(20列)の**どちらにも彩度・色数の列が1つも無く**、
画素の量はすべてグレースケール(`black_among_drawn` / `drawn_gray_mean` / `preview_dark`)。
**現行の列ではカラーのベース層を同定できない。** 合成モードの列も無い。
依頼4本(彩度ヒストグラム / 全ページ走査 / 層一覧の色の列 / ラスタ)は登録 §4。
**Track L の 2,963層・原寸 export の後ろに並べる。**

**彩度の走査は census ではない**: 1,348 / 7,634ページ(**17.7%**)の標本である。
混在47作品のカラーページ「およそ468」は標本比を掛けた推定で、測定ではない。

### 6. 抽出側へ依頼4本を送った(2026-10-24 08:17)

便: `outbox/request_m_colour_pages_20261024.md`(9,940 bytes、便ごとに残す)。
**Track L の 2,963層・原寸 export の後ろに並べる**と明記した。

着弾確認: `dir` で 9,940 bytes、**SHA256 が両側で一致**
(`cb10e39a607671378f3237d398bc5bf199372f005a695b044624fd3191f729ca`)。
`type` / PowerShell の表示が化けるのは SSH 越しのコンソールのコードページで、
**ファイルは壊れていない**(ハッシュ一致が根拠)。

便の中身:

- **依頼1+2(1回の走査でまとめられる)**: 全7,634ページの彩度走査 + ページあたりの
  彩度ヒストグラム(8刻み32箱)。彩度の定義も一行もらう。いまの在庫は 17.7% の標本
- **依頼3**: カラーページの層の全数一覧に **(a) 層の描画画素の彩度分布、
  (b) 量子化後の色数と最大色の占有率、(c) 合成モード、(d) フォルダ内の重ね順**を足す。
  **(a)(b) が本体。** 既存20列はすべてグレースケールで面を見分けない
- **依頼4**: **まず8ページのパイロットだけ。** 作者名指しの4作品から色画素率の
  高い端と低い端を1つずつ(0.0274〜0.9726 を跨がせる。端で条件が成り立つかを見るため)。
  原寸・カラーのまま・2値化しない・アルファ残す。0.5GB を超えそうなら
  **縮小せずに枚数と合計サイズを先に知らせてもらう**(縮小すると平坦さと境界が消える)

便には**こちら側で分かったことも返した**: 2% の棒は谷の上に立っていないが
**触るのは最大28ページ(2.1%)で、そちら側に直すものはない**こと、
色画素率は塗り切り度を順位づけないこと(全カラー作品で 36倍の広がり)、
平均彩度が第2の軸でないこと(spearman 0.9646)。
質問1本: **カラーの作品に線画層があるか**(依頼3の一覧で分かるので先に答は要らない)。
