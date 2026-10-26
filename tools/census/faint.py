"""`faint_alpha_905_20261027.csv` —— 依頼3の3列が埋まり、薄い画素が見える。

**2つの別のことを運ぶ:**

1. **`body_init` / `body_blocks` / `body_stored`** —— これで
   **「被覆が下限の層が何層か」が計数できる**(それまで「機序は確定・件数は未計数」)
2. **`alpha_ge1_ratio` と `alpha_1_25_px`** —— **`drawn_ratio` の閾は 26 なので、
   アルファ 1〜25 の画素は落ちている。** その落ちている量が見える

**この便が答えない**:
- **墨があるか。** 墨の閾は 128 で、アルファとは別の軸
- **薄い画素が線か滲みか。** 画素は見ていない(比だけ)
"""
import collections
import csv

# **描画の層 = `layer_type` のビット1 が立つもの**(本体がラスタ)。
#
# **2026-10-27 の誤り**: ここに `"2"` を入れていた。**`layer_type = 2` は
# 「マスクを持つ、ラスタでない層」= 塗り(トーン・ベタ)**で、本体がラスタではない
# (ビットの表: 1 = 本体がラスタ / 2 = 層のマスクがある)。
# **その結果「描画層 11,171」と出たが、それは塗りの層だった** ——
# 基盤が別経路で「`mask_fill` 本体の 87.7% が分類1」と報告した数と同じものである。
# **ビットの意味を取り違えると、母集団の名前が間違う。**
DRAWING_LAYER_TYPES = ("1", "3")
# **`layer_type == 2` = マスクを持ち、ラスタ本体が無い層。**
# **「塗り」と名付けてはいけない** —— 全数で型2は 119,619層で、
# **塗り 118,370 / 画像素材 799 / ベクター 437 / テキスト 13** を含む。
# `v_layer_kind == mask_fill` とは**同じ集合ではない**(基盤が 96層の差を踏んだ)。
# > **欲しい列が無い表は、それと相関する最も近い列を使う許可にならない。**
# > **軸を置き換えたら、置き換えをラベルに持っていく。**
LAYER_TYPE_EQ2 = ("2",)

# 落ちている量の階級。**閾は先に宣言する**(分布を見てから動かさない)
GAP_BUCKETS = ((0.0, "1 差なし"), (1e-6, "2 0 超〜0.001"), (0.001, "3 0.001〜0.01"),
               (0.01, "4 0.01〜0.1"), (0.1, "5 0.1 以上"))


def load(path: str, drops):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            if rec["error"].strip():
                drops.drop(f"抽出側が error を付けた行: {rec['error'][:40]}")
                continue
            rows.append(rec)
    return rows


def bucket(gap: float) -> str:
    label = GAP_BUCKETS[0][1]
    for lo, name in GAP_BUCKETS:
        if gap >= lo:
            label = name
    return label


def faint_gap(rows, restrict=None):
    """**`drawn_ratio`(アルファ≥26)が、アルファ≥1 からどれだけ落ちているか。**

    描画の層かどうかで分ける —— **落ちているのが描画の層なら、それは薄い線である。**
    """
    cells = collections.Counter()
    skipped = collections.Counter()
    for r in rows:
        if restrict is not None and r["page_path"] not in restrict:
            continue
        try:
            a = float(r["alpha_ge1_ratio"])
            d = float(r["drawn_ratio"])
        except ValueError:
            skipped["alpha_ge1_ratio か drawn_ratio が数でない"] += 1
            continue
        gap = a - d
        if gap < 0:
            skipped["**アルファ≥1 が drawn_ratio より小さい**(起きないはず)"] += 1
            continue
        is_draw = "yes" if r["layer_type"].strip() in DRAWING_LAYER_TYPES else "no"
        cells[(is_draw, bucket(gap))] += 1
    out = [[d, b, n] for (d, b), n in sorted(cells.items())]
    return out, skipped


def bound_classes(rows, classify, restrict=None):
    """`body_init` / `blocks` / `stored` から5つに分ける。**依頼3の母集団で計数できる。**

    **`rendered` で割る** —— **下限になる値すら無い層を「下限」と数えると母集団が膨らむ**
    (基盤の指摘、2026-10-27)。`rendered = 0` の層は `drawn_ratio` が空欄で、
    **下限にすらならない。**
    """
    cells = collections.Counter()
    rendered = collections.Counter()
    pages = collections.defaultdict(set)
    types = collections.defaultdict(collections.Counter)
    for r in rows:
        if restrict is not None and r["page_path"] not in restrict:
            continue
        cls = classify(r["body_init"].strip(),
                       int(r["body_blocks"] or 0), int(r["body_stored"] or 0))
        cells[cls] += 1
        if r["rendered"].strip() == "1":
            rendered[cls] += 1
        pages[cls].add(r["page_path"])
        types[cls][r["layer_type"].strip()] += 1
    return cells, pages, types, rendered
