#!/usr/bin/env python
"""**形式に `PatternStyle` が残っている頁** —— 基盤の発注(2026-11-02)、抽出側の納品(2026-11-03)。

**これはパターンブラシ(④)の同定ではない**(規律13)。**ベクターの ④ には既知セルが無い**
(作者が名指しした層はラスター)。**ラスターで引いた ④ はこの表に一切映らない** ——
**0 の頁に ④ が無いとは言えない**(条例6)。

**材料**: `clip_brushstyle_patternstyle_pages_20261103.csv`(抽出側。頁ごと1行。埋め込みの
SQLite の表と行の数だけ。画素も `VectorData` も読んでいない)。

**既知セル**(T132。この器具が作っていない入力が2つ):

1. **供給元が便に書いた合計**(ベクター層の在る頁 2,839 / `BrushStyle` の表の在る頁 2,856 /
   `PatternStyle != 0` の頁 18・行 58 / `FirstPatternImage` 18 / 食い違い 95・78 / 表の無い頁 4,806)
2. **構造 DB の層の表**(`layers.vector_objects > 0` の層の数)—— **別の器具が作った列。**
   **DB の `n_vector_layers` はコマ枠の層を外して数えている**(`NOT has_frame_line`)ので、
   **比べるのはコマ枠を含めた数**。**コマ枠を外した数は別の列に出す**(T138: 同じ列名に2つの量を入れない)

**届く範囲**(名前より狭く書く): **形式の表に値が残っているか、まで。**
**`PatternStyle` の値の意味は誰も知らない**(抽出側: 不明)。**ストロークが参照しているかも読んでいない。**

**落とすもの**(規律9。器具が理由別に印字する): 作者の恒久の除外 / 構造 DB でも開けない `.clip` /
`.lip`(開けたが中身は未測定)

    ./venv/bin/python tools/census/pattern_style.py
"""
import collections
import csv
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import exclusions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = "inbox/clip_brushstyle_patternstyle_pages_20261103.csv"
NOTICE = "deliver_census_brushstyle_patternstyle_pages_20261103.md"
DB = "/home/sh1/deepl/census_db/manuscripts.duckdb"
OUT = "census/pattern_style_pages.csv"
# **供給元が便に書いた数**(既知セル1)
SUPPLIER = {"n_pages_vector_any": 2839, "n_pages_brushstyle_table": 2856,
            "n_pages_patternstyle_nonzero": 18, "n_patternstyle_rows": 58,
            "n_pages_first_pattern_image": 18, "vec0_with_table": 95,
            "vec_without_table": 78, "no_table": 4806, "clip_rows": 7662, "lip_rows": 1986}
TRUE = {"True", "true", "1"}


def year_of(page_path: str) -> str:
    return page_path.split("\\")[2]


def main() -> int:
    print("  [形式に `PatternStyle` が残っている頁(④の同定ではない)]")
    excluded = exclusions.load()
    con = duckdb.connect(DB, read_only=True)
    db_pages = dict(con.execute("select page_path, error from pages").fetchall())
    vec = {p: (a, b) for p, a, b in con.execute(
        "select page_path, count(*) filter (where vector_objects > 0), "
        "count(*) filter (where vector_objects > 0 and not has_frame_line) "
        "from layers group by 1").fetchall()}
    with open(os.path.join(ROOT, SRC), encoding="utf-8-sig", newline="") as fh:
        rows = {r["page_path"]: r for r in csv.DictReader(fh)}

    drops = collections.Counter()
    for p, err in db_pages.items():
        if p in rows:
            continue
        if p in excluded:
            drops["作者の恒久の除外(納品にも無い)"] += 1
        elif err:
            kind = "SQLite の塊が無い" if "CHNKSQLi" in err else ".clip でない"
            drops[f"構造 DB でも開けない `.clip`({kind}。納品の母集団に入っていない)"] += 1
        else:
            drops["**理由の分からない欠け**"] += 1
    stray = set(rows) - set(db_pages)
    if stray:
        drops["**納品に在って母集団に無い頁**"] += len(stray)

    agg = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    db_cmp = collections.Counter()
    for p, r in rows.items():
        if r["file_kind"] == "lip":
            drops["`.lip`(開けた・中身は未測定)"] += 1
            tot["lip_rows"] += 1
            continue
        tot["clip_rows"] += 1
        nv = int(r["n_vector_layers"])
        table = r["has_brushstyle_table"] in TRUE
        nps = int(r["n_patternstyle_nonzero"])
        a, b = vec.get(p, (0, 0))
        db_cmp["一致" if nv == a else "不一致"] += 1
        c = agg[year_of(p)]
        for key, val in (("n_clip_pages", 1), ("n_pages_vector_any", nv > 0),
                         ("n_pages_vector_nonframe", b > 0),
                         ("n_pages_brushstyle_table", table),
                         ("n_pages_patternstyle_nonzero", nps > 0),
                         ("n_patternstyle_rows", nps),
                         ("n_pages_first_pattern_image", r["has_first_pattern_image"] in TRUE),
                         ("n_pages_ps_nonframe", nps > 0 and b > 0),
                         ("vec0_with_table", nv == 0 and table),
                         ("vec_without_table", nv > 0 and not table),
                         ("no_table", not table)):
            c[key] += int(val)
            tot[key] += int(val)
    for why, n in sorted(drops.items()):
        print(f"    数えなかった頁 {n:,} —— {why}")

    bad = 0
    print("    **既知セル1(供給元の合計)**")
    for k, v in SUPPLIER.items():
        ok = tot[k] == v
        bad += not ok
        print(f"      {k:32s} 供給元 {v:>6,} / ここ {tot[k]:>6,}  {'一致' if ok else '**不一致**'}")
    print(f"    **既知セル2(構造 DB、コマ枠を含むベクター層の数)**: "
          f"一致 {db_cmp['一致']:,} / 不一致 {db_cmp['不一致']:,}")
    bad += db_cmp["不一致"] > 0
    bad += drops["**理由の分からない欠け**"] + drops["**納品に在って母集団に無い頁**"]
    print(f"    **コマ枠以外のベクター層を持つ頁 {tot['n_pages_vector_nonframe']:,}**"
          f"(コマ枠を含めると {tot['n_pages_vector_any']:,})/ "
          f"`PatternStyle != 0` の頁のうちコマ枠以外のベクター層を持つ頁 {tot['n_pages_ps_nonframe']}"
          f"/{tot['n_pages_patternstyle_nonzero']}")

    cols = ["n_clip_pages", "n_pages_vector_any", "n_pages_vector_nonframe",
            "n_pages_brushstyle_table", "n_pages_patternstyle_nonzero", "n_patternstyle_rows",
            "n_pages_first_pattern_image"]
    out = [[y] + [agg[y][k] for k in cols] + [""] for y in sorted(agg)]
    out.append(["ALL"] + [tot[k] for k in cols] + [""])
    out += [[""] * (len(cols) + 1) + [n] for n in NOTE]
    n = bc.write_csv(os.path.join(ROOT, OUT), ["year_label", *cols, "note"], out)
    print(f"    {os.path.basename(OUT)} {n}行")
    return 1 if bad else 0


NOTE = [
    "**④(パターンブラシ)の同定ではない**(規律13)。**示すのは「ベクターのブラシの型の表に `PatternStyle` の値が"
    "残っている頁」だけ。** **ベクターの ④ に既知セルは無い**(作者が名指しした層はラスター)。"
    "**ラスターで引いた ④ はこの表に一切映らない —— 0 の頁に ④ が無いとは言えない**(条例6)",
    "**測ったのは抽出側**(um350、2026-11-03。埋め込みの SQLite の表と行の数だけ。画素も `VectorData` も読んでいない)。"
    "**census は頁ごとの行から数え直し、供給元の合計と全部一致した。** **`PatternStyle` の値の意味は誰も知らない**(先方: 不明)",
    "**`n_pages_vector_any` はコマ枠の層を含む**(先方の定義: `VectorObjectList` に行が在る層)。"
    "**`n_pages_vector_nonframe` はコマ枠の層を外した数で、構造 DB の `n_vector_layers` と同じ定義** —— "
    "**2つは別の量。** **ベクター層の在る頁の大半は、ベクターがコマ枠だけの頁である。** "
    "**構造 DB の `vector_objects > 0` の層の数と、納品の `n_vector_layers` は `.clip` の全頁で一致**(別の器具どうし)",
    "**母集団**: 9,653頁のうち `.clip` 7,662頁。**落とした頁**: `.lip` 1,986(開けたが中身は未測定)/ "
    "**構造 DB でも開けない `.clip` 4**(先方の母集団 9,649頁には初めから入っていない。**先方の「読めない 0」はその 9,649頁に対する数**)/ "
    "作者の恒久の除外 1",
    "**「`BrushStyle` はベクター層の在る頁にだけ在る」は全数では成り立たない**(先方の前の言明を先方が直した): "
    "**ベクター層 0 なのに表が在る頁 95**(先方: 95頁とも `VectorObjectList` の表は在るが行が 0。消した跡と推定)/ "
    "**ベクター層が在るのに表が無い頁 78**(理由は不明)。**`FirstPatternImage` と `PatternStyle` の食い違いは1頁ずつ**(関係は不明)",
    "**同じ名前の頁が年の札をまたいで対で出ている**(先方: 版違いと推定、判定はしていない)—— "
    "**双子を畳んでいない。年をまたいで足すと同じ頁を2回数えうる**(版の同定は作者に問う。条例9)",
]

if __name__ == "__main__":
    raise SystemExit(main())
