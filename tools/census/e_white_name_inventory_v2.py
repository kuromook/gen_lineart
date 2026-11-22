#!/usr/bin/env python
"""**白い層の名前の棚卸し v2 —— 母集団の側の6列を足す**(Track E の頼み、2026-11-23。v1 は消さない)。
あわせて **作者の名前の裁定(2026-11-23)の3名前を全数で数える**。

- 行 = v1 `e_white_name_inventory_20261123.csv` の 135行(名前 × 種別)。足す列:
  `n_layers_all`(構造 DB の全層 437,561 のうち、その名前 × 種別。**`.lip` を含む**)/ `n_layers_all_clip` /
  `n_layers_white_ge099`(v1 の `n_layers`)/ `n_layers_already_excluded_v7`(a v7 の `template_excluded`。a の母集団 437,425 の外の層は数えない)/
  `n_units_all` / `years_all`
- 名前の照合は**ちょうど一致**(前後の空白・大小も変えない)。番号違い・後ろに字の付く変種は別の名前として数えるだけ
- 作者の裁定の3名前(`RULED`): 種別ごとの全数 / a v7 の状態 / 作者のラベル(役割の表 v3 の聞き取り)との衝突を印字する

出力 `data/census_full/e_white_name_inventory_v2_20261123.csv` / `data/census_full/author_ruling_bg_base_names_20261123.csv`(非公開)。
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
V1 = os.path.join(D, "e_white_name_inventory_20261123.csv")
AV7 = os.path.join(D, "a_verdict_v7_20261116.csv")
ROLE = os.path.join(D, "layer_role_table_20261112_v3.csv")
OUT = os.path.join(D, "e_white_name_inventory_v2_20261123.csv")
OUT_R = os.path.join(D, "author_ruling_bg_base_names_20261123.csv")
RULED = ["bg base", "base", "rendering_base_back0001"]
RULING = "OK　これらは背景のベースマスクなので名前で処理してしまおう(3Dのベースと心理トーンがある)"


def main() -> int:
    c = db.connect()
    unit = split_units.units(c)
    allr = c.execute("select l.page_path, l.layer_id, coalesce(l.name,''), v.v_layer_kind, p.format, p.year from layers l "
                     "join v_layer_kind v using (page_path, layer_id) join pages p using (page_path)").fetchall()
    av = {}
    for r in csv.DictReader(open(AV7, encoding="utf-8")):
        av[(r["page_path"], int(r["layer_id"]))] = r["a_verdict_v7"]
    print(f"構造 DB の層 {len(allr):,} / a v7 の母集団 {len(av):,}")
    g = collections.defaultdict(list)
    for p, i, n, k, f, y in allr:
        g[(n, k)].append((p, i, f, y))
    v1 = list(csv.DictReader(open(V1, encoding="utf-8")))
    rows = []
    for r in v1:
        xs = g[(r["layer_name"], r["v_layer_kind"])]
        ys = collections.Counter(y for *_, y in xs)
        rows.append(dict(layer_name=r["layer_name"], v_layer_kind=r["v_layer_kind"],
                         n_layers_all=len(xs), n_layers_all_clip=sum(f == "clip" for _, _, f, _ in xs),
                         n_layers_white_ge099=int(r["n_layers"]),
                         n_layers_already_excluded_v7=sum(av.get((p, i)) == "template_excluded" for p, i, _, _ in xs),
                         n_pages_white=r["n_pages"], n_units_white=r["n_units"], years_white=r["years"],
                         n_units_all=len({unit[wd(p)] for p, *_ in xs}),
                         years_all=" ".join(f"{y}:{n}" for y, n in sorted(ys.items()))))
    assert sum(r["n_layers_white_ge099"] for r in rows) == 4878
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    t20 = rows[:20]
    print(f"v2 {len(rows)} 行: 白い層 {sum(r['n_layers_white_ge099'] for r in rows):,} / 同じ名前×種別の全層 {sum(r['n_layers_all'] for r in rows):,}"
          f"(.clip {sum(r['n_layers_all_clip'] for r in rows):,})/ うち a v7 が既に落とした {sum(r['n_layers_already_excluded_v7'] for r in rows):,}")
    print(f"  上位20: 白い層 {sum(r['n_layers_white_ge099'] for r in t20):,} / 全層 {sum(r['n_layers_all'] for r in t20):,} / 既に落とした {sum(r['n_layers_already_excluded_v7'] for r in t20):,}")
    # 作者の裁定
    lab = collections.defaultdict(list)
    for r in csv.DictReader(open(ROLE, encoding="utf-8")):
        if r["author_stage"]:
            lab[r["name"]].append(r["author_stage"])
    print(f"\n作者の裁定(2026-11-23、逐語)「{RULING}」")
    out = []
    for n in RULED:
        for (nm, k), xs in sorted(g.items()):
            if nm != n:
                continue
            st = collections.Counter(av.get((p, i), "a の母集団の外") for p, i, _, _ in xs)
            out.append(dict(layer_name=n, v_layer_kind=k, n_layers_all=len(xs), n_layers_all_clip=sum(f == "clip" for _, _, f, _ in xs),
                            n_pages=len({p for p, *_ in xs}), n_units=len({unit[wd(p)] for p, *_ in xs}),
                            n_template_excluded_v7=st["template_excluded"], n_undecided_v7=st["undecided"],
                            n_not_excluded_v7=st["not_excluded"], n_outside_a=st["a の母集団の外"],
                            author_labels=" ".join(f"{a}:{m}" for a, m in collections.Counter(lab.get(n, [])).items())))
            print(f"  {n} / {k}: 全 {len(xs):,}(.clip {out[-1]['n_layers_all_clip']:,})/ 頁 {out[-1]['n_pages']:,} / 単位 {out[-1]['n_units']}"
                  f" / a v7: 既に落とした {st['template_excluded']:,}・未決 {st['undecided']:,}・除外しない {st['not_excluded']:,}・外 {st['a の母集団の外']}")
    print(f"  作者のラベル(聞き取り 265層)で同じ名前: " + (" / ".join(f"{n} {lab[n]}" for n in RULED if lab.get(n)) or "0"))
    near = collections.Counter(nm for nm, k in g for _ in [0] if nm not in RULED and any(nm.startswith(x) for x in ("bg base", "rendering_base")) )
    print("  裁定に入れなかった近い名前(ちょうど一致でない): " + " / ".join(f"{nm} {sum(len(g[(nm, k)]) for (m, k) in g if m == nm)}" for nm in sorted(near)))
    with open(OUT_R, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
    print(f"出力: {OUT}\n出力: {OUT_R}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
