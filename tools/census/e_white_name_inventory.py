#!/usr/bin/env python
"""**白さ ≥ 0.99 で a がまだ落としていない層の、層名の棚卸し**(Track E の頼み、2026-11-23。DB と白さの便の引き直し。新しい走行ではない)。

- 層 = 白さの便(16便、312,614層)の `drawn_as = raster` かつ `n_drawn` > 0 かつ `n_white / n_drawn` ≥ 0.99
  (E の 0.99 の行 12,119 を再現する述語)
- − a が落とした層(`a_verdict = template_excluded`)。**v3(E の数えた版)と v7(いまの版)の両方で引き、前後を印字する**。表の母集団は v7
- − 層名がちょうど `white`(E の印 `name_white_raster` と同じ述語)/ − 種別 `text`(条例11。raster の行なので 0 のはず。数える)
- 名前ごとに: 層名 / `v_layer_kind` / 層数 / 頁数 / 単位数(`split_units`)/ 年の札の並び(最も早い札から。時期の上限)
- 落とした数を理由別に印字する(規律9)

出力 `data/census_full/e_white_name_inventory_20261123.csv`(非公開。層名を含む)。
"""
import collections
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
WC = sorted(glob.glob(os.path.join(D, "white_column", "white_column_layers_b*_2026112*.csv")))
AV = {"v3": os.path.join(D, "a_verdict_v3_20261115.csv"), "v7": os.path.join(D, "a_verdict_v7_20261116.csv")}
OUT = os.path.join(D, "e_white_name_inventory_20261123.csv")
TH = 0.99


def main() -> int:
    drop, base = collections.Counter(), []
    for f in WC:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["drawn_as"] != "raster":
                drop["raster として描いていない"] += 1; continue
            nd = int(r["n_drawn"] or 0)
            if nd == 0:
                drop["n_drawn = 0"] += 1; continue
            if int(r["n_white"] or 0) / nd < TH:
                drop[f"白さ < {TH}"] += 1; continue
            base.append((r["page_path"], int(r["layer_id"])))
    print(f"白さの便 {len(WC)} 本: 白さ ≥ {TH} の raster の層 {len(base):,}(落とした: " + " / ".join(f"{k} {v:,}" for k, v in drop.items()) + ")")
    av = {}
    for v, f in AV.items():
        av[v] = {(r["page_path"], int(r["layer_id"])) for r in csv.DictReader(open(f, encoding="utf-8")) if r[f"a_verdict_{v}"] == "template_excluded"}
    for v in AV:
        print(f"  a が落とした層({v}): {sum(k in av[v] for k in base):,} → 残り {sum(k not in av[v] for k in base):,}")
    c = db.connect()
    meta = {}
    pages = sorted({p for p, _ in base})
    for i in range(0, len(pages), 900):
        ch = pages[i:i + 900]
        for p, lid, nm, kind, y in c.execute(
                f"select l.page_path, l.layer_id, coalesce(l.name,''), v.v_layer_kind, p.year from layers l join v_layer_kind v "
                f"using (page_path, layer_id) join pages p using (page_path) where l.page_path in ({','.join('?' * len(ch))})", ch).fetchall():
            meta[(p, lid)] = (nm, kind, y)
    unit = split_units.units(c)
    keep, d2 = [], collections.Counter()
    for k in base:
        if k in av["v7"]:
            continue
        nm, kind, y = meta[k]
        if nm == "white":
            d2["名前がちょうど white"] += 1; continue
        if kind == "text":
            d2["種別 text"] += 1; continue
        keep.append((k, nm, kind, y))
    print(f"  v7 の残りから落とした: " + (" / ".join(f"{k} {v}" for k, v in d2.items()) or "0") + f" → **棚卸しの層 {len(keep):,}**")
    v3keep = sum(1 for k in base if k not in av["v3"] and meta[k][0] != "white" and meta[k][1] != "text")
    print(f"  参考: 同じ述語を v3 で引くと {v3keep:,}(E の 4,878 と比べる)")
    g = collections.defaultdict(list)
    for k, nm, kind, y in keep:
        g[(nm, kind)].append((k, y))
    rows = []
    for (nm, kind), xs in g.items():
        ps = {k[0] for k, _ in xs}
        us = {unit[wd(p)] for p in ps}
        ys = collections.Counter(y for _, y in xs)
        rows.append(dict(layer_name=nm, v_layer_kind=kind, n_layers=len(xs), n_pages=len(ps), n_units=len(us),
                         years=" ".join(f"{y}:{n}" for y, n in sorted(ys.items()))))
    rows.sort(key=lambda r: (-r["n_layers"], r["layer_name"]))
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    cum = 0
    for j, r in enumerate(rows, 1):
        cum += r["n_layers"]
        if j in (1, 5, 10, 20, 50):
            print(f"  上位 {j} 名前で {cum:,} 層({cum / len(keep):.1%})")
    print(f"異なる名前 {len(rows):,} / 名前が空の層 {sum(r['n_layers'] for r in rows if not r['layer_name'])}")
    print(f"出力: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
