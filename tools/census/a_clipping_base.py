#!/usr/bin/env python
"""**クリッピングの土台の札 `a_is_clipping_base`**(Track E の頼み、2026-11-23。DB の引き直し。新しい走行ではない。a の点数は動かない)。

- 層 = a v7 の母集団 437,425(`a_verdict_v7_20261116.csv`)。DB に在って a に無い層は数えて外す
- 土台 = 同じ親の中で、その層のすぐ上(`order_in_parent` の大きい側。`z_global` と逆向きの組を数えて印字)から続く `clip_raw = 1` の兄弟が1つ以上。
  **その層自身が `clip_raw = 1` でも土台になりうる**(CSP はクリッピングの列の一番下の非クリッピングの層に切る —— 自身がクリッピングの層は土台にならない、として数える。列に書く)
  - `n_clipped_direct` = 続く兄弟の数 / `n_clipped_layers` = そのうちフォルダーでない層 + フォルダーの中の層(フォルダーごと切られる)/ `n_clipped_folders`
  - `n_clipped_3d` = そのうち種別 `3d` か名前が `rendering_` で始まる層(名前で引けない 3D は数えない。下限)
- **自身が `clip_raw = 1` の層は土台にしない**(切られる側)。その数を印字する
- 名前の棚卸し v2 の **率(白い層 ÷ 全層)≥ 0.5 の上位20のうちの行**について、名前 × 種別ごとの土台の数を印字する(E の §4)。層名は表と印字に出る(非公開)

**v2(11-23 E の頼み)**: 土台ごとに、乗っている層(フォルダーでない層)の a v7 の判定を数える3列 `n_clipped_excluded` / `n_clipped_not_excluded` / `n_clipped_undecided`(と a の母集団の外 `n_clipped_outside_a`)を足す。v1 の表は消さない

出力 `data/census_full/a_clipping_base_v2_20261123.csv`(437,425行: page_path, layer_id, a_is_clipping_base, n_clipped_direct, n_clipped_layers, n_clipped_folders, n_clipped_3d)と
`data/census_full/a_clipping_base_by_name_20261123.csv`(棚卸しの上位20の名前 × 種別)。非公開。
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
AV7 = os.path.join(D, "a_verdict_v7_20261116.csv")
INV = os.path.join(D, "e_white_name_inventory_v2_20261123.csv")
OUT = os.path.join(D, "a_clipping_base_v2_20261123.csv")
OUT_N = os.path.join(D, "a_clipping_base_by_name_v2_20261123.csv")


def main() -> int:
    c = db.connect()
    rows = c.execute("select l.page_path, l.layer_id, l.parent_id, l.order_in_parent, l.z_global, l.clip_raw, l.folder, "
                     "coalesce(l.name,''), v.v_layer_kind from layers l join v_layer_kind v using (page_path, layer_id)").fetchall()
    L = {(r[0], r[1]): r for r in rows}
    kids = collections.defaultdict(list)
    for r in rows:
        kids[(r[0], r[2])].append(r)
    no_ord = 0
    for v in kids.values():
        no_ord += sum(r[3] is None for r in v)
        v.sort(key=lambda r: (r[3] is None, r[3] or 0, r[4] or 0))
    bad = sum(1 for v in kids.values() for a, b in zip(v, v[1:]) if None not in (a[3], b[3], a[4], b[4]) and a[4] > b[4])
    print(f"DB の層 {len(rows):,} / order_in_parent が空 {no_ord} / 兄弟の並びで order と z が逆向きの組 {bad}")
    pos = {}
    for v in kids.values():
        for j, r in enumerate(v):
            pos[(r[0], r[1])] = j

    def desc(r):
        out = []
        for k in kids.get((r[0], r[1]), []):
            out.append(k)
            if k[6]:
                out += desc(k)
        return out

    st = {}
    for r in csv.DictReader(open(AV7, encoding="utf-8")):
        st[(r["page_path"], int(r["layer_id"]))] = r["a_verdict_v7"]
    res, vc = {}, {}
    for k, r in L.items():
        if r[5] == 1:
            res[k] = (0, 0, 0, 0, 0); vc[k] = collections.Counter(); continue
        sib = kids[(r[0], r[2])]
        cl = []
        for s in sib[pos[k] + 1:]:
            if s[5] != 1:
                break
            cl.append(s)
        allc = []
        for s in cl:
            allc.append(s)
            if s[6]:
                allc += desc(s)
        lay = [s for s in allc if not s[6]]
        vc[k] = collections.Counter(st.get((s[0], s[1]), "outside") for s in lay)
        res[k] = (int(bool(cl)), len(cl), len(lay), sum(1 for s in allc if s[6]),
                  sum(1 for s in lay if s[8] == "3d" or s[7].startswith("rendering_")))
    a = [(r["page_path"], int(r["layer_id"])) for r in csv.DictReader(open(AV7, encoding="utf-8"))]
    miss = [k for k in a if k not in res]
    print(f"a v7 の母集団 {len(a):,} / DB に無い {len(miss)} / DB に在って a に無い {len(set(res) - set(a)):,}(外した)")
    n_self = sum(1 for k in a if k in L and L[k][5] == 1)
    out = []
    for k in a:
        x = res.get(k, ("", "", "", "", ""))
        out.append(dict(page_path=k[0], layer_id=k[1], a_is_clipping_base=x[0], n_clipped_direct=x[1],
                        n_clipped_layers=x[2], n_clipped_folders=x[3], n_clipped_3d=x[4],
                        n_clipped_excluded=vc[k]["template_excluded"], n_clipped_not_excluded=vc[k]["not_excluded"],
                        n_clipped_undecided=vc[k]["undecided"], n_clipped_outside_a=vc[k]["outside"]))
    base = [o for o in out if o["a_is_clipping_base"] == 1]
    print(f"自身がクリッピングの層(土台にしない) {n_self:,} / **土台 {len(base):,} 層**(乗っている層の計 {sum(o['n_clipped_layers'] for o in base):,}、"
          f"うち 3D と名前か種別で分かる {sum(o['n_clipped_3d'] for o in base):,})")
    cnt = collections.Counter(st[(o["page_path"], int(o["layer_id"]))] for o in base)
    print("  土台の a v7: " + " / ".join(f"{k} {v:,}" for k, v in cnt.most_common()))
    print("  土台の判定 × 乗る層の判定(層、v7):")
    for v in ("template_excluded", "not_excluded", "undecided"):
        bb = [o for o in base if st[(o["page_path"], int(o["layer_id"]))] == v]
        print(f"    土台 {v} {len(bb):,}: 乗る層 {sum(o['n_clipped_layers'] for o in bb):,} = 落とした {sum(o['n_clipped_excluded'] for o in bb):,}"
              f" / 除外しない {sum(o['n_clipped_not_excluded'] for o in bb):,} / 未決 {sum(o['n_clipped_undecided'] for o in bb):,}"
              f" / a の外 {sum(o['n_clipped_outside_a'] for o in bb):,}(除外しないが1つ以上乗る土台 {sum(1 for o in bb if o['n_clipped_not_excluded']):,})")
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
    inv = list(csv.DictReader(open(INV, encoding="utf-8")))[:20]
    byname = collections.defaultdict(list)
    for k, r in L.items():
        byname[(r[7], r[8])].append(k)
    nrows = []
    print("\n棚卸し v2 の上位20(率 = 白い層 ÷ 全層):")
    for j, r in enumerate(inv, 1):
        ks = [k for k in byname[(r["layer_name"], r["v_layer_kind"])] if k in res]
        b = [k for k in ks if res[k][0]]
        ratio = int(r["n_layers_white_ge099"]) / int(r["n_layers_all"])
        nrows.append(dict(rank=j, layer_name=r["layer_name"], v_layer_kind=r["v_layer_kind"], ratio_white=round(ratio, 4),
                          n_layers_all=len(ks), n_clipping_base=len(b), n_clipped_layers=sum(res[k][2] for k in b),
                          n_clipped_3d=sum(res[k][4] for k in b)))
        x = nrows[-1]
        print(f"  {j:2d} 率 {ratio:.3f} {'≥0.5' if ratio >= 0.5 else '<0.5'} 全 {x['n_layers_all']:,} / 土台 {x['n_clipping_base']:,} / 乗る層 {x['n_clipped_layers']:,}(3D {x['n_clipped_3d']:,})")
    with open(OUT_N, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(nrows[0])); wr.writeheader(); wr.writerows(nrows)
    print(f"出力: {OUT}\n出力: {OUT_N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
