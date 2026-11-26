#!/usr/bin/env python
"""**『下地』の系統から5枚の抜き取り**(Track E の通達 `note_census_e_shitaji_sample5_order_20261126.md`。
ユーザーの指示 11-26「下地の系統は直接ファイルを見るのが早そうだ　5枚ぐらい抜き取りで見る」。**新しい走行ではない** —— DB の引き直し)。

母集団 = 名前がちょうど `下地 1`〜`下地 4` の層(E の指定)。取り方は E が数を見る前に決めたもの:
1. 名前で割る: `下地 1` 2 / `下地 2` 2 / `下地 3` 1。`下地 4` は外す(1層、名乗る)
2. 同じ名前の2枚は違う年の札から。**E の決まりに無い所を census が埋めた(数を見る前に、ここに書く)**:
   2枚取る名前は、その名前の層が在る札の**最も早い札と最も遅い札**(散らすため)。1枚の名前は札を問わず名前の中で取る
3. 名前 × 札の中で、頁の全層数(フォルダーを含む構造 DB の層の行)が中位の層。偶数なら下の中位。同点は頁の道・層 id の順
4. 同じ単位(`split_units`)から2枚取らない —— 当たったら中位に次に近い層へ(取れなければ書く)

列は E の §3。白さは白さの列の便(`white_column_layers_b*.csv`)、乗る層の数は `a_clipping_base_v3_v8_20261123.csv`。
出力 `data/census_full/e_shitaji_sample5_20261126.csv`(層名を含むので非公開)。E の `data/census/` へ写す。
"""
import collections
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import split_units  # noqa: E402
from layer_role_table import page_order  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
OUT = os.path.join(D, "e_shitaji_sample5_20261126.csv")
NAMES = {"下地 1": 2, "下地 2": 2, "下地 3": 1, "下地 4": 0}


def main() -> int:
    c = db.connect()
    pop = c.execute("select l.page_path, l.layer_id, l.name, p.year from layers l join pages p using (page_path) "
                    "where l.name in ('下地 1','下地 2','下地 3','下地 4')").fetchall()
    print("母集団: " + " / ".join(f"{n} {sum(1 for r in pop if r[2] == n)}" for n in NAMES) + f" = {len(pop)}")
    pages = sorted({r[0] for r in pop})
    q = ",".join("?" * len(pages))
    nlay = dict(c.execute(f"select page_path, count(*) from layers where page_path in ({q}) group by 1", pages).fetchall())
    un = split_units.units(c)
    unit = lambda p: un[wd(p)].split("\\", 2)[-1]  # noqa: E731
    order = list(sorted(order_years := {r[3] for r in pop}))  # noqa: F841
    pick, used = [], set()
    for n, k in NAMES.items():
        rs = [r for r in pop if r[2] == n]
        if k == 0:
            print(f"  {n}: 外す({len(rs)}層。抜き取りにならない)"); continue
        ys = sorted({r[3] for r in rs})
        cells = [ys[0], ys[-1]] if k == 2 else [None]
        if k == 2 and len(ys) < 2:
            print(f"  {n}: 札が1つしかない({ys})"); cells = [ys[0], ys[0]]
        for y in cells:
            cand = sorted([r for r in rs if y is None or r[3] == y], key=lambda r: (nlay[r[0]], r[0], r[1]))
            m = (len(cand) - 1) // 2
            near = sorted(range(len(cand)), key=lambda i: (abs(i - m), i))
            got = None
            for i in near:
                if unit(cand[i][0]) not in used and (cand[i][0], cand[i][1]) not in {(x[0], x[1]) for x in pick}:
                    got = cand[i]; break
            skip = near.index(cand.index(got)) if got else None
            print(f"  {n} 札 {y or '問わない'}: 層 {len(cand)} / 頁の層数 中位 {nlay[cand[m][0]]}"
                  + (f" → 取った(頁の層数 {nlay[got[0]]}、単位の重なりで飛ばした {skip})" if got else " → **取れない**"))
            if got:
                pick.append(got); used.add(unit(got[0]))
    po = page_order(c, sorted({r[0] for r in pick}))
    clip = {}
    for r in csv.DictReader(open(os.path.join(D, "a_clipping_base_v3_v8_20261123.csv"), encoding="utf-8")):
        clip[(r["page_path"], r["layer_id"])] = r
    keys = {(r[0], str(r[1])) for r in pick}
    wh = {}
    for f in sorted(glob.glob(os.path.join(D, "white_column", "white_column_layers_b*.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if (r["page_path"], r["layer_id"]) in keys:
                wh[(r["page_path"], r["layer_id"])] = r
    out = []
    for p, lid, name, year in pick:
        L = c.execute("select l.parent_id, l.order_in_parent, k.v_layer_kind, l.opacity, l.blend_mode, l.clip_raw, l.has_mask, "
                      "pg.v_canvas_px_w, pg.v_canvas_px_h from layers l join v_layer_kind k using (page_path, layer_id) "
                      "join pages pg using (page_path) where l.page_path = ? and l.layer_id = ?", [p, lid]).fetchone()
        sib = c.execute("select l.order_in_parent, coalesce(l.name,''), k.v_layer_kind from layers l join v_layer_kind k using (page_path, layer_id) "
                        "where l.page_path = ? and l.parent_id is not distinct from ? and l.layer_id <> ? order by l.order_in_parent",
                        [p, L[0], lid]).fetchall()
        above = [s for s in sib if s[0] > L[1]]
        below = [s for s in sib if s[0] < L[1]]
        w = wh.get((p, str(lid)), {})
        cb = clip.get((p, str(lid)), {})
        nd, nw = w.get("n_drawn", ""), w.get("n_white", "")
        cpx = L[7] * L[8] if L[7] and L[8] else None
        out.append(dict(page_path=p, layer_id=lid, layer_name=name, v_layer_kind=L[2], year_label=year, split_unit=unit(p),
                        pos_from_top=po[(p, lid)][0] if (p, lid) in po else "", ancestors=po[(p, lid)][2] if (p, lid) in po else "",
                        sibling_above_name=above[0][1] if above else "", sibling_above_kind=above[0][2] if above else "",
                        sibling_below_name=below[-1][1] if below else "", sibling_below_kind=below[-1][2] if below else "",
                        opacity=L[3], blend_mode=L[4], clip_raw=L[5], has_mask=int(bool(L[6])),
                        a_is_clipping_base=cb.get("a_is_clipping_base", ""), n_clipped_layers=cb.get("n_clipped_layers", ""),
                        n_layers_on_page=nlay[p], white_render_status=w.get("render_status", "(白さの便に無い)"),
                        n_drawn=nd, n_white=nw,
                        white_frac=f"{int(nw) / int(nd):.4f}" if nd and nw and int(nd) else "",
                        drawn_frac=f"{int(nd) / cpx:.4f}" if nd and cpx else "",
                        canvas_px_w=L[7], canvas_px_h=L[8]))
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
    for o in out:
        print(f"  {o['layer_name']} / {o['year_label']} / 頁の層 {o['n_layers_on_page']} / 白 {o['white_frac']} / 描いた割合 {o['drawn_frac']} / "
              f"乗る {o['n_clipped_layers']} / {o['white_render_status']}")
    print(f"出力: {OUT}({len(out)}行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
