#!/usr/bin/env python
"""**下絵の層のフォルダー(雛形の名前 `sketch set`)の中の層が、census の数にどれだけ入っているか**(2026-11-03)。

**作者の答え**(2026-11-03、抽出側経由): その名前は**層のフォルダー**で、中に置くのは**下絵・ラフと資料**、
ほかに**下絵より下の層を隠す塗りつぶし**。**作者の決定**: 中の層は**遮蔽物ではない** —— **仕上げのものではない。**

**この器具が答える問い**: **墨の濃さの主結論の母集団**(2段目 層別 L1 形式2、40,323層)に、
その中の層が何層入っているか / 外すと主値がどう動くか。

**前後は名前を付けて両方印字する**(規律10)。**主値 0.9177 は撤回しない** —— **作者の裁定は点数と独立なので棒ではない**
が、**公開の主値を替えるかは、この器具は決めない**(登録の持ち主に上げる)。

**既知セル**(T132): **抽出側が便に書いた全数**(フォルダーの在る頁 / 中の層 / うちフォルダーでない層)——
**この器具が作っていない入力。** **供給元は「除外の1頁を除いた」と書いたが、数は除外の頁を含めた数と一致する** —— 両方の行を出す。

**届く範囲**(名前より狭く書く): **祖先の鎖に、その名前の層のフォルダーが在るか**まで。
**中の層が下絵か資料か塗りつぶしかは分けない**(名前では分けきれない。抽出側)。**画素は読まない。**

    ./venv/bin/python tools/census/worksheet_folder_reach.py
"""
import csv
import glob
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import exclusions  # noqa: E402
import ink_darkness as ik  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = "/home/sh1/deepl/census_db/manuscripts.duckdb"
OUT = "census/worksheet_folder_reach.csv"
FOLDER_NAME = "sketch set"
# **抽出側が便に書いた全数**(`push_sketch_set_author_answer_and_picks_20261103.md` §3)
SUPPLIER = {"n_pages": 7013, "n_layers": 46327, "n_leaf": 32963}
MAIN_VALUE = "0.9177"


def inside_keys(con):
    return con.execute("""
        with recursive anc as (
          select page_path, layer_id, folder, parent_id as a from layers
          union all
          select anc.page_path, anc.layer_id, anc.folder, l.parent_id
          from anc join layers l on l.page_path = anc.page_path and l.layer_id = anc.a)
        select distinct anc.page_path, anc.layer_id, anc.folder
        from anc join layers f on f.page_path = anc.page_path and f.layer_id = anc.a
        where f.folder and f.name = ?""", [FOLDER_NAME]).fetchall()


def frac(rows):
    ink = sum(int(r["n_px_ink"] or 0) for r in rows)
    pb = sum(int(r["n_px_pure_black"] or 0) for r in rows)
    return (f"{pb / ink:.4f}" if ink else ""), ink


def main() -> int:
    print("  [下絵の層のフォルダーの中の層が census の数に入っている量]")
    excluded = exclusions.load()
    con = duckdb.connect(DB, read_only=True)
    folder_pages = {p for (p,) in con.execute(
        "select distinct page_path from layers where folder and name = ?", [FOLDER_NAME]).fetchall()}
    inside = inside_keys(con)
    bad = 0
    out = []
    for label, keep in (("全数(除外の頁を含む)", lambda p: True),
                        ("全数(除外の頁を除く)", lambda p: p not in excluded)):
        rows = [x for x in inside if keep(x[0])]
        got = {"n_pages": sum(1 for p in folder_pages if keep(p)), "n_layers": len(rows),
               "n_leaf": sum(1 for x in rows if not x[2])}
        same = all(got[k] == SUPPLIER[k] for k in SUPPLIER)
        print(f"    {label}: フォルダーの在る頁 {got['n_pages']:,} / 中の層 {got['n_layers']:,} / "
              f"うち葉 {got['n_leaf']:,}  —— 供給元の数と {'一致' if same else '不一致'}")
        out.append([label, got["n_pages"], got["n_layers"], got["n_leaf"], "", "",
                    "一致" if same else "—", ""])
    if not any(r[6] == "一致" for r in out):
        print("    **供給元の数とどちらの行も一致しない**")
        bad += 1

    keys = {(p, int(i)) for p, i, _ in inside}
    st2 = []
    for p in sorted(glob.glob(os.path.join(ROOT, ik.STAGE2_GLOB))):
        rep = ik.STAGE2_REPLACED.get(os.path.basename(p))
        if rep:
            p = os.path.join(os.path.dirname(p), rep)
        st2 += list(csv.DictReader(open(p, encoding="utf-8")))
    main_pop = [r for r in st2 if r["stratum"] == "L1" and r["pixel_formats"] == "2"]
    ins = [r for r in main_pop if (r["page_path"], int(r["layer_id"])) in keys]
    outs = [r for r in main_pop if (r["page_path"], int(r["layer_id"])) not in keys]
    for label, rows in (("墨の濃さの主結論の母集団(11-02 までの主値)", main_pop),
                        ("うち下絵の層のフォルダーの中", ins),
                        ("うち外(作業用の層を外した値、11-03)", outs)):
        v, ink = frac(rows)
        print(f"    {label}: 層 {len(rows):,} / 頁 {len({r['page_path'] for r in rows}):,} / "
              f"墨画素 {ink:,} / 墨のうち純黒 {v}")
        out.append([label, len({r["page_path"] for r in rows}), len(rows), "", ink, v, "", ""])
    if out[2][5] != MAIN_VALUE:
        print(f"    **主値が {MAIN_VALUE} を再現しない**({out[2][5]})")
        bad += 1
    out += [[""] * 7 + [n] for n in NOTE]
    n = bc.write_csv(os.path.join(ROOT, OUT),
                     ["worksheet_subset", "n_pages", "n_layers", "n_leaf", "n_px_ink",
                      "frac_pure_black_among_ink", "known_cell", "note"], out)
    print(f"    {os.path.basename(OUT)} {n}行")
    return 1 if bad else 0


NOTE = [
    "**作者の答え(2026-11-03、抽出側経由)**: 下絵を意味する雛形の名前は**層のフォルダー**で、"
    "中は**下絵・ラフと資料**、ほかに**下絵より下の層を隠す塗りつぶし**。**作者の決定: 中の層は遮蔽物ではない**"
    "(仕上げのものではない)。**名前は作品フォルダーの名前と同じ綴りを含むので、公開物では書かない**",
    "**届く範囲**: **祖先の鎖にその層のフォルダーが在るか**まで。**中の層が下絵か資料か塗りつぶしかは分けない** "
    "(抽出側: 名前では分けきれない)。**画素は読まない**",
    "**墨の濃さの主値 0.9177 は撤回しない**(規律10)。**外した値は名前を付けて隣に置く** —— "
    "**主値をどちらにするかはこの器具は決めない。** **作者の裁定は点数と独立なので棒ではないが、"
    "母集団を動かすのは棒の一部である**(結果を見る前に決めた裁定であることを記録する)",
    "**全数の2行**: **供給元は「除外の1頁を除いた」と書いたが、数は除外の頁を含めた行と一致する。** "
    "**除外の頁を除いた行も出す**(どちらかを消さない)",
    "**`n_pages` は行ごとに述語が違う** —— 全数の2行は「その層のフォルダーが在る頁」、"
    "墨の濃さの3行は「その層が在る頁」。**足さない**",
]

if __name__ == "__main__":
    raise SystemExit(main())
