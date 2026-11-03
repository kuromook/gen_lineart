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
import collections
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
OUT_WORK = "census/worksheet_folder_reach_by_work.csv"
WORK_IDS = "data/census_full/ink_darkness_work_ids.csv"
# **作者の聞き取り**(2026-11-03 22:37 に記入済み。原稿の名前を含むので `data/` 配下)
HEARING = "data/census_full/hearing_worksheet_folder_20261103.xlsx"
# **中の層を外す答え**(「仕上げの線」は数に入れる。「混在」「分からない」は外さず、数を印字する)
HEARING_EXCLUDE = {"下絵・ラフ", "資料(他の絵を貼った)", "塗りつぶし"}


def load_hearing():
    """**頁ごとの作者の答え** —— `{page_path: answer}`。無ければ空(印字する)。"""
    path = os.path.join(ROOT, HEARING)
    if not os.path.exists(path):
        print("    **作者の聞き取りの表が無い**")
        return {}
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["頁"]
    rows = list(ws.iter_rows(values_only=True))
    head = list(rows[0])
    ip, ia = head.index("頁(ファイルの場所)"), head.index("この頁の「sketch set」の中にある線画の層は何ですか")
    return {r[ip]: (r[ia] or "") for r in rows[1:] if r[ip]}
FOLDER_NAME = "sketch set"
# **抽出側が便に書いた全数**(`push_sketch_set_author_answer_and_picks_20261103.md` §3)
SUPPLIER = {"n_pages": 7013, "n_layers": 46327, "n_leaf": 32963}
MAIN_VALUE = "0.9177"
LABEL_AUTH = "作者が外すと答えた層だけを外した値(11-03 22:37 の聞き取り)"


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
    hearing = load_hearing()
    auth = [r for r in main_pop if not ((r["page_path"], int(r["layer_id"])) in keys
                                       and hearing.get(r["page_path"]) in HEARING_EXCLUDE)]
    for label, rows in (("墨の濃さの主結論の母集団(11-02 までの主値)", main_pop),
                        ("うち下絵の層のフォルダーの中", ins),
                        ("うち外(作業用の層を外した値、11-03)", outs),
                        (LABEL_AUTH, auth)):
        v, ink = frac(rows)
        print(f"    {label}: 層 {len(rows):,} / 頁 {len({r['page_path'] for r in rows}):,} / "
              f"墨画素 {ink:,} / 墨のうち純黒 {v}")
        out.append([label, len({r["page_path"] for r in rows}), len(rows), "", ink, v, "", ""])
    if out[2][5] != MAIN_VALUE:
        print(f"    **主値が {MAIN_VALUE} を再現しない**({out[2][5]})")
        bad += 1
    bad += by_work(keys)
    out += [[""] * 7 + [n] for n in NOTE]
    n = bc.write_csv(os.path.join(ROOT, OUT),
                     ["worksheet_subset", "n_pages", "n_layers", "n_leaf", "n_px_ink",
                      "frac_pure_black_among_ink", "known_cell", "note"], out)
    print(f"    {os.path.basename(OUT)} {n}行")
    return 1 if bad else 0


def by_work(keys) -> int:
    """**作品フォルダーごとの行**(2026-11-03 20:30、ユーザーの依頼)。**登録上の主値は作品ごとの行**なので、
    畳んだ1行の前後だけでは足りない。**1段目と2段目の両方**、層別 L1 形式2。"""
    print("  [作品ごとの前後]")
    hearing = load_hearing()
    answers = collections.Counter(hearing.values())
    print(f"    **作者の聞き取り** {len(hearing)}頁: " + " / ".join(f"{k or '空'} {v}" for k, v in answers.most_common()))
    # **規律13**: フォルダーを「仕上げでない」の代わりに使えるか。**聞いた頁で的中率を印字する**
    hit = sum(1 for a in hearing.values() if a in HEARING_EXCLUDE)
    print(f"    **フォルダーの中 = 仕上げでない、の的中**: {hit}/{len(hearing)}頁(作者に聞いた頁だけ。聞いていない頁には届かない)")
    ids = {r["work_dir"]: r["work_id"] for r in
           csv.DictReader(open(os.path.join(ROOT, WORK_IDS), encoding="utf-8"))}
    rows = [dict(r, ink_stage="1") for r in
            csv.DictReader(open(os.path.join(ROOT, ik.STAGE1), encoding="utf-8"))]
    for p in sorted(glob.glob(os.path.join(ROOT, ik.STAGE2_GLOB))):
        rep = ik.STAGE2_REPLACED.get(os.path.basename(p))
        if rep:
            p = os.path.join(os.path.dirname(p), rep)
        rows += [dict(r, ink_stage="2") for r in csv.DictReader(open(p, encoding="utf-8"))]
    agg = {}
    for r in rows:
        if r["stratum"] != "L1" or r["pixel_formats"] != "2":
            continue
        k = (r["ink_stage"], r["year_label"], ids.get(r["work_dir"], "**名札なし**"))
        a = agg.setdefault(k, collections.Counter())
        ins = (r["page_path"], int(r["layer_id"])) in keys
        ink, pb = int(r["n_px_ink"] or 0), int(r["n_px_pure_black"] or 0)
        a["n"] += 1
        a["ink"] += ink
        a["pb"] += pb
        if ins:
            a["n_in"] += 1
            a["ink_in"] += ink
            a["pb_in"] += pb
            ans = hearing.get(r["page_path"])
            if ans is not None:
                a["n_in_checked"] += 1
            if ans in HEARING_EXCLUDE:
                a["n_in_author_excluded"] += 1
                a["ink_x"] += ink
                a["pb_x"] += pb
    lo, hi = ik.BAND_PURE_BLACK

    def f(pb, ink):
        return f"{pb / ink:.4f}" if ink else ""

    def band(v, stage):
        if not v or stage != "2":
            return ""
        return "below" if float(v) < lo else "above" if float(v) > hi else "inside"
    out = []
    tot = collections.Counter()
    moved = collections.Counter()
    for k in sorted(agg, key=lambda k: (k[0], k[1], int(k[2][1:]) if k[2][1:].isdigit() else 0)):
        a = agg[k]
        if not a["ink"]:
            moved["墨の画素が 0 の作品(値が無い)"] += 1
            continue
        v_all = f(a["pb"], a["ink"])
        v_out = f(a["pb"] - a["pb_in"], a["ink"] - a["ink_in"])
        v_in = f(a["pb_in"], a["ink_in"])
        b_all, b_out = band(v_all, k[0]), band(v_out, k[0])
        if a["n_in"] == 0:
            moved["中の層が無い(前後が同じ)"] += 1
        elif not v_out:
            moved["**全部の層が中に在る(外すと作品ごと消える)**"] += 1
        elif abs(float(v_out) - float(v_all)) >= 0.01:
            moved["0.01 以上動く"] += 1
        else:
            moved["動きが 0.01 未満"] += 1
        if k[0] == "2" and b_all != b_out:
            moved[f"**帯の読みが変わる**({b_all} → {b_out or '値なし'})"] += 1
        v_auth = f(a["pb"] - a["pb_x"], a["ink"] - a["ink_x"])
        tot["ink"] += a["ink"]
        tot["pb"] += a["pb"]
        tot["ink_x"] += a["ink_x"]
        tot["pb_x"] += a["pb_x"]
        out.append([k[2], k[1], k[0], a["n"], a["n_in"], a["n_in_checked"], a["n_in_author_excluded"],
                    f"{a['ink_in'] / a['ink']:.3f}", v_all, v_out, v_in, v_auth, b_all, b_out,
                    band(v_auth, k[0]), ""])
    for why, n in sorted(moved.items()):
        print(f"    作品 {n:>3} —— {why}")
    print(f"    **作者が外すと答えた層を外した値(畳んだ行、1段目と2段目)**: "
          f"{f(tot['pb'], tot['ink'])} → {f(tot['pb'] - tot['pb_x'], tot['ink'] - tot['ink_x'])}")
    changed = [r for r in out if r[11] != r[8]]
    for r in changed:
        print(f"      作者の答えで動く作品: {r[0]} 札 {r[1]}  {r[8]} → {r[11]}(外した層 {r[6]})")
    out += [[""] * 15 + [n] for n in NOTE_WORK]
    n = bc.write_csv(os.path.join(ROOT, OUT_WORK),
                     ["work_id", "year_label", "ink_stage", "n_layers", "n_layers_inside",
                      "n_layers_inside_author_checked", "n_layers_author_excluded",
                      "share_ink_inside", "frac_pure_black_all", "frac_pure_black_outside",
                      "frac_pure_black_inside", "frac_pure_black_author_checked",
                      "band_all", "band_outside", "band_author_checked", "note"], out)
    print(f"    {os.path.basename(OUT_WORK)} {n}行")
    return 0


NOTE_WORK = [
    "**【11-03 22:37 作者の聞き取り】中を外す値(`_outside`)は、フォルダーを「仕上げでない」の代わりに使った値で、"
    "その代用は作者の分類を再現しない**(規律13)—— **大きく動く・消える16作品で、中に線画の層が在る 129頁を作者が見た: "
    "127頁は仕上げの線、2頁は塗りつぶし**(同じ頁の双子。頁全体の塗りつぶしで、用途は不明)。"
    "**作者: 「ほとんどは完成イラストで、フォルダーの運用ルールは守られず、中で完成まで進めた」。** "
    "**`_outside` は撤回しないが、外す規則としては使えない**(前後を名前付きで残す。規律10)",
    "**`frac_pure_black_author_checked` は、作者が「下絵・ラフ / 資料 / 塗りつぶし」と答えた頁の中の層だけを外した値。** "
    "**聞いたのは16作品の129頁だけ** —— **他の作品の中の層は聞いていないので外していない**(`n_layers_inside_author_checked` で数が分かる)。"
    "**作者が見たのは線画の層が何かだけ** —— 中に資料の画像が在る頁もあったが、線画の層以外は見ていない(作者)",
    "**作品フォルダーごとの前後**(2026-11-03)。**登録上の主値は作品ごとの行**(`doc/prereg_ink_darkness_20261030.md` §2)"
    "なので、畳んだ1行(0.9177 → 0.9162)の前後だけでは足りない。**層別 L1 形式2、1段目と2段目の両方。** "
    "**`frac_pure_black_all` は公開中の `ink_darkness_stage*.csv` の値と同じ**(撤回しない)。**`_outside` は中の層を外した値、`_inside` は中の層だけ**",
    "**`band_*` は事前登録の帯(1段目の形式2 の行で出た 0.5262〜0.9712)に対する位置。2段目の行だけ**"
    "(1段目の行は帯を作った側なので空)",
    "**外すと作品ごと消える行が在る**(`frac_pure_black_outside` が空)—— **その作品では名前で選んだ線画の層が全部そのフォルダーの中に在る。** "
    "**中の層が仕上げの線か下絵・資料かは、この表では届かない**(名前では分けきれない。**絵が何を描いているかは作者が決める**。規律6)",
    "**大きく動く行は、外に残った少数の層が値を決めている**(例: 外に残った層の墨の大半が純黒でない背景の層)。"
    "**動いた向きを良し悪しとして読まない**",
    "**work_id は `ink_darkness_stage*.csv` と同じ匿名の名札**(条例10)。**同じ数の2行は双子(札 2019 と 2019b)。足さない**",
]


NOTE = [
    "**【11-03 22:37 作者の聞き取り】フォルダーの中の層は「仕上げでない」とは言えなかった** —— "
    "**中を外すと値が大きく動く・消える16作品で、中に線画の層が在る 129頁を作者が見た: 127頁は仕上げの線、2頁は塗りつぶし。** "
    "**作者: ほとんどは完成イラストで、フォルダーの運用ルールは守られず、中で完成まで進めた。** "
    "**だから「うち外」の行(0.9162)は外す規則として使えない**(撤回はしない。規律10)。"
    "**作者が外すと答えた層だけを外すと 0.9156** —— **外したのは2層(頁全体の塗りつぶし、名前は線画)で、墨画素の 2.4%**。"
    "**聞いていない頁(他の作品の中の層)は外していない**",
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
