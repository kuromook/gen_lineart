#!/usr/bin/env python
"""**頁の層の行**(`m_colour_page_layers_v2_20261102.csv` と同じ64列)を構造 DB から作る —— Track M の発注。

**発注**: `note_m_order_layer_rows_holdout8_20261107.md`(2026-11-07)。8頁は M が census の材料表
`mask_material_pages` から seed 0 で引いた標本外の頁で、**作者が点検して「対象外にしたいページはない」**。

**census が作れるのは構造の列だけ**(条例10: 原稿の画素を読まない)。
**画素から測った列**(`rendered` / `drawn_ratio` / `black_among_drawn` / `drawn_gray_mean` /
`preview_dark` / `formats` / `draw_color_*` / `sat_*` / `colours_q16` / `top_colour_*`)は**空で出す。**
**画素は M が抽出側に発注済み**(`push_m_order_holdout8_pages_20261107.md`、置き場 `holdout8_20261107/`)。
構造 DB の `layer_colour` には8頁のうち2頁の行が在るが、**10-26 の断面で zlib・1ビットの直しの前の値**
なので**混ぜない**(版を1つの表に混ぜない)。

**既知セル**(この器具が作っていない入力): **M の手元の `m_colour_page_layers_v2_20261102.csv`
(抽出側が 11-02 に作った 872頁・72,839行)** —— **同じ規則でその872頁を作り、構造の列を全行突き合わせる。**
一致しなければ8頁を出さない。

    ./venv/bin/python tools/census/m_layer_rows.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWN = "../lineart-colour-manuscripts/inbox/m_colour_page_layers_v2_20261102.csv"
OUT = "data/census_full/m_holdout8_page_layers_20261107.csv"
PAGES = [r"F:\manuscripts\2016\gal\gal\page0014.clip",
         r"F:\manuscripts\2016\gal\page0010.clip",
         r"F:\manuscripts\2016\gran\gran\page0010.clip",
         r"F:\manuscripts\2016\phantom\PhantomThief\page0017.clip",
         r"F:\manuscripts\2016\suc\subbcubussBK\page0002.clip",
         r"F:\manuscripts\2017\illustration\illustBK\page0009.clip",
         r"F:\manuscripts\2017\maruchan\maru2\page0016.clip",
         r"F:\manuscripts\2024\housei\kazenagare\page0008.clip"]
# **構造の列**(DB から作る)。それ以外の列は画素から測った列で、空で出す
STRUCT = ["page_id", "page_path", "layer_id", "name", "ancestors", "folder", "layer_type", "visibility",
          "opacity", "has_mask", "has_gradation_fill", "tone_enabled", "tone_density_pct", "tone_lpi",
          "canvas", "composite", "clip", "parent_id", "order_in_parent"]


def fmt_num(v):
    if v is None:
        return ""
    return str(int(v)) if float(v).is_integer() else str(v)


def page_rows(con, p):
    pid = con.execute("select pid_000 from page_ids where page_path = ?", [p]).fetchone()[0]
    cw, ch = con.execute("select v_canvas_px_w, v_canvas_px_h from pages where page_path = ?", [p]).fetchone()
    ls = con.execute("""select layer_id, parent_id, coalesce(name, ''), folder, layer_type, visibility, opacity,
                               has_mask, has_gradation, tone_enabled, tone_density_pct, tone_lpi, blend_mode,
                               clip_raw, order_in_parent
                          from layers where page_path = ?""", [p]).fetchall()
    by = {r[0]: r for r in ls}
    out = {}
    for r in ls:
        chain, x = [], r[1]
        while x is not None and by[x][1] is not None:      # **根(親の無い層)は名前に入れない**
            chain.append(by[x][2])
            x = by[x][1]
        out[r[0]] = {"page_id": pid, "page_path": p, "layer_id": str(r[0]), "name": r[2],
                     "ancestors": "/" + "/".join(reversed(chain)) if chain else "",
                     "folder": str(r[3]), "layer_type": str(r[4]), "visibility": str(r[5]),
                     "opacity": str(r[6]), "has_mask": str(r[7]), "has_gradation_fill": str(r[8]),
                     "tone_enabled": str(r[9]), "tone_density_pct": fmt_num(r[10]),
                     "tone_lpi": fmt_num(r[11]), "canvas": f"[{cw}, {ch}]", "composite": str(r[12]),
                     "clip": str(r[13]), "parent_id": "" if r[1] is None else str(r[1]),
                     "order_in_parent": "" if r[14] is None else str(r[14])}
    return out


def main() -> int:
    print("[頁の層の行(Track M の発注、8頁)]")
    con = db.connect()
    known = list(csv.DictReader(open(os.path.join(ROOT, KNOWN), encoding="utf-8")))
    header = list(known[0].keys())
    cache, bad, n = {}, {}, 0
    for r in known:
        p = r["page_path"]
        if p not in cache:
            cache = {p: page_rows(con, p)}
        mine = cache[p].get(int(r["layer_id"]))
        n += 1
        for c in STRUCT:
            a = r[c]
            b = "" if mine is None else mine[c]
            if c in ("tone_density_pct", "tone_lpi") and a and b:
                a, b = float(a), float(b)
            if a != b:
                bad[c] = bad.get(c, 0) + 1
    print(f"    **既知セル**: 抽出側の 872頁・{n:,}行と構造の {len(STRUCT)}列を突き合わせた —— "
          + ("**全行一致**" if not bad else f"**食い違い {bad}**"))
    if bad:
        print("    **8頁は出さない**")
        return 1
    rows, missing = [], []
    for p in PAGES:
        pr = page_rows(con, p)
        if not pr:
            missing.append(p)
        for lid in sorted(pr):
            rows.append([pr[lid].get(c, "") for c in header])
        print(f"    {p.split(chr(92))[2]} … {len(pr)}層")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"    8頁 {len(rows)}行 → `{OUT}`(列 {len(header)}、うち構造 {len(STRUCT)}、画素の {len(header) - len(STRUCT)}列は空)"
          + (f" / **層の無い頁 {missing}**" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
