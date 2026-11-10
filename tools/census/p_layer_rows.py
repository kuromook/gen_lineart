#!/usr/bin/env python
"""**300頁の層の行** —— Track P の依頼(`note_p_order_layer_rows_300_20261107.md`、2026-11-07 夜)。

**列は holdout8 と同じ64列**(`m_layer_rows.py` と同じ規則・同じ既知セル)に、**後ろへ列を足す**:

- **マスクの画素の数**(census が測ったのではない。**供給元の表の値をそのまま写す**):
  - `mask_face_*` —— 面の表 `census_face_layer_alpha_mask_20261104.csv` の `plane = mask` の行
    (**キャンバス全体**で数えた値。区切りは `>0` / `=255` / `1〜254`)
  - `mask_cover_px_ge128` / `mask_cover_px_1_127` —— `mask_cover_request3_v2_20261102.csv`
    (**マスクのバッファの範囲**で数えた値。区切りは `>=128` / `1〜127`)
  - **2つの区切りは違う。片方から片方を作らない**(規律13: 近い列で代用しない)。
    P が名指しした `mask_px_ge128` / `mask_px_1_127` は後者の区切りで、**300頁のうち引ける頁は少ない**
- `colour_class` —— 作者の決め方の第1段(`colour_class.py`)。**聞き取り待ちはカラーと読まない**
- `wip_status` —— `wip_pages_20261106.csv`

**画素から測った列**(条例10: census は画素を読まない)は、**抽出側の manifest の値をそのまま写す**
(2026-11-10。納品 `deliver_p_order300_part1/2_20261109.md`、置き場 `../lineart/data/incoming_manuscript/face_merge_300_20261107/bNN/`、
**写す前に manifest の sha256 を便ごとに照らす**):
- `colour_layers_manifest_bNN.json` → `rendered` / `drawn_ratio` / `formats` / `draw_color_*` / `sat_mean` /
  `colours_q16` / `top_colour_*` / `sat_*`(32箱)。**書かなかった層は `rendered = False`、理由を `unrendered_reason` に**
- **書き方は M の 872頁の表に合わせる**: **`rendered` は「読めた」の意味**(M の表で空の層 14,781 は
  `True`・`drawn_ratio` 0.0)。**「1画素も描かれていない」層は `rendered = True`** で `drawn_ratio` / `colours_q16` / `sat_*` を 0、
  フォルダーと「描画できなかった」層は `False`(数は空)。描いていない層の描画色は `[None, None, None]`
  (**manifest の書かなかった層には描画色・`formats` が無い** —— M の表には在るので、既知セルで違いとして数える)
- **`black_among_drawn` / `drawn_gray_mean` / `preview_dark` は manifest に無い**(層の PNG とプレビューから測る量)。**空のまま**
- `layer_masks_manifest_bNN.json` → `mask_px_ge128` / `mask_px_1_127` / `mask_init_color` / `mask_error`
  (**キャンバス全体で数えた値**。`mask_cover_*` はバッファの範囲で数えた値なので別の列に置く。規律13)
- **既知セル**: 300頁のうち M の 872頁の表(抽出側 11-02、**読み手の直しの前の版**)と重なる頁で、
  画素の列を突き合わせ、列ごとの一致数と不一致を印字する(版が違うので**一致を門にはしない**。
  不一致は版の差として数える)

    ./venv/bin/python tools/census/p_layer_rows.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import m_layer_rows as ml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORDER = "inbox/p_order_300_pages_20261107.csv"
FACE = "inbox/census_face_layer_alpha_mask_20261104.csv"
COVER = "inbox/mask_cover_request3_v2_20261102.csv"
COLOUR = "data/census_full/colour_class_20261107.csv"
WIP = "data/census_full/wip_pages_20261106.csv"
OUT = "data/census_full/p_order300_page_layers_20261110.csv"
MANI = "../lineart/data/incoming_manuscript/face_merge_300_20261107"
KNOWN = "../lineart-colour-manuscripts/inbox/m_colour_page_layers_v2_20261102.csv"
PIXC = ["rendered", "drawn_ratio", "formats", "draw_color_enable", "draw_color_main", "draw_color_sub", "sat_mean",
        "colours_q16", "top_colour_share", "top_colour_q16"]
EXTRA = ["mask_face_n_px_gt0", "mask_face_n_px_255", "mask_face_n_px_1_254", "mask_face_n_px_total",
         "mask_face_status", "mask_cover_px_ge128", "mask_cover_px_1_127", "colour_class", "wip_status",
         "unrendered_reason", "mask_px_ge128", "mask_px_1_127", "mask_init_color", "mask_error"]


def py(v):
    """M の 872頁の表と同じ書き方(Python の str。None は空、リストは中の None を残す)"""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    return str(v)


def manifests():
    """便ごとに sha256 を照らしてから読む。返す: (頁, 層) → 画素の列 / 書かなかった理由 / マスクの値"""
    import glob
    import hashlib
    import json
    pix, skip, mask = {}, {}, {}
    for d in sorted(glob.glob(os.path.join(ROOT, MANI, "b*"))):
        b = os.path.basename(d)
        want = {}
        for ln in open(os.path.join(d, f"face_merge_300_{b}.sha256"), encoding="utf-8"):
            h, f = ln.split(maxsplit=1)
            want[f.strip().lstrip("*")] = h
        for f in (f"colour_layers_manifest_{b}.json", f"layer_masks_manifest_{b}.json"):
            got = hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
            if want.get(f) != got:
                raise SystemExit(f"    **sha256 が合わない: {b}/{f}**")
        cm = json.load(open(os.path.join(d, f"colour_layers_manifest_{b}.json"), encoding="utf-8"))
        for pg in cm["pages"]:
            p = pg["source"]
            for L in pg["layers"]:
                v = {k: py(L[k]) for k in PIXC if k != "rendered"}
                v["formats"] = ";".join(L["formats"])
                v["rendered"] = "True"
                for i, n in enumerate(L["sat_hist"]):
                    v[f"sat_{i * 8:03d}_{i * 8 + 7:03d}"] = str(n)
                pix[(p, L["layer_id"])] = v
            for x in pg["skipped"]:
                skip[(p, x["layer_id"])] = x["reason"]
        mm = json.load(open(os.path.join(d, f"layer_masks_manifest_{b}.json"), encoding="utf-8"))
        for L in mm["layers"]:
            mask[(L["page_path"], L["layer_id"])] = L
    return pix, skip, mask


def main() -> int:
    print("[300頁の層の行(Track P の依頼)]")
    con = db.connect()
    header, n, bad = ml.known_check(con)
    print(f"    **既知セル**: 抽出側の 872頁・{n:,}行と構造の {len(ml.STRUCT)}列を突き合わせた —— "
          + ("**全行一致**" if not bad else f"**食い違い {bad}**"))
    if bad:
        print("    **300頁は出さない**")
        return 1
    order = list(csv.DictReader(open(os.path.join(ROOT, ORDER), encoding="utf-8-sig")))
    pages = [r["page_path"] for r in order]
    excl = exclusions.load()
    wip = {r["page_path"]: r["wip_status"] for r in csv.DictReader(open(os.path.join(ROOT, WIP), encoding="utf-8"))}
    col = {r["page_path"]: r["colour_class"] for r in csv.DictReader(open(os.path.join(ROOT, COLOUR), encoding="utf-8"))}
    pset = set(pages)
    face, cover = {}, {}
    for r in csv.DictReader(open(os.path.join(ROOT, FACE), encoding="utf-8")):
        if r["plane"] == "mask" and r["page_path"] in pset:
            face[(r["page_path"], int(r["layer_id"]))] = r
    for r in csv.DictReader(open(os.path.join(ROOT, COVER), encoding="utf-8")):
        if r["page_path"] in pset:
            cover[(r["page_path"], int(r["layer_id"]))] = r
    print(f"    頁 {len(pages)}(重複 {len(pages) - len(pset)})/ 恒久除外の頁 {sum(p in excl for p in pages)}"
          f" / 作業途中 {sum(wip.get(p) == '作業途中' for p in pages)}")
    if any(p in excl for p in pages):
        print("    **恒久除外の頁が入っている。出さない**")
        return 1
    pix, skip, mask = manifests()
    print(f"    manifest(sha256 を10便とも照らした): 書いた層 {len(pix):,} / 書かなかった層 {len(skip):,}"
          f"(" + " / ".join(f"{k} {v}" for k, v in collections.Counter(skip.values()).items()) + f")/ マスクの行 {len(mask):,}")
    rows, missing, c = [], [], collections.Counter()
    under = 0
    for o in order:
        p = o["page_path"]
        pr = ml.page_rows(con, p)
        if not pr:
            missing.append(p)
            continue
        n_mask = 0
        for lid in sorted(pr):
            x = pr[lid]
            f, v = face.get((p, lid)), cover.get((p, lid))
            hm = x["has_mask"] == "True"
            n_mask += hm
            px, sk, mk = pix.get((p, lid)), skip.get((p, lid)), mask.get((p, lid))
            nodraw = dict(rendered="False", draw_color_main="[None, None, None]", draw_color_sub="[None, None, None]")
            zero = dict(drawn_ratio="0.0", colours_q16="0", **{h: "0" for h in header if h.startswith("sat_") and h != "sat_mean"})
            if x["folder"] == "True":
                c["フォルダー(画素の列は空)"] += 1
                x = dict(x, **nodraw)
                if px or sk:
                    c["  **フォルダーなのに画素の行がある**"] += 1
            elif px:
                c["葉: 書いた"] += 1
                x = dict(x, **px)
            elif sk:
                c[f"葉: 書かなかった({sk})"] += 1
                # 1画素も無い層は数が 0 と言える。描画できなかった層は言えないので空
                # (M の表の `rendered` は「読めた」の意味: 空の層も True で drawn_ratio 0.0)
                x = dict(x, **(dict(nodraw, rendered="True", **zero) if sk == "1画素も描かれていない" else nodraw))
            else:
                c["  **葉なのに manifest に行が無い**"] += 1
            c["has_mask でマスクの行が無い" if hm and not mk else "マスクの行が has_mask でない層にある" if mk and not hm else "-"] += 1
            if hm:
                c["has_mask の層"] += 1
                c[f"  面の表の行あり(status {f['status']})" if f else "  面の表の行なし"] += 1
                c["  mask_cover の行あり" if v else "  mask_cover の行なし"] += 1
            rows.append([x.get(h, "") for h in header] + [
                f["n_px_gt0"] if f else "", f["n_px_255"] if f else "", f["n_px_1_254"] if f else "",
                f["n_px_total"] if f else "", f["status"] if f else "",
                v["mask_px_ge128"] if v else "", v["mask_px_1_127"] if v else "",
                col.get(p, ""), wip.get(p, ""), sk or "",
                py(mk.get("mask_px_ge128")) if mk else "", py(mk.get("mask_px_1_127")) if mk else "",
                py(mk.get("init_color")) if mk else "", py(mk.get("error")) if mk else ""])
        under += n_mask < int(o["census_n_material"])
    del c["-"]
    print(f"    **層 {len(rows):,}行**(頁 {len(pages) - len(missing)})")
    for k, v in c.items():
        print(f"      {k} {v:,}")
    print(f"    **既知セル(依頼の材料の数)**: has_mask の層の数が材料の数より少ない頁 {under}(0 であるはず)")
    cc = collections.Counter(col.get(p, "表に無い") for p in pages)
    print("    カラーかモノクロか(第1段): " + " / ".join(f"{k} {v}" for k, v in cc.items()))
    print("    面の表で引ける頁 " + str(len({p for p, _ in face}))
          + " / mask_cover で引ける頁 " + str(len({p for p, _ in cover})))
    bad_known = known_pixels(header, rows)
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header + EXTRA)
        w.writerows(rows)
    print(f"    → `{OUT}`(非公開。列 {len(header) + len(EXTRA)}。`black_among_drawn` / `drawn_gray_mean` / `preview_dark` は空)"
          + (f" / **層の無い頁 {len(missing)}**" if missing else ""))
    bad = c["  **フォルダーなのに画素の行がある**"] + c["  **葉なのに manifest に行が無い**"] + c["has_mask でマスクの行が無い"]
    return 1 if (missing or under or bad or bad_known) else 0


def known_pixels(header, rows):
    """既知セル: M の 872頁の表(抽出側 11-02、旧い読み手)と重なる頁で画素の列を比べる。門にしない(版が違う)"""
    known = {(r["page_path"], r["layer_id"]): r for r in csv.DictReader(open(os.path.join(ROOT, KNOWN), encoding="utf-8"))}
    ix = {h: i for i, h in enumerate(header)}
    cols = PIXC + [h for h in header if h.startswith("sat_") and h != "sat_mean"]
    both = [(r, known[(r[ix["page_path"]], r[ix["layer_id"]])]) for r in rows
            if (r[ix["page_path"]], r[ix["layer_id"]]) in known]
    pages = {r[ix["page_path"]] for r, _ in both}
    print(f"    **既知セル(M の 872頁の表、旧い読み手)**: 重なる頁 {len(pages)} / 層 {len(both):,}")
    if not both:
        return 1
    diff = collections.Counter()
    ex = {}
    for r, k in both:
        for cn in cols:
            a, b = r[ix[cn]], k[cn]
            if a != b:
                diff[cn] += 1
                ex.setdefault(cn, (r[ix["page_path"]].split("\\")[-1], r[ix["layer_id"]], b, a))
    any_row = sum(any(r[ix[cn]] != k[cn] for cn in cols) for r, k in both)
    print(f"      画素の列({len(cols)}列)が全部一致した層 {len(both) - any_row:,} / 1列でも違う層 {any_row:,}")
    for cn, n in diff.most_common():
        print(f"      違う列 {cn}: {n}層(例: 旧 {ex[cn][2]!r} → 今 {ex[cn][3]!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
