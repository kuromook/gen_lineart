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

**画素から測った列は空**(条例10)。抽出側の納品(`face_merge_300_20261107/`)が来たら便ごとに足す。

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
OUT = "data/census_full/p_order300_page_layers_20261108.csv"
EXTRA = ["mask_face_n_px_gt0", "mask_face_n_px_255", "mask_face_n_px_1_254", "mask_face_n_px_total",
         "mask_face_status", "mask_cover_px_ge128", "mask_cover_px_1_127", "colour_class", "wip_status"]


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
            if hm:
                c["has_mask の層"] += 1
                c[f"  面の表の行あり(status {f['status']})" if f else "  面の表の行なし"] += 1
                c["  mask_cover の行あり" if v else "  mask_cover の行なし"] += 1
            rows.append([x.get(h, "") for h in header] + [
                f["n_px_gt0"] if f else "", f["n_px_255"] if f else "", f["n_px_1_254"] if f else "",
                f["n_px_total"] if f else "", f["status"] if f else "",
                v["mask_px_ge128"] if v else "", v["mask_px_1_127"] if v else "",
                col.get(p, ""), wip.get(p, "")])
        under += n_mask < int(o["census_n_material"])
    print(f"    **層 {len(rows):,}行**(頁 {len(pages) - len(missing)})")
    for k, v in c.items():
        print(f"      {k} {v:,}")
    print(f"    **既知セル(依頼の材料の数)**: has_mask の層の数が材料の数より少ない頁 {under}(0 であるはず)")
    cc = collections.Counter(col.get(p, "表に無い") for p in pages)
    print("    カラーかモノクロか(第1段): " + " / ".join(f"{k} {v}" for k, v in cc.items()))
    print("    面の表で引ける頁 " + str(len({p for p, _ in face}))
          + " / mask_cover で引ける頁 " + str(len({p for p, _ in cover})))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header + EXTRA)
        w.writerows(rows)
    print(f"    → `{OUT}`(非公開。列 {len(header) + len(EXTRA)}、画素の列は空)"
          + (f" / **層の無い頁 {len(missing)}**" if missing else ""))
    return 1 if (missing or under) else 0


if __name__ == "__main__":
    raise SystemExit(main())
