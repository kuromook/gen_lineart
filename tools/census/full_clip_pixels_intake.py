#!/usr/bin/env python
"""**全 .clip 頁の層の画素(16便)の取り込み** —— 発注 `push_census_order_full_clip_pixels_20261112.md` と追補(`pixel_formats`)。

**数を見る前に決めたこと**:
- 便ごとに sha256 で固定する(`BATCHES`)。届いた便だけを読み、届いていない便は「未着」と印字する
- 頁の集合が発注の一覧(`outbox/full_clip_pixels_order_20261112.csv`)のその便と一致し、頁ごとの層の数が `n_layers` と一致すること。
  層の鍵 (`page_path`, `layer_id`) が構造 DB の `layers` に在ること。どれかが外れたら止める
- `empty_c` は `stage_hearing_round2_b.empty_c` の正本の規則で作る(読めない層は "0" = 空ではない = 測れない)
- **既知セル**: 役割の表 v3(別の走行で作った表)と重なる層で、`empty_c` が一致するかを印字する。不一致は数と鍵を出す
- 3D の線画の10行(Track R、作者の答えつき)は、全部の行が届くまで的中率を出さない。届いた行数だけを印字する
- 出力は非公開の `data/census_full/full_clip_pixels_layers_merged.csv`(届いた便を全部つないで `batch` と `empty_c` を足したもの)

    ./venv/bin/python tools/census/full_clip_pixels_intake.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
from stage_hearing_round2_b import TABLE_V3, empty_c  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORDER = "outbox/full_clip_pixels_order_20261112.csv"
ORDER_SHA = "8d8a2f6028036e70"
DIR = "data/census_full/full_clip_pixels"
R3D = "data/census_full/r_3d_lineart_rows_author_20261113.csv"
OUT = "data/census_full/full_clip_pixels_layers_merged.csv"
# 便: (層の CSV, sha256, 頁の CSV, sha256, 届いた日)
BATCHES = {
    1: ("full_clip_pixels_batch01_20261113/full_clip_pixels_batch01_layers_20261113.csv",
        "a0a86c7d50f6e833888eb475d8422f6725d8e650b5f6231933d219b276eb8044",
        "full_clip_pixels_batch01_20261113/full_clip_pixels_batch01_pages_20261113.csv",
        "f1c9164b9bcf6caf88a7b2e329871071b3784699227769d93e1d84ed561e0d0c", "2026-11-13"),
}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> int:
    print("[全 .clip 頁の層の画素の取り込み]")
    op = os.path.join(ROOT, ORDER)
    if not sha(op).startswith(ORDER_SHA):
        print("    **発注の一覧の sha256 が違う**")
        return 1
    order = collections.defaultdict(dict)
    for r in csv.DictReader(open(op, encoding="utf-8")):
        order[int(r["batch"])][r["page_path"]] = int(r["n_layers"])
    con = db.connect()
    keys = set(con.execute("select page_path, layer_id from layers").fetchall())
    out, fields = [], None
    for b in sorted(order):
        if b not in BATCHES:
            print(f"    便{b}: 未着({len(order[b])}頁 / {sum(order[b].values())}層)")
            continue
        lp, lsha, pp, psha, as_of = BATCHES[b]
        for p, s in ((lp, lsha), (pp, psha)):
            if sha(os.path.join(ROOT, DIR, p)) != s:
                print(f"    **便{b}: sha256 が違う** {p}")
                return 1
        pages = {r["page_path"]: r for r in csv.DictReader(open(os.path.join(ROOT, DIR, pp), encoding="utf-8"))}
        rows = list(csv.DictReader(open(os.path.join(ROOT, DIR, lp), encoding="utf-8")))
        if set(pages) != set(order[b]):
            print(f"    **便{b}: 頁の集合が発注と違う**(届いた {len(pages)} / 発注 {len(order[b])})")
            return 1
        per = collections.Counter(r["page_path"] for r in rows)
        bad = [p for p in order[b] if per[p] != order[b][p] or int(pages[p]["n_layers"]) != order[b][p]]
        if bad:
            print(f"    **便{b}: 層の数が発注と違う頁 {len(bad)}**")
            return 1
        miss = [r for r in rows if (r["page_path"], int(r["layer_id"])) not in keys]
        if miss:
            print(f"    **便{b}: 構造 DB に無い層 {len(miss)}**")
            return 1
        perr = sum(1 for p in pages.values() if p["error"])
        rs = collections.Counter(r["render_status"] for r in rows)
        pf = collections.Counter(r["pixel_formats"] or "(空欄)" for r in rows)
        for r in rows:
            r["empty_c"] = empty_c(r) if r["render_status"] != "描画できなかった" else ""
            r["batch"] = str(b)
        ec = collections.Counter(r["empty_c"] or "(測れない)" for r in rows)
        print(f"    便{b}({as_of}): {len(pages)}頁 / {len(rows)}層。頁のエラー {perr}。発注と頁・層の数が一致、全層が構造 DB に在る")
        print("        render_status: " + " / ".join(f"{k} {n}" for k, n in rs.most_common()))
        print("        pixel_formats: " + " / ".join(f"{k} {n}" for k, n in pf.most_common()))
        print("        empty_c: " + " / ".join(f"{k} {n}" for k, n in sorted(ec.items())) +
              "(「測れない」は描画できなかった層 = フォルダーなど)")
        f5 = collections.Counter(r["empty_c"] for r in rows if "5" in r["pixel_formats"].split(";"))
        print(f"        うち形式5 を含む層(形式5 は 10-31 から調査中): empty_c=1 {f5['1']} / 0 {f5['0']}"
              " —— empty_c は正本の規則のまま。形式5 の空を信じてよいかは調査中のまま")
        fields = fields or list(rows[0])
        out.extend(rows)
    if not out:
        print("    届いた便が無い")
        return 0
    # 既知セル: 役割の表 v3 と重なる層
    v3 = {(r["page_path"], r["layer_id"]): empty_c(r) for r in csv.DictReader(open(os.path.join(ROOT, TABLE_V3), encoding="utf-8"))}
    both = [(r, v3[(r["page_path"], r["layer_id"])]) for r in out if (r["page_path"], r["layer_id"]) in v3]
    diff = [(r["page_path"], r["layer_id"], r["empty_c"], v) for r, v in both if r["empty_c"] != v]
    print(f"    既知セル(役割の表 v3 と重なる層): {len(both)}層、empty_c の不一致 {len(diff)}"
          + ("" if both else "(重なりが無いので、この便では照合できていない)"))
    for d in diff[:10]:
        print(f"        不一致: layer_id {d[1]} 今回 {d[2]} / v3 {d[3]}")
    # 3D の10行
    r3 = list(csv.DictReader(open(os.path.join(ROOT, R3D), encoding="utf-8")))
    got = {(r["page_path"], r["layer_id"]): r for r in out}
    n3 = sum(1 for r in r3 if (r["page_path"], r["layer_id"]) in got)
    print(f"    3D の線画の行(作者の答えつき、R の一覧 {len(r3)}行、異なる層 {len({(r['page_path'], r['layer_id']) for r in r3})}): 届いた {n3}行。"
          "全部届くまで的中率は出さない")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"    → `{OUT}`(非公開、{len(out)}層)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
