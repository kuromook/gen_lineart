#!/usr/bin/env python
"""**白さの便の行に「名前が `white`・種別 raster」の印**(Track E の頼み、2026-11-22。既知のセルのため。新しい走行ではない)。

- 行 = 白さの列の発注の一覧 `outbox/white_column_order_20261121.csv`(312,614層。`no` で白さの便と結ぶ)
- 印 `name_white_raster` = 構造 DB(10-26 の断面)の層名が **ちょうど `white`**(前後の空白も大小も変えない)かつ `v_layer_kind = raster` なら 1
  —— E の数えた 2,550層(名前の聞き取り第2巡の表。`.clip` 2,549 + `.lip` 1)と同じ述語
- 層名そのものは出さない(条例11)。一覧に入らなかった 2,550層の残りを理由別に数える(規律9)

出力 `data/census_full/e_white_name_flag_20261122.csv`(非公開)。
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORDER = os.path.join(ROOT, "outbox", "white_column_order_20261121.csv")
OUT = os.path.join(ROOT, "data", "census_full", "e_white_name_flag_20261122.csv")


def main() -> int:
    c = db.connect()
    w = {(p, i): f for p, i, f in c.execute(
        "select l.page_path, l.layer_id, p.format from layers l join v_layer_kind v using (page_path, layer_id) "
        "join pages p using (page_path) where l.name = 'white' and v.v_layer_kind = 'raster'").fetchall()}
    print(f"名前がちょうど white・raster: {len(w):,} 層(" + " / ".join(
        f"{k} {v:,}" for k, v in collections.Counter(w.values()).items()) + ")")
    rows, hit = [], set()
    for r in csv.DictReader(open(ORDER, encoding="utf-8")):
        k = (r["page_path"], int(r["layer_id"]))
        f = int(k in w)
        if f:
            hit.add(k)
        rows.append(dict(no=r["no"], page_path=r["page_path"], layer_id=r["layer_id"], name_white_raster=f))
    miss = collections.Counter("lip(画素が無い)" if w[k] == "lip" else "clip だが一覧に無い" for k in w if k not in hit)
    print(f"白さの一覧 {len(rows):,} 行のうち印 1: {len(hit):,} / 一覧に入らなかった: " + (" / ".join(f"{k} {v}" for k, v in miss.items()) or "0"))
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"出力: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
