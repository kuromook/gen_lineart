#!/usr/bin/env python
"""**P の300頁の全層に `v_layer_kind` を付ける** —— Track P の依頼(`note_p_census_layer_kind_300_20261109.md`)。
11-09 に受けて、11-12 まで答えていなかった(census の取りこぼし)。

- 行: `p_order300_page_layers_20261110_v2.csv` の全行(40,879行)と同じ (`page_path`, `layer_id`)。構造 DB の `v_layer_kind` を引く
- 引けなかった行は数えて印字する(黙って落とさない)

    ./venv/bin/python tools/census/p_layer_kind_300.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = "data/census_full/p_order300_page_layers_20261110_v2.csv"
OUT = "data/census_full/p_order300_layer_kind_20261112.csv"


def main() -> int:
    print("[P の300頁の層の種別]")
    con = db.connect()
    rows = [(r["page_path"], int(r["layer_id"])) for r in csv.DictReader(open(os.path.join(ROOT, SRC), encoding="utf-8"))]
    pages = sorted({p for p, _ in rows})
    q = ",".join("?" * len(pages))
    kind = {(p, lid): k for p, lid, k in con.execute(
        f"select page_path, layer_id, v_layer_kind from v_layer_kind where page_path in ({q})", pages).fetchall()}
    miss = [k for k in rows if k not in kind]
    print(f"    行 {len(rows):,} / 頁 {len(pages)} / 種別が引けなかった行 {len(miss)}")
    print("    種別: " + " / ".join(f"{a} {b:,}" for a, b in collections.Counter(kind.get(k, "") for k in rows).most_common()))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page_path", "layer_id", "v_layer_kind"])
        for k in rows:
            w.writerow([k[0], k[1], kind.get(k, "")])
    print(f"    → `{OUT}`(非公開)")
    return 1 if miss else 0


if __name__ == "__main__":
    raise SystemExit(main())
