#!/usr/bin/env python
"""**`is_base_mask` v3** —— 作者の答え(2026-11-26、census の問い): `下地 1 のコピー` と `下地 2 のコピー` も「同じ扱いでよい」。

- v3 = v2(`is_base_mask_v2_20261126.csv`、5,309層)+ 層名が**ちょうど** `下地 1 のコピー` / `下地 2 のコピー` でフォルダーでない層
- `下地白 `(末尾に空白)は作者「わからない　一応直接見ておく」—— **入れない**(答えを待つ)
- v1・v2 は書き換えない(列を残す)。前後の数を印字する。2層とも .lip(2014)でクリッピングの土台、乗る層が各2
- a の判定に当てるのは Track E(v11 を立てたら census が表を作る)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
V2 = os.path.join(D, "is_base_mask_v2_20261126.csv")
OUT = os.path.join(D, "is_base_mask_v3_20261126.csv")
NAMES = ["下地 1 のコピー", "下地 2 のコピー"]


def main() -> int:
    c = db.connect()
    add = {(p, str(i)) for p, i in c.execute(
        f"select page_path, layer_id from layers where name in ({','.join('?' * len(NAMES))}) and not folder", NAMES).fetchall()}
    rows, n2, n3 = [], 0, 0
    for r in csv.DictReader(open(V2, encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        v3 = int(r["is_base_mask_v2"] == "1" or k in add)
        n2 += r["is_base_mask_v2"] == "1"; n3 += v3
        rows.append(dict(page_path=k[0], layer_id=k[1], is_base_mask_v3=v3, is_base_mask_v2=r["is_base_mask_v2"],
                         is_base_mask_v1=r["is_base_mask_v1"], in_a_population=r["in_a_population"]))
    print(f"[is_base_mask v3] 足す層 {len(add)} / v2(11-26){n2:,} → v3(11-26){n3:,}(+{n3 - n2})")
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"    → {OUT}({len(rows):,}行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
