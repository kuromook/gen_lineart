#!/usr/bin/env python
"""**`is_base_mask` v4** —— 作者の答え(2026-11-26、ファイルを開いて。逐語「下地白　植物のベースマスクのようだ　使い道は３Dのそれと同じだが、この植物は３Dではなくパターンブラシで描いたもののようにみえる　あつかいとしては下地Nと同じでよい」)。

- v4 = v3(`is_base_mask_v3_20261126.csv`、5,311層)+ 層名が**ちょうど** `下地白 `(末尾に空白1つ)でフォルダーでない層
- 描いた画素は全部純黒(白 0)。**3D の出力ではなくパターンブラシで描いた植物のベースマスク**(作者の見立て)—— 下地の系統は 3D の印にならない
- v1〜v3 は書き換えない(列を残す)。前後の数を印字する。土台 0・乗る層 0
- a の判定に当てるのは Track E(v12 を立てたら census が表を作る)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
V3 = os.path.join(D, "is_base_mask_v3_20261126.csv")
OUT = os.path.join(D, "is_base_mask_v4_20261126.csv")
NAMES = ["下地白 "]


def main() -> int:
    c = db.connect()
    add = {(p, str(i)) for p, i in c.execute(
        f"select page_path, layer_id from layers where name in ({','.join('?' * len(NAMES))}) and not folder", NAMES).fetchall()}
    rows, n3, n4 = [], 0, 0
    for r in csv.DictReader(open(V3, encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        v4 = int(r["is_base_mask_v3"] == "1" or k in add)
        n3 += r["is_base_mask_v3"] == "1"; n4 += v4
        rows.append(dict(page_path=k[0], layer_id=k[1], is_base_mask_v4=v4, is_base_mask_v3=r["is_base_mask_v3"], is_base_mask_v2=r["is_base_mask_v2"],
                         is_base_mask_v1=r["is_base_mask_v1"], in_a_population=r["in_a_population"]))
    print(f"[is_base_mask v4] 足す層 {len(add)} / v3(11-26){n3:,} → v4(11-26){n4:,}(+{n4 - n3})")
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"    → {OUT}({len(rows):,}行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
