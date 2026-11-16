#!/usr/bin/env python
"""**非表示の取り込んだ画像の層 4,356 の鍵 v3** —— 鍵 v2 に頁の代表の列を足す(Track E の頼み 2026-11-17)。

**数を見る前に決めたこと**: 鍵 v2(`image_material_hidden_groups_key_v2_20261116.csv`)の行と列は変えない。
`page_representative.csv` から `is_representative`(1/0。表に無い頁は空)と `representative_path` を足すだけ。
代表は「ファイルの sha256 の束で最も古い札」(作者の裁定)。束の大きさ `cluster_size` も足す。
印字: 名前の群ごとの代表でない層の数(E の母集団 3,557 = 群1 と群3、v7 で群2 の85層が加わったので、群ごとに分けて出す)

    ./venv/bin/python tools/census/hidden_groups_key_v3.py
"""
import collections
import csv
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KEY = "data/census_full/image_material_hidden_groups_key_v2_20261116.csv"
KEY_SHA = "084691179990f6cd"
REP = "data/census_full/page_representative.csv"
OUT = "data/census_full/image_material_hidden_groups_key_v3_20261117.csv"


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def main() -> int:
    print("[非表示の取り込んだ画像の層の鍵 v3(代表の列)]")
    if not sha(KEY).startswith(KEY_SHA):
        print(f"    **sha256 が違う** {KEY}")
        return 1
    rep = {r["page_path"]: r for r in csv.DictReader(open(os.path.join(ROOT, REP), encoding="utf-8"))}
    rows = list(csv.DictReader(open(os.path.join(ROOT, KEY), encoding="utf-8")))
    c = collections.Counter()
    for r in rows:
        x = rep.get(r["page_path"])
        r["is_representative"] = "" if x is None else int(x["is_representative"] == "yes")
        r["representative_page_path"] = "" if x is None else x["representative_path"]
        r["cluster_size"] = "" if x is None else x["cluster_size"]
        c[(r["name_group"], r["is_representative"])] += 1
    for g in sorted({k[0] for k in c}):
        print(f"    群 {g}: 代表 {c[(g, 1)]:,} / 代表でない {c[(g, 0)]:,} / 表に無い {c[(g, '')]:,}")
    print(f"    計 {len(rows):,}: 代表でない {sum(v for k, v in c.items() if k[1] == 0):,} / 表に無い {sum(v for k, v in c.items() if k[1] == ''):,}")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"    → `{OUT}`(非公開、{len(rows):,}行、sha256 {sha(OUT)[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
