#!/usr/bin/env python
"""**作品で分ける単位を、構造 DB の全作品フォルダーについて作る**(`p_split_columns.py` と同じ規則を関数にした)。

規則: **作者の編の裁定**(`arc_assignment_20261030.csv`)で束ね、**同じ中身のファイル(sha256)を共有する作品フォルダー**を束ねる。
**編の境を跨ぐ中身の共有は束ねない**(条例9)。名前の似かたでは束ねない。

**既知セル**: Track P の 941頁の `split_unit`(`p_split_columns_20261107.csv`)と、分け方(同じ単位かどうか)が一致すること。

    ./venv/bin/python tools/census/split_units.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
from p_split_columns import ARC, UF, wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P_SPLIT = "data/census_full/p_split_columns_20261107.csv"


def units(con):
    """作品フォルダー → 単位の代表(作品フォルダー名)。全頁の作品フォルダーを含む"""
    arc_of = {f"F:\\manuscripts\\{r['year_label']}\\{r['work']}": r["arc"]
              for r in csv.DictReader(open(os.path.join(ROOT, ARC), encoding="utf-8"))}
    by_hash = collections.defaultdict(set)
    for p, s in con.execute("select page_path, f_sha256 from file_hashes where f_sha256 is not null").fetchall():
        by_hash[s].add(wd(p))
    u = UF()
    for p, in con.execute("select page_path from pages").fetchall():
        u.f(wd(p))
    for ws in by_hash.values():
        ws = sorted(ws)
        for w in ws[1:]:
            a, b = arc_of.get(ws[0]), arc_of.get(w)
            if a and b and a != b:
                continue
            u.u(ws[0], w)
    first = {}
    for w, a in sorted(arc_of.items()):
        if a in first:
            u.u(first[a], w)
        else:
            first[a] = w
    groups = collections.defaultdict(list)
    for w in list(u.p):
        groups[u.f(w)].append(w)
    rep = {r: min(ws) for r, ws in groups.items()}
    return {w: rep[u.f(w)] for w in u.p}


def main() -> int:
    con = db.connect()
    un = units(con)
    print(f"[作品で分ける単位] 作品フォルダー {len(un)} → 単位 {len(set(un.values()))}")
    rows = list(csv.DictReader(open(os.path.join(ROOT, P_SPLIT), encoding="utf-8")))
    pairs = collections.Counter()
    for r in rows:
        pairs[(r["split_unit"], un[r["work_dir"]])] += 1
    a = collections.Counter(k[0] for k in pairs)
    b = collections.Counter(k[1] for k in pairs)
    ok = all(v == 1 for v in a.values()) and all(v == 1 for v in b.values())
    print(f"    **既知セル**: Track P の 941頁の単位 {len(a)} と、この関数の単位 {len(b)} —— "
          + ("**分け方が一致**" if ok else "**分け方が違う**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
