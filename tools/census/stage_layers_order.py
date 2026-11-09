#!/usr/bin/env python
"""**工程の段の画素が取り出せるか(Track E の問い2)を抽出側に数えてもらう層の一覧。** ユーザーの承認 2026-11-09。

- 頁: `.clip`(恒久除外を除く)で、線画系と下絵系の名前の層が両方在る 2,102頁(census の名前の当て方。
  **作者の段ではない**。的中率は `stage_hearing.py` の答えで測る)
- 層: その頁の、名前が線画系 / 下絵系に当たるフォルダーでない層(テキストを除く)
  + **聞き取り表の層で上に入らないもの**(答えが来たら作者の段で引き直すため)

    ./venv/bin/python tools/census/stage_layers_order.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
from stage_hearing import LINE, SKETCH, OUT_KEY  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = "outbox/stage_layers_order_20261109.csv"


def main() -> int:
    print("[工程の段の画素の数え上げの層の一覧(抽出側への発注)]")
    con = db.connect()
    ex = exclusions.load()
    rows = con.execute("""select l.page_path, l.layer_id, coalesce(l.name, ''), k.v_layer_kind
                            from layers l join pages p using (page_path) join v_layer_kind k using (page_path, layer_id)
                           where p.format = 'clip' and not l.folder and k.v_layer_kind <> 'text'""").fetchall()
    pg = collections.defaultdict(list)
    for r in rows:
        if r[0] not in ex:
            pg[r[0]].append(r)
    both = {p for p, rs in pg.items() if any(LINE.search(r[2]) for r in rs) and any(SKETCH.search(r[2]) for r in rs)}
    out = []
    for p in sorted(both):
        for r in pg[p]:
            a, b = bool(LINE.search(r[2])), bool(SKETCH.search(r[2]))
            if a or b:
                out.append((p, r[1], "両方当たる" if a and b else "線画系" if a else "下絵系", "名前", r[3]))
    seen = {(o[0], o[1]) for o in out}
    kind = {(r[0], r[1]): r[3] for r in rows}
    hear = sorted({(r["page_path"], int(r["layer_id"]))
                   for r in csv.DictReader(open(os.path.join(ROOT, OUT_KEY), encoding="utf-8"))})
    for p, lid in hear:
        if (p, lid) not in seen:
            out.append((p, lid, "", "聞き取り", kind[(p, lid)]))
    print(f"    頁 {len({o[0] for o in out}):,}(名前で両方 {len(both):,} + 聞き取りの頁)/ 層 {len(out):,}")
    print("    " + " / ".join(f"{k or '(名前に当たらない)'} {v:,}" for k, v in collections.Counter(o[2] for o in out).most_common()))
    print("    種別: " + " / ".join(f"{k} {v:,}" for k, v in collections.Counter(o[4] for o in out).most_common()))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["no", "page_path", "layer_id", "name_family", "why_listed", "v_layer_kind"])
        for i, o in enumerate(out, 1):
            w.writerow([i, *o])
    print(f"    → `{OUT}`(非公開。抽出側へ SCP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
