#!/usr/bin/env python
"""**全 .clip 頁の層の画素の発注の一覧** —— Track E の発注(`note_census_order_empty_pixels_corpus_and_two_rulings_20261112.md` §1、
ユーザーの承認「2発注する」)。空(定義C)を全部の層について言うため。

**決めたこと(数える前に)**:
- 母集団: 構造 DB の `.clip` の頁で、層が1つ以上ある頁。恒久除外の頁を外す。層の無い `.clip` の頁は数を印字して外す
- 便の切り方: **年代札ごと**(E の頼み「年代札か作品フォルダー単位で便を切る」)。札の中はパスの辞書順
- `ガイド`(raster)の層がある頁に印を付ける(E の発注(5)。この走行に必ず含める)
- 列は census の側の参考(層の数・作品単位・便)。測るのには使わない

    ./venv/bin/python tools/census/full_clip_pixels_order.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = "outbox/full_clip_pixels_order_20261112.csv"


def main() -> int:
    print("[全 .clip 頁の画素の発注の一覧]")
    con = db.connect()
    ex = exclusions.load()
    un = split_units.units(con)
    pages = con.execute("""select p.page_path, p.year, count(l.layer_id), sum(case when l.name = 'ガイド' then 1 else 0 end)
                             from pages p left join layers l using (page_path) where p.format = 'clip' group by 1, 2""").fetchall()
    drop = collections.Counter()
    keep = []
    for p, y, n, g in pages:
        if p in ex:
            drop["恒久除外の頁"] += 1
        elif n == 0:
            drop["層が無い頁"] += 1
        else:
            keep.append((p, y, n, g or 0))
    print(f"    .clip の頁 {len(pages):,} → 落とした " + " / ".join(f"{a} {b}" for a, b in drop.most_common()) + f" → **{len(keep):,} 頁**")
    keep.sort(key=lambda t: (t[1], t[0]))
    years = sorted({t[1] for t in keep})
    batch = {y: i for i, y in enumerate(years, 1)}
    out = []
    for i, (p, y, n, g) in enumerate(keep, 1):
        out.append(dict(no=i, batch=batch[y], year_tag=y, page_path=p, split_unit=un.get(wd(p)) or wd(p), n_layers=n, has_guide=int(g > 0)))
    print(f"    層 {sum(t[2] for t in keep):,} / 作品単位 {len({o['split_unit'] for o in out})} / `ガイド` のある頁 {sum(o['has_guide'] for o in out)}")
    print("    便(年代札): " + " / ".join(f"{batch[y]}={y} {sum(1 for o in out if o['year_tag'] == y)}頁" for y in years))
    print(f"    見積もり: {len(keep):,} 頁 × 10.6 秒 = {len(keep) * 10.6 / 3600:.1f} 時間(E の第3段の実測。最大 49 秒/頁)")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"    → `{OUT}` sha256 {hashlib.sha256(open(os.path.join(ROOT, OUT), 'rb').read()).hexdigest()[:16]}(非公開)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
