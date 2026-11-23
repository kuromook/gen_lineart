#!/usr/bin/env python
"""**頁のファイルの複製の組**(Track Q の頼み、2026-11-23。Q20 で別の時期のフォルダーへ複製された頁が3組見つかった。新しい走行ではない)。

- 述語: 構造 DB(10-26 の断面)の `file_hashes.f_sha256`(抽出側がファイル全体から計算した値。census はファイルを開いていない)が
  **同じ頁どうしを1組**とする。全 9,653頁(.lip を含む)
- **ファイルが同じ = 複製**であって、**ファイルが違う = 別の版**ではない(作業途中の保存で中身の違う同じ頁がある。
  Q20 の q20_09 = q20_12 は枠の層の画が一致してファイルは違う)。**どれを採るかは決めない**(版の同定は作者に聞く。条例9)
- 列: `copy_group`(組の番号。組に入らない頁は空)/ `copy_group_size` / `copy_group_years`(組の年代札、名前の順)/
  `earliest_in_group`(組の中で年代札が最も早い頁か。同じ札が複数なら路の順で最初。**採る側の推しではなく引く段で1頁に畳むための決まった鍵**)
出力 `data/census_full/page_file_copies_20261123.csv`(非公開)。
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "census_full", "page_file_copies_20261123.csv")


def main() -> int:
    c = db.connect()
    rows = c.execute("select f.page_path, f.f_sha256, p.year from file_hashes f join pages p using (page_path)").fetchall()
    by = collections.defaultdict(list)
    for p, s, y in rows:
        by[s].append((y, p))
    gid, out = {}, []
    n = 0
    for s in sorted(by, key=lambda s: sorted(by[s])[0]):
        if len(by[s]) > 1:
            n += 1
            gid[s] = n
    for p, s, y in sorted(rows, key=lambda r: r[0]):
        g = by[s]
        multi = len(g) > 1
        out.append(dict(page_path=p, year_label=y, copy_group=gid.get(s, ""), copy_group_size=len(g) if multi else 1,
                        copy_group_years=";".join(sorted({x[0] for x in g})) if multi else "",
                        earliest_in_group=int(sorted(g)[0][1] == p) if multi else ""))
    sz = collections.Counter(len(v) for v in by.values())
    inm = sum(len(v) for v in by.values() if len(v) > 1)
    print(f"頁 {len(rows):,} / 異なるファイル {len(by):,} / 組 {n:,}(頁 {inm:,})/ 組の大きさ: " +
          " / ".join(f"{k} が {v:,}" for k, v in sorted(sz.items()) if k > 1))
    cross = sum(1 for v in by.values() if len(v) > 1 and len({x[0] for x in v}) > 1)
    print(f"  組のうち年代札を跨ぐ {cross:,} / 同じ札の中だけ {n - cross:,}")
    print(f"  引く段で1頁に畳むと落ちる頁 {inm - n:,}")
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
    print(f"出力: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
