#!/usr/bin/env python
"""**もう1台のドライブ(D)のファイルと、母集団 9,653 の被りを数にする。**

**約束した所**: 発注の便(`push_census_d_drive_inventory_20261030.md`)で
**「突き合わせはこちらでやる」**と書いた。先方は D 側の hash を出すだけ。

**この器具が届く範囲**(T132 の規則で、名前より狭く書く):

- **届く**: **1バイトも違わない同じファイルが両方のドライブに在るか** / D の中の重複 /
  形式ごとの内訳
- **届かない**: **同じ絵かどうか。** 新しい作業で描き直したもの、保存し直したもの、
  版が違うものは、**同じ絵でも別の hash になる**(発注の便にそう書いて頼んだ)。
  そして**ペアデータセットとの被りには届かない** —— `deliveries` は配達の記録で、
  学習用データの記録ではない(**それはペアを作った側の記録**)
- **届かない**: **hash そのものの正しさ。** 両側とも**抽出側が計算した列**である(条例5)

    ./venv/bin/python tools/census/other_drive.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F_SIDE = "inbox/request9_file_sha256_all_20261026.csv"
D_SIDE = "inbox/d_drive_file_sha256_20261031.csv"


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def label_of(page_path: str) -> str:
    """`D:\\2025b\\...` から札だけを取る。**作品フォルダーの名前は取らない**(条例10)。"""
    parts = page_path.replace("/", "\\").split("\\")
    return parts[1] if len(parts) > 2 else "不明"


def main() -> int:
    f_rows, d_rows = load(F_SIDE), load(D_SIDE)
    # **読み取り誤りの行を先に数える**(規律9。落としたものを理由別に印字)
    d_err = [r for r in d_rows if r["f_read_error"] or not r["f_sha256"]]
    f_err = [r for r in f_rows if r["f_read_error"] or not r["f_sha256"]]
    print(f"  F 側 {len(f_rows)}行(hash の無い行 {len(f_err)})"
          f" / D 側 {len(d_rows)}行(hash の無い行 {len(d_err)})")

    f_sha = collections.Counter(r["f_sha256"] for r in f_rows if r["f_sha256"])
    rows = []
    for label in sorted({label_of(r["page_path"]) for r in d_rows}) + ["ALL"]:
        sel = [r for r in d_rows
               if label == "ALL" or label_of(r["page_path"]) == label]
        for fmt in ("clip", "lip", "ALL"):
            sub = [r for r in sel if fmt == "ALL" or r["format"] == fmt]
            if not sub:
                continue
            sha = [r["f_sha256"] for r in sub]
            distinct = set(sha)
            # **「被り」は2つの量**: 相手側に双子を持つファイルの数 と、その異なり数
            twin_files = sum(1 for s in sha if s in f_sha)
            twin_distinct = len({s for s in distinct if s in f_sha})
            rows.append([label, fmt, len(sub), len(distinct),
                         len(sub) - len(distinct), twin_files, twin_distinct,
                         sum(int(r["f_bytes"] or 0) for r in sub), ""])
    rows.append(["", "", "", "", "", "", "", "", NOTE])
    n = bc.write_csv(os.path.join(ROOT, "census", "other_drive_overlap.csv"),
                     ["year_label", "format", "n_files", "n_distinct_files",
                      "n_duplicate_files", "n_files_with_twin_in_population",
                      "n_distinct_with_twin_in_population", "n_bytes", "note"], rows)
    # **双子が母集団のどの札に居るか** —— **札ごとの数は足せない**
    # (同じ hash が複数の札の下に在るので、合計は異なり数を超える)
    by_sha = collections.defaultdict(set)
    for r in f_rows:
        parts = r["page_path"].replace("/", "\\").split("\\")
        if r["f_sha256"] and len(parts) > 2:
            by_sha[r["f_sha256"]].add(parts[2])
    d_distinct = {r["f_sha256"] for r in d_rows if r["f_sha256"]}
    twins = d_distinct & set(by_sha)
    per_year = collections.Counter()
    for s in twins:
        for y in by_sha[s]:
            per_year[y] += 1
    yrows = [[y, n, ""] for y, n in sorted(per_year.items())]
    yrows.append(["", len(twins), NOTE_YEARS])
    n_y = bc.write_csv(os.path.join(ROOT, "census", "other_drive_twin_years.csv"),
                       ["year_label", "n_distinct_twin_hashes", "note"], yrows)

    for r in rows[:-1]:
        print(f"    {r[0]:8s} {r[1]:5s} ファイル {r[2]:5d} / 異なり {r[3]:5d}"
              f" / 母集団に双子を持つ {r[5]:5d}")
    print(f"  双子の異なり hash {len(twins)} が母集団の札 "
          f"{len(per_year)} に在る(**札ごとの数は足せない**)")
    print(f"  other_drive_overlap.csv {n}行 / other_drive_twin_years.csv {n_y}行")
    return 0


NOTE = (
    "**被りは「1バイトも違わない同じファイル」だけを数えている。** "
    "**同じ絵かどうかには届かない** —— 新しい作業で描き直したもの、保存し直したもの、"
    "版が違うものは、**同じ絵でも別の hash になる**(発注のときにそう書いて頼んだ)。"
    "**ペアデータセットとの被りにも届かない** —— "
    "`deliveries` は配達の記録で、学習用データの記録ではない。"
    "**hash は両側とも抽出側が計算した列である**(条例5)。"
    "**`n_files` はファイルの数で、頁として同定したものではない** —— "
    "この集計の 9,653 には1行も入っていない"
)

NOTE_YEARS = (
    "**最後の行の数は、母集団に双子を持つ D の異なり hash の数**(札ごとの合計ではない)。"
    "**札ごとの数は足せない** —— **同じ hash が複数の札の下に在る**ので、"
    "足すと異なり数を超える(母集団の中にも重複が在る。`duplicate_quantities.csv`)。"
    "**これは「その札の作品が D に在る」ではなく「その札の下のファイルと"
    "1バイトも違わないファイルが D に在る」である**(条例1)"
)

if __name__ == "__main__":
    raise SystemExit(main())
