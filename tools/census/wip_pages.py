#!/usr/bin/env python
"""**作業途中の頁**(`done*` フォルダーの頁)の列 —— Track L の照会(2026-11-06)への答え。

**作者の言葉**(11-06、L 経由):「done は途中のファイルで、いくらかの原稿フォルダにはこのような
ものがのこっている。census で先日調査したものだ」。**ユーザーの指示**(11-06、このセッション):
**「作業途中というあつかいにしてほしい。列がまだないならカラムをつくって」**。

**同定の規則は名前ではなく裁定である**(条例9)。**名前の述語は候補の生成にだけ使い、
どのフォルダーがどの裁定に当たるかを `RULINGS` に書く。** 名前に当たって裁定に当たらない
フォルダーが在れば**指摘**として鳴る(黙って広げない)。

- **述語**: 構造 DB の頁(恒久除外の頁を除く)のうち、**パスの途中のフォルダーの名前が
  `done` で始まる頁**(大文字・小文字を区別しない)
- **値**: `作業途中`(裁定の在るフォルダーの頁)/ `裁定なし`(それ以外の全頁)。
  **`裁定なし` を「現行」と読まない**
- **届かないもの**: **対になる現行の頁**(作者「バラバラに … 最終的にマージ」—— 名前では
  対応しない。作品を跨ぐ対も在る)/ **ファイル名に `done` を持つ頁**(フォルダーではない。
  **別の作品群では作者が `*done` / `*Done` を「完成形で納品用」と言っている**。Track M 10-29)

    ./venv/bin/python tools/census/wip_pages.py
"""
import collections
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import db  # noqa: E402
import exclusions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_FULL = "data/census_full/wip_pages_20261106.csv"
OUT_PUB = "census/wip_pages.csv"
WIP = "作業途中"
NONE = "裁定なし"

# **裁定 → フォルダー**(`年の札\作品\done*`)。**公開物には裁定の番号だけを書く**
RULINGS = {
    "R1": dict(as_of="2026-10-26 00:38", via="基盤(問い 1-A)",
               words="「作業途中の退避だ　バラバラに別のcmc 管理化に作成されたものが最終的に〔作品2つ〕に"
                     "マージされている　この〔2つのフォルダー〕下の done?/*.clip は背景や仕上げがぬけている」",
               folders=[r"2018\ako6\done", r"2018\ako6\done2", r"2018\ako6\done3", r"2018\ako6\done4",
                        r"2018\ako6\done5", r"2018\ako6\done6", r"2018\ako6\done7", r"2018\ako6\done8",
                        r"2019\ako7\done", r"2019\ako7\done2"]),
    "R2": dict(as_of="2026-10-26 01:07", via="基盤(問い I)",
               words="「I 〔2つのフォルダー〕同様に done? はマージ前のバラバラページ」",
               folders=[r"2019b\ako7\done", r"2019b\ako7\done2"]),
    "R3": dict(as_of="2026-10-26 01:23", via="基盤(問い L。フォルダーのパスと頁の数を見せて聞いた)",
               words="「全部同じ運用だ　共通して背景がない」",
               folders=[r"2017\ako5\done", r"2017\ako5\done2", r"2017\ako5\done4",
                        r"2017\copy\done", r"2017\copy\done2", r"2017\kacho\done",
                        r"2017\kacho\done2", r"2017\pink\done1", r"2017\running\done"]),
    "R4": dict(as_of="2026-11-06", via="このセッション(R3 の問いに出ていなかった8フォルダーを示して聞いた)",
               words="「作業途中というあつかいにしてほしい」",
               folders=[r"2017\ako4\done", r"2017\ako4\done2-1", r"2017\ako4\done3-1",
                        r"2017\ako4\done3-2", r"2017\ako4\done3-3", r"2017\ako4\done4-1",
                        r"2017\ako5\done3", r"2017\ako5\done5"]),
}


def main() -> int:
    print("[作業途中の頁(`done*` フォルダー)]")
    folder_rule = {f.lower(): r for r, d in RULINGS.items() for f in d["folders"]}
    con = db.connect()
    excl = exclusions.load()
    pages = [p for (p,) in con.execute("select page_path from pages order by 1").fetchall()]
    n_excl = sum(1 for p in pages if p in excl)
    pages = [p for p in pages if p not in excl]
    full, agg, unruled, seen_folders = [], collections.defaultdict(lambda: [set(), 0]), collections.Counter(), set()
    n_file_done = 0
    for p in pages:
        parts = p.split("\\")
        idx = [i for i in range(2, len(parts) - 1) if re.match("done", parts[i], re.I)]
        if not idx:
            n_file_done += bool(re.search("done", parts[-1], re.I))
            full.append([p, NONE, "", ""])
            continue
        folder = "\\".join(parts[2:idx[0] + 1])
        r = folder_rule.get(folder.lower())
        if r is None:
            unruled[folder] += 1
            full.append([p, NONE, "", folder])
            continue
        seen_folders.add(folder.lower())
        full.append([p, WIP, r, folder])
        a = agg[(parts[2], r)]
        a[0].add(folder.lower())
        a[1] += 1
    n_wip = sum(1 for r in full if r[1] == WIP)
    print(f"    頁 {len(pages):,}(恒久除外の頁 {n_excl} を除いた)/ **作業途中 {n_wip}** / 裁定なし {len(pages) - n_wip:,}")
    for r, d in RULINGS.items():
        n = sum(1 for x in full if x[2] == r)
        print(f"      {r} {d['as_of']} フォルダー {len(d['folders'])} / 頁 {n}")
    missing = sorted(set(folder_rule) - seen_folders)
    print(f"    **名前の述語に当たって裁定に無いフォルダー {len(unruled)}**(頁 {sum(unruled.values())})"
          f" / **裁定に在って頁が無いフォルダー {len(missing)}**")
    print(f"    ファイル名に `done` を持つ頁(フォルダーではない。数えない){n_file_done}")
    with open(os.path.join(ROOT, OUT_FULL), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page_path", "wip_status", "ruling", "done_folder"])
        w.writerows(full)
    rows = [[y, r, RULINGS[r]["as_of"], len(f), n, ""] for (y, r), (f, n) in sorted(agg.items())]
    rows.append(["ALL", "", "", sum(len(d["folders"]) for d in RULINGS.values()), n_wip, ""])
    rows += [["", r, d["as_of"], "", "", f"**{r}**({d['via']}): {d['words']}"] for r, d in RULINGS.items()]
    rows += [[""] * 5 + [n] for n in NOTE]
    n_pub = bc.write_csv(os.path.join(ROOT, OUT_PUB),
                         ["year_label", "ruling", "as_of", "n_folders", "n_pages", "note"], rows)
    print(f"    {OUT_PUB} {n_pub}行 / 頁ごとの全量 {len(full):,}行(`{OUT_FULL}`)")
    return 1 if unruled or missing else 0


NOTE = [
    "**Track L の照会(2026-11-06)への答え。** **ユーザーの指示(11-06)**: 「**作業途中というあつかいにしてほしい。列がまだないならカラムをつくって**」",
    "**述語**: 構造 DB の頁(恒久除外の頁を除く)のうち、**パスの途中のフォルダーの名前が `done` で始まる頁**。"
    "**値を決めるのは名前ではなく裁定** —— **フォルダーごとにどの裁定に当たるかを器具に書いた**。名前に当たって裁定に無いフォルダーが在れば器具が鳴る",
    "**頁ごとの列は `data/census_full/wip_pages_20261106.csv`**(`page_path`, `wip_status` = `作業途中` / `裁定なし`, `ruling`, `done_folder`)。"
    "**頁のパスを持つので公開しない。** **`裁定なし` を「現行」と読まない**",
    "**作者の言葉は逐語。作品とフォルダーの名前だけ〔 〕で伏せた**(公開物に固有名を書かない)",
    "**作者が述べた共通の性質**: 「背景や仕上げがぬけている」(R1)/「共通して背景がない」(R3)。**census はこれを画素で確かめていない**",
    "**対になる現行の頁は、この列に無い** —— 作者「バラバラに … 最終的にマージ」。**名前では対応しない**(作品を跨ぐ対も在る)",
    "**ファイル名に `done` を持つ頁は数えていない**(フォルダーではない)。"
    "**別の作品群では作者が `*done` / `*Done` を「完成形で納品用」と言っている**(Track M 経由 10-29)—— **同じ語で意味が逆**",
]

if __name__ == "__main__":
    raise SystemExit(main())
