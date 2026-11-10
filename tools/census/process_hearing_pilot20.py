#!/usr/bin/env python
"""**工程の聞き取りの試し 20作** —— Track N の依頼(`note_n_census_process_pilot20_20261110.md`、ユーザー承認「GO」)。

**作者の方針**(2026-11-10、逐語): 「作者の聞き取りはまず２０くらいからはじめるのがいい　いきなり３００とかやっても
疲れるし、大抵は不具合のようなものが見つかって前提がくずれる（仮説も結論もだ）」

**選び方**(census が決める。答えを見る前に書く):
- 作品 = 構造 DB の (`year`, `work`) の組(`author_process.py` と同じ単位)
- 区間1 = 札 `2021` / `2022` / `2022b` / `2023` / `2023b` から 10作、区間2 = 札 `2024` / `2024b` / `2024c` から 10作
  (**`2025` の札は母集団に無い** —— N の希望「2024〜2025」は 2024 系の3札になる)
- 除く: 裁定済みの11作(`author_process_20261031.csv`)/ **`.clip` の頁が1つも無い作品**(画素の素材に回せないので。
  数を印字する)/ 恒久除外の頁は頁の数から外す / 札の直下に置かれた頁(作品フォルダーが無い)は候補にしない
- 区間ごとに `random.Random(20261110)` で引く。札で層化しない(札の内訳は印字する)
- **選ばなかった作品の数を理由別に印字する**(規律9)

**表**: 作品・札・下絵・線画・機材の備考(前回 10-31 と同じ問い)。下絵・線画はプルダウン
(アナログ / デジタル / 混在 / 分からない)。思い出す助けに頁数・`.clip` の頁数・頁の例を並べる。

**答えから出すこと**(N が決める。census は `author_process.csv` に取るだけ): 空欄は「未記入」で、「分からない」と混ぜない。

    ./venv/bin/python tools/census/process_hearing_pilot20.py
"""
import collections
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RULED = "data/census_full/author_process_20261031.csv"
OUT_XLSX = "data/census_full/hearing_process_pilot20_20261110.xlsx"
OUT_CSV = "data/census_full/hearing_process_pilot20_20261110.csv"
SEED = 20261110
BANDS = [("2021〜2023b", ("2021", "2022", "2022b", "2023", "2023b"), 10),
         ("2024〜2024c", ("2024", "2024b", "2024c"), 10)]
CHOICES = ["アナログ", "デジタル", "混在", "分からない"]


def main() -> int:
    print("[工程の聞き取りの試し 20作(Track N の依頼)]")
    con = db.connect()
    ex = exclusions.load()
    ruled = {(r["year_label"], r["work"]) for r in csv.DictReader(open(os.path.join(ROOT, RULED), encoding="utf-8"))}
    works = collections.defaultdict(lambda: collections.Counter())
    sample = {}
    n_ex = 0
    n_nowork = collections.Counter()
    for p, y, w, f in con.execute("select page_path, year, work, format from pages order by page_path").fetchall():
        if p in ex:
            n_ex += 1
            continue
        if w is None:
            n_nowork[y] += 1
            continue
        works[(y, w)][f] += 1
        if f == "clip":
            sample.setdefault((y, w), p)
    print(f"    恒久除外の頁 {n_ex}(頁の数から外した)/ **作品フォルダーに入っていない頁(札の直下)** "
          + " / ".join(f"{a} {b}頁" for a, b in sorted(n_nowork.items())) + "(作品が無いので候補にしない)")
    rng = random.Random(SEED)
    picks = []
    for band, labels, k in BANDS:
        drop = collections.Counter()
        cand = []
        for key in sorted(works):
            if key[0] not in labels:
                continue
            if key in ruled:
                drop["裁定済み"] += 1
            elif not works[key]["clip"]:
                drop[".clip の頁が無い"] += 1
            else:
                cand.append(key)
        rng.shuffle(cand)
        got = cand[:k]
        picks += [(band, key) for key in sorted(got)]
        print(f"    {band}: 作品 {len(cand) + sum(drop.values())} → 除いた "
              + " / ".join(f"{a} {b}" for a, b in drop.items()) + f" → 候補 {len(cand)} → 引いた {len(got)}"
              f"(選ばなかった候補 {len(cand) - len(got)})")
        print("      札の内訳: " + " / ".join(f"{a} {b}" for a, b in sorted(collections.Counter(y for y, _ in got).items())))
    rows = []
    for i, (band, (y, w)) in enumerate(picks, 1):
        c = works[(y, w)]
        rows.append(dict(no=i, band=band, year_label=y, work=w, n_pages=sum(c.values()), n_clip=c["clip"],
                         example_page=sample[(y, w)]))
    with open(os.path.join(ROOT, OUT_CSV), "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    write_xlsx(rows)
    print(f"    → `{OUT_XLSX}` / 一覧 `{OUT_CSV}`(非公開)")
    return 0


def write_xlsx(rows):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    f, b = Font(name="Arial"), Font(name="Arial", bold=True)
    fill = PatternFill("solid", fgColor="FFFF00")
    ws = wb.active
    ws.title = "説明"
    lines = [("制作の工程の聞き取り(試しの20作。Track N の依頼、census が作成、2026-11-10)", b),
             ("何について答えるか: 「作品」のシートの1行が1つの作品です。その作品の下絵と線画を、アナログで描いたかデジタルで描いたかを選んでください。", f),
             ("前回(10-31)の11作と同じ問いです。機材(iPad など)を覚えていれば備考に書いてください。", f),
             ("黄色の列だけ記入してください。下絵・線画はプルダウン(アナログ / デジタル / 混在 / 分からない)です。", f),
             ("思い出せない作品は「分からない」で構いません。空欄は「まだ答えていない」として扱い、「分からない」とは分けて数えます。", f),
             ("頁数・.clip の頁数・頁の例は、思い出す助けのために並べています。", f),
             ("", f),
             ("記入例(このシートの中だけの例です): 下絵 → アナログ / 線画 → デジタル / 機材の備考 → iPad で清書", f)]
    for i, (t, ft) in enumerate(lines, 1):
        ws.cell(i, 1, t).font = ft
    ws.column_dimensions["A"].width = 120
    wk = wb.create_sheet("作品")
    hdr = ["番号", "区間", "札", "作品", "頁数", ".clip の頁数", "頁の例", "下絵", "線画", "機材の備考", "コメント"]
    for j, h in enumerate(hdr, 1):
        wk.cell(1, j, h).font = b
    for i, r in enumerate(rows, 2):
        for j, v in enumerate([r["no"], r["band"], r["year_label"], r["work"], r["n_pages"], r["n_clip"],
                               r["example_page"], "", "", "", ""], 1):
            wk.cell(i, j, v).font = f
        for j in (8, 9, 10, 11):
            wk.cell(i, j).fill = fill
    dv = DataValidation(type="list", formula1='"' + ",".join(CHOICES) + '"', allow_blank=True)
    wk.add_data_validation(dv)
    dv.add(f"H2:I{len(rows) + 1}")
    for col, wdt in zip("ABCDEFGHIJK", (6, 14, 8, 20, 7, 11, 60, 14, 14, 30, 30)):
        wk.column_dimensions[col].width = wdt
    wk.freeze_panes = "E2"
    wb.save(os.path.join(ROOT, OUT_XLSX))


if __name__ == "__main__":
    raise SystemExit(main())
