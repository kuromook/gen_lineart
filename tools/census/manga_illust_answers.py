#!/usr/bin/env python
"""**作品フォルダーは漫画かイラストか** —— 作者の答え(第2段、`colour_class.py` の表)を取り込む。

**作者の記入(2026-11-10、逐語)**: 「２１行まで入力　残りは未確認」。
**読み方(答えを見る前に決めた。ユーザーへの報告で明示する)**:
- 「21行」は Excel の行番号と読む(1行目は見出し)→ **2〜21行目の 20 フォルダーが作者の答え**。
  聞き取りの規律「Excel の表は1回 20行」(作者 2026-11-10)とも合う
- **22行目以降にも値が入っているが「未確認」** —— 作者の答えとして数えない。`status = 未確認` で別に残し、
  集計には入れない(値は消さない。次の 20行の確認で作者が直すかもしれない)
- コメント列も同じ扱い(22行目以降のコメントは未確認)
- 固定の列(フォルダー名・頁数)が元の表と1セルでも違えば止める(取り違えの検出)

    ./venv/bin/python tools/census/manga_illust_answers.py
"""
import collections
import csv
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIG = "data/census_full/hearing_work_folder_manga_or_illust_20261107.xlsx"
ANSWERED = "data/census_full/hearing_work_folder_manga_or_illust_answered_upto21_20261110.xlsx"
ANSWERED_SHA = "e3db91321f89dcb83447b13eacd53c0096404b8dfb036f638f7da13fce118284"
CONFIRMED_LAST_ROW = 21
OUT = "data/census_full/work_folder_manga_or_illust_20261110.csv"
AS_OF = "2026-11-10"


def main() -> int:
    from openpyxl import load_workbook
    print("[作品フォルダーは漫画かイラストか(作者の答えの取り込み)]")
    path = os.path.join(ROOT, ANSWERED)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if sha != ANSWERED_SHA:
        print(f"    **sha256 が違う** {sha[:16]} ≠ {ANSWERED_SHA[:16]}")
        return 1
    o = list(load_workbook(os.path.join(ROOT, ORIG))["作品フォルダー"].iter_rows(values_only=True))
    a = list(load_workbook(path)["作品フォルダー"].iter_rows(values_only=True))
    if len(o) != len(a):
        print(f"    **行数が違う** 元 {len(o)} / 答え {len(a)}")
        return 1
    bad = sum(x[:6] != y[:6] for x, y in zip(o, a))
    if bad:
        print(f"    **固定の列が元の表と違う行 {bad}**")
        return 1
    print(f"    sha256 {sha[:16]} / 固定の列は元の表と全行一致({len(a) - 1} フォルダー)")
    rows = []
    cnt = collections.defaultdict(collections.Counter)
    for i, r in enumerate(a[1:], 2):
        st = "作者の答え" if i <= CONFIRMED_LAST_ROW else "未確認"
        ans = r[6] if r[6] is not None else "未記入"
        cnt[st][ans] += 1
        rows.append(dict(excel_row=i, work_dir=r[0], n_pages=r[1], n_lip=r[2], n_mono=r[3], n_colour_layer_pages=r[4],
                         n_p_pages=r[5] or 0, answer=ans, comment=r[7] or "", status=st,
                         decided_by="作者の記入式の表(census が作成、ユーザー経由)" if st == "作者の答え" else "",
                         as_of=AS_OF))
    for st in ("作者の答え", "未確認"):
        c = cnt[st]
        print(f"    {st}: {sum(c.values())} フォルダー → " + " / ".join(f"{k} {v}" for k, v in c.most_common()))
    conf = [x for x in rows if x["status"] == "作者の答え"]
    print(f"    作者の答えの 20 フォルダーの頁: カラーの層がある頁 {sum(x['n_colour_layer_pages'] for x in conf)} / "
          f"Track P の頁 {sum(x['n_p_pages'] for x in conf)}")
    for k in ("漫画", "イラスト", "漫画とイラストが混ざる"):
        s = [x for x in conf if x["answer"] == k]
        print(f"      {k}: {len(s)} フォルダー / カラーの層がある頁 {sum(x['n_colour_layer_pages'] for x in s)} / "
              f"Track P の頁 {sum(x['n_p_pages'] for x in s)}")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"    → `{OUT}`(非公開。作品名を含む)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
