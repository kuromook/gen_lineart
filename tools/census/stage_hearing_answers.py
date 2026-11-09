#!/usr/bin/env python
"""**工程の段の聞き取りの答え**(作者、2026-11-09 記入)から、`stage_hearing.py` の冒頭で**答えを見る前に**決めた数を出す。

- 答え: `data/census_full/hearing_drawing_stages_answered_20261109.xlsx`(作者の記入。sha256 先頭は印字)
- 鍵: `hearing_drawing_stages_key_20261109.csv`(census の名前の当て方。作者に見せていない)
- **作者の但し書き(逐語)**: 「描き文字関係は属性入力でぶれてしまったかもしれない」 —— **描き文字・フキダシの層
  (名前で引く。census の当て方)を含む数と除いた数を両方、名前を付けて印字する。** 答えは書き換えない

    ./venv/bin/python tools/census/stage_hearing_answers.py
"""
import collections
import csv
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_hearing import OUT_KEY  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANS = "data/census_full/hearing_drawing_stages_answered_20261109.xlsx"
OUT = "data/census_full/hearing_drawing_stages_scored_20261109.csv"
TEXTISH = re.compile(r"(描き文字|フキダシ)")
NOT_STAGE = "段の線ではない(塗り・効果・資料など)"
SHORT = {"清書(線画)": "清書", NOT_STAGE: "段ではない", "": "段ではない(空欄)"}


def main() -> int:
    import openpyxl
    print("[工程の段の聞き取りの答え]")
    print(f"    答え sha256 先頭 {hashlib.sha256(open(os.path.join(ROOT, ANS), 'rb').read()).hexdigest()[:16]}")
    wb = openpyxl.load_workbook(os.path.join(ROOT, ANS))
    done = {r[0] for r in wb["頁"].iter_rows(min_row=2, values_only=True) if r[5] and "済" in str(r[5])}
    key = {(int(r["no"]), int(r["pos"])): r for r in csv.DictReader(open(os.path.join(ROOT, OUT_KEY), encoding="utf-8"))}
    rows, blank_by_page = [], collections.Counter()
    for r in wb["層"].iter_rows(min_row=2, values_only=True):
        k = key[(int(r[0]), int(r[2]))]
        if k["name"] != (r[3] or ""):
            print(f"    **鍵と名前が合わない行** 頁{r[0]} 上から{r[2]}")
            return 1
        a = (r[7] or "").strip()
        blank_by_page[r[0]] += not a
        rows.append(dict(no=int(r[0]), pos=int(r[2]), rule=k["rule"], author=SHORT.get(a, a), comment=r[8] or "",
                         textish=bool(TEXTISH.search(k["name"])), kind=k["kind"], stratum=k["stratum"], page_path=k["page_path"]))
    pages = sorted({x["no"] for x in rows})
    print(f"    層 {len(rows)} / 頁 {len(pages)} / 空欄の段 {sum(blank_by_page.values())}")
    print(f"    **「済」の頁 {len(done)}**(宣言した規則は「済」の頁だけ集計)/ **全層に記入のある頁 {sum(blank_by_page[p] == 0 for p in pages)}**")
    print("    → 空欄が無いので「空欄 = まだ見ていない」の取り違えは起きない。**下は全層記入の12頁で数える**(「済」の頁で数えると 0頁)")
    print("    作者の段: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(x["author"] for x in rows).most_common()))

    def score(sub, label):
        n_unk = sum(x["author"] == "分からない" for x in sub)
        sub = [x for x in sub if x["author"] != "分からない"]
        print(f"  [{label}] 層 {len(sub)}(「分からない」を外した {n_unk})")
        t = collections.Counter((x["rule"], x["author"]) for x in sub)
        auth = sorted({x["author"] for x in sub})
        print("    当て方 \\ 作者: " + " | ".join(auth))
        for ru in ("清書", "下絵", "両方当たる", "なし"):
            if any(k[0] == ru for k in t):
                print(f"    {ru:6s}: " + " | ".join(f"{t[(ru, a)]}" for a in auth))
        for ru, au in (("清書", "清書"), ("下絵", "下絵")):
            said = sum(v for (r_, a_), v in t.items() if r_ == ru)
            hit = t[(ru, au)]
            truth = sum(v for (r_, a_), v in t.items() if a_ == au)
            print(f"    **{au}**: 当て方が{ru}と言った {said} のうち作者も{au} {hit}"
                  f"(的中 {hit / said:.3f})" if said else f"    **{au}**: 当て方が{ru}と言った層 0", end="")
            print(f" / 作者が{au}と言った {truth} のうち当て方が言わなかった {truth - hit}"
                  f"(取りこぼし {(truth - hit) / truth:.3f})" if truth else " / 作者がそう言った層 0")
    score(rows, "全層")
    score([x for x in rows if not x["textish"]], "描き文字・フキダシの名前の層を除く(作者の但し書き)")
    tx = [x for x in rows if x["textish"]]
    print(f"    描き文字・フキダシの名前の層 {len(tx)}: 作者の段 " + " / ".join(f"{k} {v}" for k, v in collections.Counter(x['author'] for x in tx).most_common()))
    # 頁ごと: 作者の答えで段が何種類残っているか(頁のシートの問いは空なので層から数える)
    st = collections.defaultdict(set)
    for x in rows:
        if x["author"] in ("あたり", "下絵", "清書"):
            st[x["no"]].add(x["author"])
    c = collections.Counter(len(st[p]) for p in pages)
    print("    **頁ごと、作者の答えで別々の層に残る段の数**(層のシートから数えた): " + " / ".join(f"{k}段 {c[k]}頁" for k in sorted(c)))
    for p in pages:
        print(f"      頁{p}({rows[[x['no'] for x in rows].index(p)]['stratum']}): {sorted(st[p]) or '-'}")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"    → `{OUT}`(非公開)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
