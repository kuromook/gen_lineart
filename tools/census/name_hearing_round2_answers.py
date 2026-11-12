#!/usr/bin/env python
"""**層名の聞き取り 第2回の答えを取り込む** —— 問いは「この名前の層は、どの頁でも『絵から除外』してよいですか」。

**答えを見る前に決めたこと**(`name_hearing_round2_sheet.py` と E の発注):
- **第1回(テンプレートか)の答えとは混ぜない。** 出力は別のファイルにし、全行に札 `round2_any_page_exclude` を付ける
- 固定の列(行〜.lip の頁)が作った表と全行一致することを確かめる(違えば止める)。空欄は「未記入」
- 作者のコメントは逐語で残す。**census は答えを規則にしない**(規則を立てるのは E)
- 層の数は、行の表(`name_hearing_round2_rows_20261112.csv`)の値をそのまま引いて答えごとに足す(重なりは 0層と確かめ済み)

    ./venv/bin/python tools/census/name_hearing_round2_answers.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from name_hearing_round2_sheet import OUT_CSV as ROWS_CSV, OUT_XLSX  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANSWERED = ("data/census_full/hearing_layer_names_round2_answered_20261112.xlsx",
            "d86dd3b1297360311644441e3ad635380ef92907dc4be2e712b1f6182651fd82", "2026-11-12")
OUT = "data/census_full/name_hearing_round2_answers_20261112.csv"
TAG = "round2_any_page_exclude"


def main() -> int:
    from openpyxl import load_workbook
    print("[層名の聞き取り 第2回の答え(問い: どの頁でも絵から除外してよいか)]")
    path, sha, as_of = ANSWERED
    got = hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()
    if got != sha:
        print(f"    **sha256 が違う** {got[:16]}")
        return 1
    a = list(load_workbook(os.path.join(ROOT, path))["名前"].iter_rows(values_only=True))
    o = list(load_workbook(os.path.join(ROOT, OUT_XLSX))["名前"].iter_rows(values_only=True))
    if len(a) != len(o) or any(x[:7] != y[:7] for x, y in zip(a, o)):
        print("    **固定の列が作った表と違う** —— 止める")
        return 1
    print(f"    答えの表 sha256 {got[:16]}({as_of})/ 固定の列は {len(o) - 1} 行とも作った表と一致")
    rows = list(csv.DictReader(open(os.path.join(ROOT, ROWS_CSV), encoding="utf-8")))
    out = []
    n_ans = collections.Counter()
    n_lay = collections.Counter()
    for r, x in zip(rows, a[1:]):
        ans = x[7] or "未記入"
        out.append(dict(round=TAG, answered=as_of, rank=r["rank"], key=r["key"], layer_name=r["layer_name"], v_layer_kind=r["v_layer_kind"],
                        n_layers=r["n_layers"], answer=ans, comment=x[8] or ""))
        n_ans[ans] += 1
        n_lay[ans] += int(r["n_layers"])
    for k, v in n_ans.most_common():
        print(f"    {k}: {v} 行 / 層 {n_lay[k]:,}")
    kinds = collections.Counter((o_["answer"], o_["v_layer_kind"]) for o_ in out)
    print("    答え × 種別(行): " + " / ".join(f"{a_[:2]}×{k} {v}" for (a_, k), v in sorted(kinds.items())))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"    → `{OUT}`(非公開。層名を含む)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
