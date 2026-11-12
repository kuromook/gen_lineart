#!/usr/bin/env python
"""**便A の答えを束ごとに取り込む**(採点はしない)。

**答えを見る前に決めたこと**(E `note_census_exclude_axis_and_20row_split_20261110.md` §2):
- **便A の棒は5束がそろってから読む。** この器具は取り込むだけで、名前の当て方の的中・取りこぼしを印字しない
- 束ごとに、固定の列(頁・上から・層の名前・フォルダー・表示・種類)が作った表と全行一致することを確かめる(違えば止める)
- 空欄の段は「未記入」と数える(11-09 の「段の線ではない」と数える読み方は、頁に「済」がある表のためのもの。この表には頁のシートが無い)
- 作者のコメントは逐語で残す
- **E の決定(11-11 `note_census_a1_keep_choices_20261111.md`)**: 便A の選択肢は変えない。「分からない」は書き換えず、
  画素の列(`empty`、層の役割の表 v2)で「分からない∧空 / 分からない∧空でない」に分けて印字する

    ./venv/bin/python tools/census/stage_hearing_round2_a_answers.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_hearing_round2_a_sheets import BUNDLES, KEY_OUT, XLSX  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANSWERED = {"A-1": ("data/census_full/hearing_round2_a_bundleA-1_answered_20261111.xlsx",
                    "6b8778d46b4566243c8b70eb39ebb9bf52b49e1097aaaeea200dcb932b6bb891", "2026-11-11"),
            "A-2": ("data/census_full/hearing_round2_a_bundleA-2_answered_20261112.xlsx",
                    "028fdab5ec97717c7c3e859c9101e50b2536dd22ffeb01385c7d8083caa12499", "2026-11-12"),
            "A-3": ("data/census_full/hearing_round2_a_bundleA-3_answered_20261112.xlsx",
                    "76aca5c747ec87bb1193b1d14289413c47b37a5bbf289bb7e1c1ac26025ba88a", "2026-11-12")}
OUT = "data/census_full/hearing_round2_a_answers.csv"
TABLE = "data/census_full/layer_role_table_20261110_v2.csv"


def main() -> int:
    from openpyxl import load_workbook
    print("[便A の答えの取り込み(採点しない。棒は5束そろってから)]")
    key = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(ROOT, KEY_OUT), encoding="utf-8")):
        key[r["bundle"]].append(r)
    empty = {(r["page_path"], r["layer_id"]): r["empty"]
             for r in csv.DictReader(open(os.path.join(ROOT, TABLE), encoding="utf-8"))}
    out = []
    for b, _ in BUNDLES:
        if b not in ANSWERED:
            print(f"    {b}: 未回収({len(key[b])}行)")
            continue
        path, sha, as_of = ANSWERED[b]
        got = hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()
        if got != sha:
            print(f"    **{b}: sha256 が違う** {got[:16]}")
            return 1
        a = list(load_workbook(os.path.join(ROOT, path))["層"].iter_rows(values_only=True))
        o = list(load_workbook(os.path.join(ROOT, XLSX.format(b=b)))["層"].iter_rows(values_only=True))
        if len(a) != len(o) or any(x[:8] != y[:8] for x, y in zip(a, o)):
            print(f"    **{b}: 固定の列が作った表と違う**")
            return 1
        c = collections.Counter()
        for k, r in zip(key[b], a[1:]):
            if (int(k["item"]), int(k["pos"])) != (r[0], r[3]):
                print(f"    **{b}: 鍵と表の並びが違う** {k['item']},{k['pos']} ≠ {r[0]},{r[3]}")
                return 1
            stage = r[8] if r[8] else "未記入"
            c[stage] += 1
            out.append(dict(bundle=b, item=k["item"], pos=k["pos"], page_path=k["page_path"], layer_id=k["layer_id"],
                            author_stage=stage, author_comment=r[9] or "", empty=empty[(k["page_path"], k["layer_id"])],
                            as_of=as_of))
        print(f"    {b}: {len(a) - 1}行(sha256 {sha[:16]})→ " + " / ".join(f"{s} {n}" for s, n in c.most_common()))
    dk = [x for x in out if x["author_stage"] == "分からない"]
    print(f"    分からない {len(dk)} → ∧空 {sum(x['empty'] == '1' for x in dk)} / ∧空でない {sum(x['empty'] == '0' for x in dk)}"
          f"(答えは書き換えない)")
    # 11-11 の印字「空の層 全体 1」は述語が無く、97行の数と読まれた(E が訂正: 97行に5)。母集団を名前で書いて両方出す
    allk = [k for b_ in key for k in key[b_]]
    print(f"    空の層: **回収した束の行({len(out)}行)の中で {sum(x['empty'] == '1' for x in out)}** / "
          f"**便A の全行({len(allk)}行)の中で {sum(empty[(k['page_path'], k['layer_id'])] == '1' for k in allk)}**(束ごと "
          + " / ".join(f"{b_} {sum(empty[(k['page_path'], k['layer_id'])] == '1' for k in key[b_])}" for b_, _ in BUNDLES) + ")")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"    → `{OUT}`(非公開、{len(out)}行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
