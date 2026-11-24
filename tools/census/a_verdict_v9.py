#!/usr/bin/env python
"""**a の判定 v9** —— ベースマスクの名前の裁定から「調査中」の7層を外す(ユーザーの決定 2026-11-24、逐語「戻す」)。

**数を見る前に決めたこと**:
- 元: v8 で `条項:ベースマスク(名前)` により動いた層のうち、v7 の理由が「調査中(分析対象外)」の層。E の通達
  `note_census_e_v8_confirmed_and_rounding_correction_20261123.md` が名指しした 7層
- その層は v7 の判定と理由に戻す(`undecided` / 空だが保留・調査中)。出所 `a_source_v9` にユーザーの決定を書く
- 「調査中は分析対象外」(作者 10-29)が名前の裁定より上。**v8 は撤回しない**(規律10)。v8 → v9 の前後を名前を付けて印字する
- それ以外の層(v7 の理由が image_material の 2層を含む)は v8 のまま

    ./venv/bin/python tools/census/a_verdict_v9.py
"""
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V7 = "data/census_full/a_verdict_v7_20261116.csv"
V8 = "data/census_full/a_verdict_v8_20261123.csv"
OUT = "data/census_full/a_verdict_v9_20261124.csv"
BM = "条項:ベースマスク(名前)"
UI = "調査中"
SRC = "ユーザーの決定 11-24: 調査中の層は名前の裁定から外して調査中に戻す(調査中は分析対象外、作者 10-29)"


def main() -> int:
    print("[a の判定 v9(調査中の層を名前の裁定から戻す)]")
    v7 = {(r["page_path"], r["layer_id"]): r for r in csv.DictReader(open(os.path.join(ROOT, V7), encoding="utf-8"))}
    out, back = [], 0
    for r in csv.DictReader(open(os.path.join(ROOT, V8), encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        v, why, src = r["a_verdict_v8"], r["a_rule_v8"], r["a_source_v8"]
        if why == BM and UI in r["a_rule_v7"]:
            back += 1
            v, why, src = v7[k]["a_verdict_v7"], v7[k]["a_rule_v7"], SRC
        out.append(dict(page_path=k[0], layer_id=k[1], a_verdict_v9=v, a_rule_v9=why, a_source_v9=src,
                        a_verdict_v8=r["a_verdict_v8"], is_base_mask=r["is_base_mask"], a_rule_v7=r["a_rule_v7"], keep_name=r["keep_name"]))
    n = len(out)
    c8 = collections.Counter(x["a_verdict_v8"] for x in out)
    c9 = collections.Counter(x["a_verdict_v9"] for x in out)
    print(f"    戻した層 {back}(v8 で名前の裁定、v7 で調査中)/ 戻さずに v8 のまま: 名前の裁定で動いた残り "
          f"{sum(1 for x in out if x['a_rule_v9'] == BM):,}(うち v7 に理由の在った {sum(1 for x in out if x['a_rule_v9'] == BM and x['a_rule_v7']):,})")
    for k in ("template_excluded", "undecided", "not_excluded"):
        print(f"    {k}: v8(11-23){c8[k]:,} → v9(11-24){c9[k]:,}")
    print(f"    落とす率: v8 {c8['template_excluded']:,} / {n:,} = {c8['template_excluded'] / n:.4f} → v9 {c9['template_excluded']:,} / {n:,} = {c9['template_excluded'] / n:.5f}")
    print(f"    ベースマスクの層で落とさないもの: {sum(1 for x in out if x['is_base_mask'] == '1' and x['a_verdict_v9'] != 'template_excluded')}")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"    → `{OUT}`(非公開、{n:,}行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
