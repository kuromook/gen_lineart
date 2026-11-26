#!/usr/bin/env python
"""**a の判定 v12** —— `下地白 ` を当てる(作者の裁定 11-26、ファイルを開いて: 「あつかいとしては下地Nと同じでよい」。
Track E の通達 `note_census_e_a_v12_and_shitaji_is_not_a_3d_mark_20261126.md`)。

- 集合 = `is_base_mask` v4 で v3 から足した層(層名がちょうど `下地白 `、末尾の空白を畳まない)。描いた画素は全部純黒 —— 種類で決まり色では決まらない
- v11 で未決なら `template_excluded`。v11 は書き換えない。曝露を v11 → v12 で印字する(土台でないので動かない見込み)
"""
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
V11 = os.path.join(D, "a_verdict_v11_20261126.csv")
BM4 = os.path.join(D, "is_base_mask_v4_20261126.csv")
CB = os.path.join(D, "a_clipping_base_v3_v8_20261123.csv")
OUT = os.path.join(D, "a_verdict_v12_20261126.csv")
DROP = "条項:ベースマスク(名前:下地白)"
SRC = "作者の裁定 11-26(ファイルを開いて): 下地白 は植物のベースマスク、扱いは下地N と同じ(3D ではなくパターンブラシで描いたもの)"
E_COUNT, E_EXPO = 219802, (7252, 22849)


def main() -> int:
    add = {(r["page_path"], r["layer_id"]) for r in csv.DictReader(open(BM4, encoding="utf-8"))
           if r["is_base_mask_v4"] == "1" and r["is_base_mask_v3"] != "1"}
    out, moved = [], 0
    for r in csv.DictReader(open(V11, encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        v, why, src = r["a_verdict_v11"], r["a_rule_v11"], r["a_source_v11"]
        if k in add and v != "template_excluded":
            moved += 1; v, why, src = "template_excluded", DROP, SRC
        out.append(dict(page_path=k[0], layer_id=k[1], a_verdict_v12=v, a_rule_v12=why, a_source_v12=src,
                        a_verdict_v11=r["a_verdict_v11"], is_base_mask_v4=int(r["is_base_mask_v3"] == "1" or k in add),
                        a_rule_v7=r["a_rule_v7"], keep_name=r["keep_name"]))
    n = len(out)
    c11 = collections.Counter(x["a_verdict_v11"] for x in out)
    c12 = collections.Counter(x["a_verdict_v12"] for x in out)
    print(f"[a の判定 v12] 足す層 {len(add)} / 動いた {moved}")
    for kk in ("template_excluded", "undecided", "not_excluded"):
        print(f"    {kk}: v11 {c11[kk]:,} → v12 {c12[kk]:,}")
    print(f"    落とす率: v11 {c11['template_excluded']:,} / {n:,} = {c11['template_excluded'] / n:.5f} → v12 {c12['template_excluded']:,} / {n:,} = {c12['template_excluded'] / n:.5f}")
    print(f"    E の数 {E_COUNT:,} と: " + ("一致" if c12["template_excluded"] == E_COUNT else "**食い違い**"))
    cb = {(r["page_path"], r["layer_id"]): int(r["n_clipped_layers"] or 0) for r in csv.DictReader(open(CB, encoding="utf-8"))
          if r["a_is_clipping_base"] == "1"}
    ex = {}
    for col in ("a_verdict_v11", "a_verdict_v12"):
        d = [x for x in out if (x["page_path"], x["layer_id"]) in cb and x[col] == "template_excluded"]
        ex[col] = (len(d), sum(cb[(x["page_path"], x["layer_id"])] for x in d))
    print(f"    曝露(落とした土台 / 乗る層): v11 {ex['a_verdict_v11'][0]:,} / {ex['a_verdict_v11'][1]:,} → v12 {ex['a_verdict_v12'][0]:,} / {ex['a_verdict_v12'][1]:,}"
          f"(E の計算 {E_EXPO[0]:,} / {E_EXPO[1]:,}: " + ("一致" if ex["a_verdict_v12"] == E_EXPO else "**食い違い**") + ")")
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"    → {OUT}({n:,}行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
