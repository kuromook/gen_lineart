#!/usr/bin/env python
"""**a の判定 v11** —— `下地 1 のコピー`・`下地 2 のコピー` を当てる(作者の答え 11-26「同じ扱いでよい」、census の問い。
Track E の通達 `note_census_e_a_v11_and_2014_correction_20261126.md`)。

- 集合 = `is_base_mask` v3 で v2 から足した層(層名がちょうど上の2つ・フォルダーでない)。2層とも .lip(画素が無い、名前だけが根拠)
- v10 で未決なら `template_excluded`、理由 `条項:ベースマスク(名前:下地N のコピー)`。v10 は書き換えない
- 曝露(落とした土台と乗る層)を v10 → v11 で印字する。E の計算 7,250 / 22,845 → 7,252 / 22,849 と突き合わせる
- `下地白 ` は作者「わからない　一応直接見ておく」で入れない
"""
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
V10 = os.path.join(D, "a_verdict_v10_20261126.csv")
BM3 = os.path.join(D, "is_base_mask_v3_20261126.csv")
CB = os.path.join(D, "a_clipping_base_v3_v8_20261123.csv")
OUT = os.path.join(D, "a_verdict_v11_20261126.csv")
DROP = "条項:ベースマスク(名前:下地N のコピー)"
SRC = "作者の答え 11-26(census の問い): 下地 1 のコピー・下地 2 のコピーも下地N と同じ扱いでよい"
E_COUNT, E_EXPO = 219801, (7252, 22849)


def main() -> int:
    add = {(r["page_path"], r["layer_id"]) for r in csv.DictReader(open(BM3, encoding="utf-8"))
           if r["is_base_mask_v3"] == "1" and r["is_base_mask_v2"] != "1"}
    out, moved = [], 0
    for r in csv.DictReader(open(V10, encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        v, why, src = r["a_verdict_v10"], r["a_rule_v10"], r["a_source_v10"]
        if k in add and v != "template_excluded":
            moved += 1; v, why, src = "template_excluded", DROP, SRC
        out.append(dict(page_path=k[0], layer_id=k[1], a_verdict_v11=v, a_rule_v11=why, a_source_v11=src,
                        a_verdict_v10=r["a_verdict_v10"], is_base_mask_v3=int(r["is_base_mask_v2"] == "1" or k in add),
                        a_rule_v7=r["a_rule_v7"], keep_name=r["keep_name"]))
    n = len(out)
    c10 = collections.Counter(x["a_verdict_v10"] for x in out)
    c11 = collections.Counter(x["a_verdict_v11"] for x in out)
    print(f"[a の判定 v11] 足す層 {len(add)} / 動いた {moved}")
    for kk in ("template_excluded", "undecided", "not_excluded"):
        print(f"    {kk}: v10 {c10[kk]:,} → v11 {c11[kk]:,}")
    print(f"    落とす率: v10 {c10['template_excluded']:,} / {n:,} = {c10['template_excluded'] / n:.5f} → v11 {c11['template_excluded']:,} / {n:,} = {c11['template_excluded'] / n:.5f}")
    print(f"    E の数 {E_COUNT:,} と: " + ("一致" if c11["template_excluded"] == E_COUNT else "**食い違い**"))
    cb = {(r["page_path"], r["layer_id"]): int(r["n_clipped_layers"] or 0) for r in csv.DictReader(open(CB, encoding="utf-8"))
          if r["a_is_clipping_base"] == "1"}
    ex = {}
    for col in ("a_verdict_v10", "a_verdict_v11"):
        d = [x for x in out if (x["page_path"], x["layer_id"]) in cb and x[col] == "template_excluded"]
        ex[col] = (len(d), sum(cb[(x["page_path"], x["layer_id"])] for x in d))
    print(f"    曝露(落とした土台 / 乗る層): v10 {ex['a_verdict_v10'][0]:,} / {ex['a_verdict_v10'][1]:,} → v11 {ex['a_verdict_v11'][0]:,} / {ex['a_verdict_v11'][1]:,}"
          f"(E の計算 {E_EXPO[0]:,} / {E_EXPO[1]:,}: " + ("一致" if ex["a_verdict_v11"] == E_EXPO else "**食い違い**") + ")")
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"    → {OUT}({n:,}行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
