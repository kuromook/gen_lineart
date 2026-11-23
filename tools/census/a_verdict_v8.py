#!/usr/bin/env python
"""**a の判定 v8** —— ベースマスクの裁定を当てる(Track E の通達 `note_census_e_a_v8_base_mask_applied_20261123.md`)。

**数を見る前に決めたこと**:
- 作者の裁定(11-23): ベースマスクは「それ自体が絵をもたない」/ 名前 = rendering_base の3つ・bg base・base(census の問い)/
  bg base の「清書」の1層にも「この層にも当てる」(census の問い)
- `is_base_mask_20261123.csv` の `is_base_mask = 1` の層 → `template_excluded`。**v7 で既に落ちていた層は理由を書き換えない**(前の理由を残し、
  列 `is_base_mask` で重なりが分かる)。動いた層の理由 `条項:ベースマスク(名前)`、出所 `a_source_v8` に裁定を書く
- **札であって合成からの削除ではない**(作者「文脈によって消し、文脈によって使う」)。土台の札 `a_is_clipping_base` は別の列
- E の数 219,645 / 437,425 と突き合わせる。前後の数を v7 → v8 で、名前を付けて印字する(規律10)

    ./venv/bin/python tools/census/a_verdict_v8.py
"""
import collections
import csv
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V7 = "data/census_full/a_verdict_v7_20261116.csv"
BM = "data/census_full/is_base_mask_20261123.csv"
BM_SHA = "3df50b2585d54273"
OUT = "data/census_full/a_verdict_v8_20261123.csv"
DROP = "条項:ベースマスク(名前)"
SRC = "作者の裁定 11-23: ベースマスクはそれ自体が絵をもたない。名前は rendering_base の3つ・bg base・base(census の問い)"
E_COUNT = 219645


def main() -> int:
    print("[a の判定 v8(ベースマスクの裁定)]")
    if not hashlib.sha256(open(os.path.join(ROOT, BM), "rb").read()).hexdigest().startswith(BM_SHA):
        print(f"    **sha256 が違う** {BM}"); return 1
    bm = {(r["page_path"], r["layer_id"]) for r in csv.DictReader(open(os.path.join(ROOT, BM), encoding="utf-8")) if r["is_base_mask"] == "1"}
    out, moved = [], collections.Counter()
    for r in csv.DictReader(open(os.path.join(ROOT, V7), encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        v, why, src = r["a_verdict_v7"], r["a_rule_v7"], ""
        if k in bm and v != "template_excluded":
            moved[v] += 1
            v, why, src = "template_excluded", DROP, SRC
        out.append(dict(page_path=k[0], layer_id=k[1], a_verdict_v8=v, a_rule_v8=why, a_source_v8=src,
                        is_base_mask=int(k in bm), a_rule_v7=r["a_rule_v7"], keep_name=r["keep_name"]))
    n = len(out)
    b = sum(1 for r in csv.DictReader(open(os.path.join(ROOT, V7), encoding="utf-8")) if r["a_verdict_v7"] == "template_excluded")
    a = sum(x["a_verdict_v8"] == "template_excluded" for x in out)
    print(f"    ベースマスク {len(bm):,} 層(a の母集団の中 {sum(x['is_base_mask'] for x in out):,})/ v7 で既に落ちていた {sum(1 for x in out if x['is_base_mask'] and x['a_rule_v8'] != DROP):,}")
    print("    既に落ちていた層の v7 の理由: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(
        x["a_rule_v7"] for x in out if x["is_base_mask"] and x["a_rule_v8"] != DROP).items()))
    print("    動いた層の v7 の状態: " + " / ".join(f"{k} {v:,}" for k, v in moved.items()))
    print(f"    落とす: v7(11-16){b:,} / {n:,} = {b / n:.4f} → v8(11-23){a:,} / {n:,} = {a / n:.4f}(動いた {a - b:,})")
    print(f"    E の数 {E_COUNT:,} と: " + ("一致" if a == E_COUNT else f"**食い違い** census {a:,} / E {E_COUNT:,}"))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"    → `{OUT}`(非公開、{n:,}行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
