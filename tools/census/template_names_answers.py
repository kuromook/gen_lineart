#!/usr/bin/env python
"""**テンプレートの層名の聞き取りの答えを取り込み、除外の名前の表を作る**(a: 決定論の除外)。

**答えを見る前に決めたこと**(`template_candidates.py` の札、2026-11-11):
- 作者が「テンプレート」と答えた名前**だけ**を除外の名前の表に入れる。census は自分で足さない(許可表と同じ考え方)
- 固定の列(順〜一致の割合)が作った表と全行一致することを確かめる(違えば止める)。空欄は「未記入」
- 作者のコメントは逐語で残す
- **表の単位は (層名, 種別) の完全一致。** 作者が見た行は種別つきなので、同じ名前の別の種別の層は答えの外に置き、
  数を印字する(黙って含めない・黙って落とさない)
- 名前の表が当てる層の数は、全層の棚卸し(`layer_name_inventory_20261110.csv`)から引いて印字する

    ./venv/bin/python tools/census/template_names_answers.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template_candidates import ANS, OUT_CSV as CAND_CSV, OUT_XLSX  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANSWERED = ("data/census_full/hearing_template_names_answered_20261111.xlsx",
            "3e2e799b86d301005f67e9a7c9d474438d1f9ae16257650ca456164b30dfd8f8", "2026-11-11")
INVENTORY = "data/census_full/layer_name_inventory_20261110.csv"
OUT_ANS = "data/census_full/template_names_answers_20261111.csv"
OUT_EXCL = "data/census_full/template_exclusion_names_20261111.csv"


def main() -> int:
    from openpyxl import load_workbook
    print("[テンプレートの層名の答え(a: 決定論の除外)]")
    path, sha, as_of = ANSWERED
    got = hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()
    if got != sha:
        print(f"    **sha256 が違う** {got[:16]}")
        return 1
    a = list(load_workbook(os.path.join(ROOT, path))["名前"].iter_rows(values_only=True))
    o = list(load_workbook(os.path.join(ROOT, OUT_XLSX))["名前"].iter_rows(values_only=True))
    if len(a) != len(o) or any(x[:9] != y[:9] for x, y in zip(a, o)):
        print("    **固定の列が作った表と違う**")
        return 1
    cand = {int(r["rank"]): r for r in csv.DictReader(open(os.path.join(ROOT, CAND_CSV), encoding="utf-8"))}
    inv = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(ROOT, INVENTORY), encoding="utf-8")):
        inv[r["layer_name"]][r["v_layer_kind"]] = r
    out, c = [], collections.Counter()
    for r in a[1:]:
        k = cand[r[0]]
        ans = r[9] or "未記入"
        if ans not in ANS + ["未記入"]:
            print(f"    **選択肢に無い答え** 順 {r[0]}: {ans}")
            return 1
        c[ans] += 1
        out.append(dict(rank=r[0], layer_name=k["name"], v_layer_kind=k["kind"], author_answer=ans,
                        author_comment=(r[10] or "").strip(), as_of=as_of))
    print(f"    {len(out)}行(sha256 {sha[:16]})→ " + " / ".join(f"{s} {n}" for s, n in c.most_common()))
    with open(os.path.join(ROOT, OUT_ANS), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    ex = [x for x in out if x["author_answer"] == ANS[0]]
    rows = []
    for x in ex:
        hit = inv[x["layer_name"]].get(x["v_layer_kind"])
        other = {kd: int(v["n_layers"]) for kd, v in inv[x["layer_name"]].items() if kd != x["v_layer_kind"]}
        rows.append(dict(layer_name=x["layer_name"], v_layer_kind=x["v_layer_kind"], n_layers=hit["n_layers"],
                         n_pages=hit["n_pages"], n_clip_pages=hit["n_clip_pages"], n_lip_pages=hit["n_lip_pages"],
                         n_split_units=hit["n_split_units"],
                         same_name_other_kinds_not_covered=" ".join(f"{kd}:{n}" for kd, n in sorted(other.items())),
                         author_comment=x["author_comment"], source=f"author hearing {as_of}"))
    print(f"    **除外の名前の表: {len(rows)} 行**((層名, 種別) の完全一致。作者が「テンプレート」と答えた行だけ)")
    for i, d in enumerate(rows, 1):
        print(f"      {i}. 種別 {d['v_layer_kind']}: 層 {int(d['n_layers']):,} / 頁 {int(d['n_pages']):,}"
              f"(.clip {int(d['n_clip_pages']):,} / .lip {int(d['n_lip_pages']):,})/ 単位 {d['n_split_units']}"
              + (f" / 同じ名前の別の種別(答えの外){d['same_name_other_kinds_not_covered']}"
                 if d["same_name_other_kinds_not_covered"] else ""))
    print(f"    当てる層の和 {sum(int(d['n_layers']) for d in rows):,}(棚卸しの母集団 = 恒久除外の頁を外した全層 437,425層。"
          "フォルダーの子は数えていない)")
    with open(os.path.join(ROOT, OUT_EXCL), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"    → `{OUT_ANS}` / `{OUT_EXCL}`(非公開。層名を含む)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
