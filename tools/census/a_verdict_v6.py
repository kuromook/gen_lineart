#!/usr/bin/env python
"""**a の判定 v6 と、取り込み画像の層の『名前の一群』の列** —— Track E の頼み(`note_census_hidden_reason_redecided_and_name_clause_20261116.md`、
`note_census_validation_approved_need_key_of_3557_20261116.md`)。規則は E のもの(作者の裁定3件を受けて層を見ずに登録)。census は当てるだけ。

**数を見る前に決めたこと**(一群は 11-16 に census が作者へ示したものと同じ。**名前の文字で分けた記述で、的中率は測っていない**。規律13):
- 一群(上から先に当たったもの1つ。**この順のまま**):
  1. 名前が `資料化下絵` で始まる(作者の裁定 11-16: 資料の側)
  2. 名前に `下絵` / `ラフ` / `sketch`(大小) / `線画` を含む
  3. 名前か祖先に `資料` を含む
  4. 祖先に `下絵` / `ラフ` / `sketch` / `線画` を含む(3 に当たらなかったもの)
  5. どれも無い
  - **順の効き目を印字する**: 2 に入った層のうち祖先に `資料` を含む層の数(3 を先にすれば 3 に入る層)
- **v6**: v5 の理由 `条項:image_material(見えない)` の層のうち、一群 1・3 → `template_excluded`、理由 `条項:image_material(名前か祖先が資料)`。
  一群 2・4・5 → 11-12 の判定の状態に戻し、理由 `外した:image_material(見えない)(作者の裁定 11-16: 非表示は落とす理由ではない)`。
  E の数: 落とす 3,557 / 戻す 799、215,961 → 215,162
- **(イ) の50層**(見えている層)にも同じ一群を当てた列を出す(E の検定用。名前は出さない)
- 鍵: 4,356層の `page_path` / `layer_id` / 一群 / 作品フォルダー(年の札\\作品)/ 分割の単位 / 年の札。**層の名前は出さない**(条例11)
- 前後の数を v5 → v6 で、名前を付けて印字する(規律10)

    ./venv/bin/python tools/census/a_verdict_v6.py
"""
import collections
import csv
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = "data/census_full/a_verdict_20261112.csv"
V5 = "data/census_full/a_verdict_v5_20261116.csv"
V5_SHA = "b27ebfa8c8637be9"
ST = "data/census_full/image_material_structure_20261114.csv"
B50 = "data/census_full/image_material_check50b_20261115_replaced.csv"
B50_SHA = "f1c86d3948ab50ea"
OUT = "data/census_full/a_verdict_v6_20261116.csv"
KEY = "data/census_full/image_material_hidden_groups_key_20261116.csv"
KEY50 = "data/census_full/image_material_check50b_groups_20261116.csv"
HIDDEN = "条項:image_material(見えない)"
DROP6 = "条項:image_material(名前か祖先が資料)"
BACK6 = "外した:image_material(見えない)(作者の裁定 11-16: 非表示は落とす理由ではない)"
SK = re.compile(r"下絵|ラフ|[Ss][Kk][Ee][Tt][Cc][Hh]|線画")
NAMES = {1: "名前が資料化下絵", 2: "名前に下絵・ラフ・sketch・線画", 3: "名前か祖先に資料", 4: "祖先に下絵・ラフ・sketch・線画", 5: "どれも無い"}


def group(name, anc):
    if name.startswith("資料化下絵"):
        return 1
    if SK.search(name):
        return 2
    if "資料" in name or "資料" in anc:
        return 3
    if SK.search(anc):
        return 4
    return 5


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def write(path, rows):
    with open(os.path.join(ROOT, path), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    print("[a の判定 v6 / 取り込み画像の層の名前の一群]")
    for p, s in ((V5, V5_SHA), (B50, B50_SHA)):
        if not sha(p).startswith(s):
            print(f"    **sha256 が違う** {p}")
            return 1
    con = db.connect()
    un = split_units.units(con)
    st = {(r["page_path"], r["layer_id"]): r for r in csv.DictReader(open(os.path.join(ROOT, ST), encoding="utf-8"))}
    v1 = {(r["page_path"], r["layer_id"]): r["a_verdict"] for r in csv.DictReader(open(os.path.join(ROOT, V1), encoding="utf-8"))}
    v5 = list(csv.DictReader(open(os.path.join(ROOT, V5), encoding="utf-8")))
    out, key = [], []
    swap = 0
    for r in v5:
        k = (r["page_path"], r["layer_id"])
        v, why = r["a_verdict_v5"], r["a_rule_v5"]
        if why == HIDDEN:
            s = st[k]
            g = group(s["name"], s["ancestors"])
            if g == 2 and "資料" in s["ancestors"]:
                swap += 1
            parts = k[0].split("\\")
            key.append(dict(page_path=k[0], layer_id=k[1], name_group=g, name_group_label=NAMES[g],
                            work_folder="\\".join(parts[2:4]), split_unit=un[wd(k[0])], year_label=parts[2],
                            hidden_by={("1", "0"): "ancestor", ("0", "0"): "both", ("0", "1"): "self"}[(s["self_shown_db"], s["ancestors_shown_db"])]))
            if g in (1, 3):
                v, why = "template_excluded", DROP6
            else:
                v, why = v1[k], BACK6
        out.append(dict(page_path=k[0], layer_id=k[1], a_verdict_v6=v, a_rule_v6=why, a_rule_v5=r["a_rule_v5"], keep_name=r["keep_name"]))
    c = collections.Counter(x["name_group"] for x in key)
    print("    『見えない』の層 {:,}: ".format(len(key)) + " / ".join(f"{g} {NAMES[g]} {c[g]:,}" for g in sorted(c)))
    print(f"    順の効き目: 一群2 のうち祖先に『資料』を含む層 {swap}(『資料』を先に当てれば一群3 に入る。E の 3,557 はこの順で数えた数)")
    n = len(out)
    b = sum(r["a_verdict_v5"] == "template_excluded" for r in v5)
    a = sum(x["a_verdict_v6"] == "template_excluded" for x in out)
    print(f"    落とす: v5(11-16){b:,} / {n:,} = {b / n:.4f} → v6(11-16){a:,} / {n:,} = {a / n:.4f}"
          f"(資料で落とす {sum(x['a_rule_v6'] == DROP6 for x in out):,} / 戻す {sum(x['a_rule_v6'] == BACK6 for x in out):,})")
    print("    戻した層の状態(11-12 の判定): " + " / ".join(f"{k} {v:,}" for k, v in collections.Counter(
        x["a_verdict_v6"] for x in out if x["a_rule_v6"] == BACK6).most_common()))
    d = collections.Counter(x["year_label"] for x in key if x["name_group"] in (1, 3))
    print("    資料で落とす層の年の札: " + " / ".join(f"{y} {d[y]:,}" for y in sorted(d)))
    k50 = []
    for r in csv.DictReader(open(os.path.join(ROOT, B50), encoding="utf-8")):
        s = st.get((r["page_path"], r["layer_id"]))
        g = group(s["name"], s["ancestors"]) if s else ""
        k50.append(dict(pos=r["pos"], item_id=f"im50b_{int(r['pos']):02d}", page_path=r["page_path"], layer_id=r["layer_id"],
                        name_group=g, name_group_label=NAMES.get(g, "構造の表に無い"), name_rule_says_shiryou=int(g in (1, 3)) if g else ""))
    c50 = collections.Counter(x["name_group"] for x in k50)
    print("    (イ) の50層の一群: " + " / ".join(f"{g} {NAMES.get(g, '構造の表に無い')} {c50[g]}" for g in sorted(c50, key=str))
          + f" → 名前の規則が資料と言う {sum(x['name_rule_says_shiryou'] == 1 for x in k50)}")
    write(OUT, out)
    write(KEY, key)
    write(KEY50, k50)
    print(f"    → `{OUT}`(非公開、{n:,}行)/ 鍵 `{KEY}`({len(key):,}行)/ (イ) `{KEY50}`(50行)。層の名前は鍵に出さない")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
