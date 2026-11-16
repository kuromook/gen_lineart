#!/usr/bin/env python
"""**a の判定 v7** —— 当て順を変える(Track E の頼み `note_census_ruling_shiryou_folder_sketch_is_shiryou_20261116.md`)。

**数を見る前に決めたこと**:
- 作者の裁定(11-16、E の問い「資料のフォルダーに入った下絵は、資料の側ですか。それとも下絵の側ですか」への答え): **「資料」**
- v6 の一群2(名前に 下絵・ラフ・sketch・線画)のうち **祖先に『資料』を含む層** → `template_excluded`、理由
  `条項:image_material(名前か祖先が資料)`、出所の列 `a_source_v7` に裁定を書く。E の数: 85層、215,162 → 215,247
- 一群2 の残り(祖先に『資料』を含まない)は落とさない側のまま。E の数: 8層
- 鍵(`image_material_hidden_groups_key_20261116.csv`)に列 `ancestors_has_shiryou` を足した版を別の名前で出す(名前は出さない)
- **取り込み画像以外の層には当てない**(E: 母集団が別。規律1)
- 前後の数を v6 → v7 で、名前を付けて印字する(規律10)

    ./venv/bin/python tools/census/a_verdict_v7.py
"""
import collections
import csv
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V6 = "data/census_full/a_verdict_v6_20261116.csv"
V6_SHA = "3659148ba2f52efe"
KEY = "data/census_full/image_material_hidden_groups_key_20261116.csv"
KEY_SHA = "0e6afbc25386ca93"
ST = "data/census_full/image_material_structure_20261114.csv"
OUT = "data/census_full/a_verdict_v7_20261116.csv"
KEY2 = "data/census_full/image_material_hidden_groups_key_v2_20261116.csv"
DROP = "条項:image_material(名前か祖先が資料)"
SRC = "作者の裁定 11-16: 資料のフォルダーに入った下絵は「資料」(E 経由)"


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def write(path, rows):
    with open(os.path.join(ROOT, path), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    print("[a の判定 v7(当て順: 祖先の資料を先に)]")
    for p, s in ((V6, V6_SHA), (KEY, KEY_SHA)):
        if not sha(p).startswith(s):
            print(f"    **sha256 が違う** {p}")
            return 1
    st = {(r["page_path"], r["layer_id"]): r["ancestors"] for r in csv.DictReader(open(os.path.join(ROOT, ST), encoding="utf-8"))}
    key = list(csv.DictReader(open(os.path.join(ROOT, KEY), encoding="utf-8")))
    move = set()
    for r in key:
        k = (r["page_path"], r["layer_id"])
        r["ancestors_has_shiryou"] = int("資料" in st[k])
        if r["name_group"] == "2" and r["ancestors_has_shiryou"]:
            move.add(k)
    g2 = [r for r in key if r["name_group"] == "2"]
    print(f"    一群2 {len(g2)}: 祖先に資料 {len(move)} → 落とす側 / 残り {len(g2) - len(move)} → 落とさない側のまま")
    out = []
    for r in csv.DictReader(open(os.path.join(ROOT, V6), encoding="utf-8")):
        k = (r["page_path"], r["layer_id"])
        v, why, src = r["a_verdict_v6"], r["a_rule_v6"], ""
        if k in move:
            v, why, src = "template_excluded", DROP, SRC
        out.append(dict(page_path=k[0], layer_id=k[1], a_verdict_v7=v, a_rule_v7=why, a_source_v7=src, a_rule_v6=r["a_rule_v6"], keep_name=r["keep_name"]))
    n = len(out)
    b = sum(1 for r in csv.DictReader(open(os.path.join(ROOT, V6), encoding="utf-8")) if r["a_verdict_v6"] == "template_excluded")
    a = sum(x["a_verdict_v7"] == "template_excluded" for x in out)
    d = sum(x["a_rule_v7"] == DROP for x in out)
    print(f"    落とす: v6(11-16){b:,} / {n:,} = {b / n:.4f} → v7(11-16){a:,} / {n:,} = {a / n:.4f}(動いた {a - b})")
    print(f"    資料で落とす {d:,} / 非表示から戻したまま {sum(x['a_rule_v7'].startswith('外した:image_material(見えない)') for x in out):,}")
    print("    動いた層の v6 の状態: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(
        r["a_verdict_v6"] for r in csv.DictReader(open(os.path.join(ROOT, V6), encoding="utf-8")) if (r["page_path"], r["layer_id"]) in move).items()))
    write(OUT, out)
    write(KEY2, key)
    print(f"    → `{OUT}`(非公開、{n:,}行)/ 鍵 v2 `{KEY2}`({len(key):,}行、列 ancestors_has_shiryou を足した)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
