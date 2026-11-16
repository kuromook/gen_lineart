#!/usr/bin/env python
"""**a の層ごとの判定 v4 と v5** —— Track E の頼み(2026-11-15 / 11-16)。規則と棒は E のもの。census は当てるだけで、棒を読み直さない。

**数を見る前に決めたこと**:
- **v4**(E `note_census_second50_replaced_and_3d_held_out_of_empty_20261115.md`): 種別 `3d` の層で、v3 の理由が
  `条項:空(定義C)` の層を保留に回す。状態は 11-12 の判定(`a_verdict_20261112.csv`)に戻し、理由を
  「空だが保留(種別 3d(ミップマップの中身を確かめていない。調査中))」と書く。E の数: 216,303 → 216,301
- **v5**(E `note_census_second50_bar_read_rules_withdrawn_and_new_buttons_20261116.md`。(イ) の棒は10以上の帯 = 撤回):
  理由 `条項:image_material(出力に画素が0)` と `条項:image_material(測れない)` を外し、状態を 11-12 の判定に戻す。
  `a_rule_v5` に「外した:…」と書く。E の数: 216,301 → 215,961
- 作者の答えの表(`image_material_check50b_answers_20261116.jsonl`)の素の内訳を印字し、E の数と突き合わせる(層ごとに最後の行)
- 前後の数を v3 → v4 → v5 と全部、名前を付けて印字する(規律10。過去の版は撤回しない)
- **但し書き**: (イ) の群の中身は判定保留(どれでもない + 分からない が 10以上。E の棒)。v5 が言うのは「この2つの理由の規則は絵を落とす」まで

    ./venv/bin/python tools/census/a_verdict_v5.py
"""
import collections
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = "data/census_full/a_verdict_20261112.csv"
V3 = "data/census_full/a_verdict_v3_20261115.csv"
V3_SHA = "1b83b69eca63406a"
ANS = "data/census_full/image_material_check50b_answers_20261116.jsonl"
ANS_SHA = "6cccfba2d3f97a4"
OUT4 = "data/census_full/a_verdict_v4_20261116.csv"
OUT5 = "data/census_full/a_verdict_v5_20261116.csv"
EMPTY = "条項:空(定義C)"
HOLD3D = "空だが保留(種別 3d(ミップマップの中身を確かめていない。調査中))"
DROP5 = ("条項:image_material(出力に画素が0)", "条項:image_material(測れない)")


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def write(path, rows):
    with open(os.path.join(ROOT, path), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    print("[a の層ごとの判定 v4・v5]")
    for p, s in ((V3, V3_SHA), (ANS, ANS_SHA)):
        if not sha(p).startswith(s):
            print(f"    **sha256 が違う** {p}")
            return 1
    lines = [json.loads(x) for x in open(os.path.join(ROOT, ANS), encoding="utf-8") if x.strip()]
    last = {}
    for x in lines:
        last[x["item_id"]] = x
    print(f"    (イ) の答えの行 {len(lines)} → 層ごとの最後の行 {len(last)}。素の内訳: "
          + " / ".join(f"{k} {v}" for k, v in collections.Counter(x["label"] for x in last.values()).most_common())
          + "(E の数: 資料 17 / 絵の一部 15 / どれでもない 15 / 分からない 3)")
    con = db.connect()
    kind = {(r[0], str(r[1])): r[2] for r in con.execute("select page_path, layer_id, v_layer_kind from v_layer_kind").fetchall()}
    v1 = {(r["page_path"], r["layer_id"]): r["a_verdict"] for r in csv.DictReader(open(os.path.join(ROOT, V1), encoding="utf-8"))}
    v3 = list(csv.DictReader(open(os.path.join(ROOT, V3), encoding="utf-8")))
    n = len(v3)
    o4, o5 = [], []
    for r in v3:
        k = (r["page_path"], r["layer_id"])
        v, why = r["a_verdict_v3"], r["a_rule_v3"]
        if why == EMPTY and kind.get(k) == "3d":
            v, why = v1[k], HOLD3D
        o4.append(dict(page_path=k[0], layer_id=k[1], a_verdict_v4=v, a_rule_v4=why, a_rule_v3=r["a_rule_v3"], keep_name=r["keep_name"]))
        v5, why5 = v, why
        if why in DROP5:
            v5, why5 = v1[k], f"外した:{why[3:]}((イ) の50枚の二択 11-16。絵の一部 15/50 が棒 10以上に入った。E の規則)"
        o5.append(dict(page_path=k[0], layer_id=k[1], a_verdict_v5=v5, a_rule_v5=why5, a_rule_v4=why, keep_name=r["keep_name"]))
    c3 = sum(r["a_verdict_v3"] == "template_excluded" for r in v3)
    c4 = sum(r["a_verdict_v4"] == "template_excluded" for r in o4)
    c5 = sum(r["a_verdict_v5"] == "template_excluded" for r in o5)
    print(f"    落とす: v3(11-15){c3:,} / {n:,} = {c3 / n:.6f} → v4(11-16){c4:,} = {c4 / n:.6f}(保留に回した {c3 - c4})"
          f" → v5(11-16){c5:,} = {c5 / n:.6f}(戻した {c4 - c5:,})")
    print(f"    小数4桁: v3 {c3 / n:.4f} / v4 {c4 / n:.4f} / v5 {c5 / n:.4f}")
    print("    v4 で保留に回した層の状態(11-12 の判定): " + " / ".join(
        f"{k} {v}" for k, v in collections.Counter(r["a_verdict_v4"] for r in o4 if r["a_rule_v4"] == HOLD3D).most_common()))
    back = collections.Counter((r["a_rule_v4"], r["a_verdict_v5"]) for r in o5 if r["a_rule_v5"].startswith("外した") and r["a_rule_v4"] in DROP5)
    print("    v5 で戻した層(理由・11-12 の判定): " + " / ".join(f"{a} → {b} {v:,}" for (a, b), v in back.most_common()))
    left = collections.Counter(r["a_rule_v5"] for r in o5 if r["a_verdict_v5"] == "template_excluded" and "image_material" in r["a_rule_v5"])
    print("    v5 で image_material の理由で落ちている層: " + " / ".join(f"{a} {b:,}" for a, b in left.most_common()))
    write(OUT4, o4)
    write(OUT5, o5)
    print(f"    → `{OUT4}` / `{OUT5}`(非公開、{n:,}行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
