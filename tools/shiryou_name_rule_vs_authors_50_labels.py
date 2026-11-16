#!/usr/bin/env python3
"""**新しい聞き取りの要らない検定**(規律13): 『名前か祖先が資料』の規則が、作者の『資料』の判定を再現するか。

材料は**既に在るもの**だけ —— (イ) の50層の答え(作者、11-16)と、census の一群の列(11-16)。
**この結果で名前の規則を撤回しうる**ことは、層を見る前に登録した
(`hidden_reason_redecided_and_shiryou_name_clause_20261116.validation_on_existing_labels`)。

**母集団の違いを先に書く** —— (イ) の50層は**見えている**取り込み画像の層で、名前の規則を当てる先は
**非表示の 4,356層**である。**ここで測れるのは『名前の語が作者の区分を再現するか』だけ**で、
非表示の母集団での当たり方は言えない(**規律1: 母集団は構成の一部**)。

**印字するもの**: 2×2 の表 / 的中率(精度)と取りこぼし(再現率)/ 一群ごと / 食い違った層の作者の一言。
"""
import csv
import io
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ANS = HERE / "data/role_predictor/image_material_check50b_answers_20261116.jsonl"
GRP = HERE / "inbox/image_material_check50b_groups_20261116.csv"
LABEL = {"part_of_picture": "絵の一部", "reference": "資料",
         "__none__": "どれでもない", "__unknown__": "分からない"}


def main():
    last = {}
    for line in io.open(ANS, encoding="utf-8"):
        r = json.loads(line)
        last[r["item_id"]] = r
    grp = {r["item_id"]: r for r in csv.DictReader(io.open(GRP, encoding="utf-8"))}
    assert len(last) == 50 and len(grp) == 50 and set(last) == set(grp)

    cell = Counter()
    rows = []
    for k, r in sorted(last.items()):
        says = grp[k]["name_rule_says_shiryou"] == "1"
        ans = LABEL.get(r["value"], r["value"])
        is_shiryou = ans == "資料"
        cell[(says, is_shiryou)] += 1
        rows.append((k, says, ans, grp[k]["name_group_label"], (r.get("comment") or "").strip()))

    tp = cell[(True, True)]; fp = cell[(True, False)]
    fn = cell[(False, True)]; tn = cell[(False, False)]
    print("## 2×2(作者の答えが『資料』か 対 名前の規則が『資料』と言うか)")
    print(f"   規則=資料 かつ 作者=資料      **{tp}**")
    print(f"   規則=資料 だが 作者=資料でない **{fp}**  ← **絵などを落としてしまう側**")
    print(f"   規則=資料でない だが 作者=資料 **{fn}**  ← 取りこぼし")
    print(f"   規則=資料でない かつ 作者=資料でない {tn}")
    print(f"   合計 {tp+fp+fn+tn} / 規則が資料と言う層 {tp+fp} / 作者が資料と答えた層 {tp+fn}")
    print("\n## 的中率と取りこぼし(**規律13 の印字**)")
    print(f"   **的中率(規則が資料と言った層のうち、作者も資料): {tp}/{tp+fp} = "
          + (f"{tp/(tp+fp):.3f}" if tp + fp else "—") + "**")
    print(f"   **取りこぼし(作者が資料と答えた層のうち、規則が拾えなかった): {fn}/{tp+fn} = "
          + (f"{fn/(tp+fn):.3f}" if tp + fn else "—") + "**")
    print(f"   再現率 {tp}/{tp+fn} = " + (f"{tp/(tp+fn):.3f}" if tp + fn else "—"))

    print("\n## 一群ごと")
    for lab in sorted(set(g["name_group_label"] for g in grp.values())):
        ks = [k for k in last if grp[k]["name_group_label"] == lab]
        c = Counter(LABEL.get(last[k]["value"], last[k]["value"]) for k in ks)
        says = grp[ks[0]]["name_rule_says_shiryou"] == "1"
        print(f"   [{'規則=資料' if says else '規則=資料でない'}] {lab}: {len(ks)}層 —— "
              + " / ".join(f"{a} {b}" for a, b in c.most_common()))

    print("\n## 規則が『資料』と言ったのに、作者が資料と答えなかった層(逐語)")
    for k, says, ans, lab, com in rows:
        if says and ans != "資料":
            print(f"   {k} [{ans} / {lab}]: {com or '(一言なし)'}")

    print("\n## 母集団の違い(**先に書いた限界**)")
    print("   (イ) の50層は**見えている**取り込み画像の層。名前の規則を当てる先は**非表示の 4,356層**。")
    print("   **ここで測れるのは『名前の語が作者の区分を再現するか』だけ** —— 非表示の母集団での")
    print("   当たり方は言えない(規律1: 母集団は構成の一部)。")


main()
