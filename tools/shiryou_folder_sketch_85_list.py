#!/usr/bin/env python3
"""**裁定で落とす側に移った85層**の一覧を作る —— 作者が1層ずつ裁定するための画を頼む(作者の求め 2026-11-17)。

**何者か**: 名前が `下絵 N`・`下絵 cut N`・`…ラフ N` の形で、**祖先に『資料』のフォルダーを持つ**非表示の
取り込み画像の層。**作者の形の裁定(2026-11-16、逐語「資料」)で、まとめて落とす側に移った85層**。
(ウ) の母集団 3,557 には**入っていない**(当て順で『下絵』より『資料』が先に当たるため)ので、
**この50枚の検定は覆っていない** —— そこを作者自身の裁定で埋める。

**全部を覆う**(標本ではない):
- 85層のうち **68層が代表の頁**、17層は**バイトが同じ写しの頁**の層で、**17層とも代表の側の同じ番号の層が
  85 の中に在る**(確かめた)。だから **68層を裁定すれば 85層が決まる**
- 画の二択は**1回50まで**(作者、2026-11-10)。**50 + 18 の2回**に分ける。
  **並びは種で混ぜる**(作品単位ごとに固まらないように。並びは押し方を動かしうる。条例2)

**これは測定ではなく裁定**(規律9-5)。それでも**先に書いておく**(規律10):
- 作者が『下絵』『絵の一部』と答えた層は**候補に戻す**。全部戻れば a は **215,247 → 215,162**(= v6 の数)、
  全部そのままなら **215,247 のまま**。**前後を両方、名前を付けて印字する**
- **形の裁定(資料のフォルダーの下の下絵は資料)を取り消すかどうかは、作者の言葉で決める** ——
  数で決めない。個別の裁定が形と食い違ったら、**形をどうするかを作者に問う**
"""
import csv
import hashlib
import io
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "inbox/image_material_hidden_groups_key_v3_20261117.csv"
OUT = HERE / "data/role_predictor/shiryou_folder_sketch_85_20261117.csv"
SEED = "20261117b"
ROUND1 = 50


def main():
    rows = list(csv.DictReader(io.open(SRC, encoding="utf-8")))
    g2 = [r for r in rows if r["name_group"] == "2"]
    anc = [r for r in g2 if r["ancestors_has_shiryou"] == "1"]
    print(f"一群2(名前に下絵・ラフ・sketch・線画) {len(g2)} → **祖先に『資料』 {len(anc)}**"
          f"(残り {len(g2) - len(anc)} は落とさない側のまま)")
    assert len(anc) == 85, len(anc)

    rep = [r for r in anc if r["is_representative"] == "1"]
    nonrep = [r for r in anc if r["is_representative"] == "0"]
    repkeys = {(r["page_path"], r["layer_id"]) for r in rep}
    covered = [r for r in nonrep
               if (r["representative_page_path"], r["layer_id"]) in repkeys]
    print(f"   代表の頁 {len(rep)} / 写しの頁 {len(nonrep)} —— "
          f"写しは **{len(covered)}/{len(nonrep)} が代表の側の同じ番号の層で覆われる**")
    assert len(covered) == len(nonrep), "覆われない写しが在る。別に出すこと"

    order = sorted(rep, key=lambda r: hashlib.sha256(
        f"{SEED}|{r['page_path']}|{r['layer_id']}".encode()).hexdigest())
    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos", "round", "page_path", "layer_id", "work_folder",
                    "split_unit", "year_label", "hidden_by", "covers_copies"])
        for i, r in enumerate(order, 1):
            copies = sum(1 for c in nonrep
                         if c["representative_page_path"] == r["page_path"]
                         and c["layer_id"] == r["layer_id"])
            w.writerow([i, 1 if i <= ROUND1 else 2, r["page_path"], r["layer_id"],
                        r["work_folder"], r["split_unit"], r["year_label"],
                        r["hidden_by"], copies])
    print(f"\n出力 {OUT}")
    print("   sha256 先頭", hashlib.sha256(OUT.read_bytes()).hexdigest()[:16])
    print(f"   {len(order)}層 = 第1便 {ROUND1} + 第2便 {len(order) - ROUND1}"
          f"(**覆う層は 85**)")
    print("   作品単位", dict(Counter(r["split_unit"].rsplit("\\", 2)[-1] for r in order)))
    print("   年の札", dict(Counter(r["year_label"] for r in order)))
    print("   非表示の形", dict(Counter(r["hidden_by"] for r in order)))
    print("   異なる頁", len(set(r["page_path"] for r in order)))
    print("   第1便の作品単位", dict(Counter(
        r["split_unit"].rsplit("\\", 2)[-1] for r in order[:ROUND1])))


main()
