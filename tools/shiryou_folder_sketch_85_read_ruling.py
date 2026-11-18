#!/usr/bin/env python3
"""**85層の1層ずつの裁定**を読む —— 答えを読む前に書いた器具(登録 `per_layer_ruling_ui_registered_before_images_20261117`)。

**これは測定ではなく裁定**(規律9-5)。棒は無い。**(ウ) の 0/50 も名前の規則も動かない。**
**作者の答えは書き換えない**(規律6)。層ごとに最後の行を採る。

**先に書いてある動き**: 『下絵』『絵の一部』と答えた層は**候補に戻す**。
- 全部戻れば a は **215,247 → 215,162**(= v6 の数)、全部そのままなら **215,247 のまま**
- **写しの層も一緒に動く**(代表の層1つが、一覧の `covers_copies` の数だけ写しを覆っている)
- **形の裁定(資料のフォルダーの下の下絵は資料)を取り消すかは、数ではなく作者の言葉で決める**

便は2回に分かれるので、**両方の答えを読み、足りない便があればそう書く**。
"""
import collections
import csv
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LIST = HERE / "data/role_predictor/shiryou_folder_sketch_85_20261117.csv"
# 画面を立てたときの実際の名前(器具を書いた時の仮の名前から直した。**配達の都合で、答えと独立**)
ANS = [HERE / "data/role_predictor/shiryou_folder_sketch_85_answers_r1_20261117.jsonl",
       HERE / "data/role_predictor/shiryou_folder_sketch_85_answers_r2_20261117.jsonl"]
CHECK = HERE / "data/role_predictor/shiryou_folder_sketch_im85_check_20261117.csv"

A_NOW, A_TOTAL = 215247, 437425
A_IF_ALL_RETURN = 215162            # = v6。85層が全部戻ったときの数
BACK = ("sketch_underdrawing", "part_of_picture")   # 候補に戻す答え
STAY = ("reference",)                                # 落としたままの答え
HELD = ("__none__", "__unknown__", "background_3d", "cutout_collage")


def main():
    rows = {r["pos"]: r for r in csv.DictReader(io.open(LIST, encoding="utf-8"))}
    last = {}
    for i, path in enumerate(ANS, 1):
        if not path.is_file():
            print(f"**{i}回目の答えがまだ無い**: {path.name}")
            continue
        n = 0
        for line in io.open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("item_id"):
                    last[rec["item_id"]] = rec
                    n += 1
        print(f"{i}回目: 行 {n}")
    print(f"→ 答えのある層 {len(last)} / 68(覆う層は 85)")

    cnt = collections.Counter(r.get("value") for r in last.values())
    print("\n**生の数**: " + " / ".join(f"{k} {v}" for k, v in cnt.most_common()))

    back_layers = [i for i, r in last.items() if r.get("value") in BACK]
    held_layers = [i for i, r in last.items() if r.get("value") in HELD]
    copies_back = sum(int(rows[str(int(i.split("_")[1]))]["covers_copies"]) for i in back_layers)
    moved = len(back_layers) + copies_back
    print(f"\n**候補に戻す**: 代表の層 {len(back_layers)} + 写しの層 {copies_back} = **{moved}層**")
    print(f"**判じかねた答え(3D背景・切り出し・どれでもない・分からない)**: {len(held_layers)}"
          " → **作者に問う**(落とす側か、戻す側か)")
    a_after = A_NOW - moved
    print(f"\n**a 前 {A_NOW} / {A_TOTAL} = {A_NOW/A_TOTAL:.4f}"
          f" → 後 {a_after} / {A_TOTAL} = {a_after/A_TOTAL:.4f}**(両方に名前を付けて残す)")
    print(f"   参考: 85層が全部戻ったときの数は {A_IF_ALL_RETURN} = {A_IF_ALL_RETURN/A_TOTAL:.4f}(= v6)")
    print("   **裁定は点数と独立**(規律9-5)。(ウ) の 0/50 も名前の規則も動かない")

    print("\n**作品単位ごと(記述)**")
    by = collections.defaultdict(collections.Counter)
    for iid, rec in last.items():
        by[rows[str(int(iid.split("_")[1]))]["split_unit"].rsplit("\\", 1)[-1]][rec.get("value")] += 1
    for u, c in sorted(by.items()):
        print(f"   {u}: 計 {sum(c.values())} / " + " ".join(f"{k} {v}" for k, v in c.most_common()))

    if CHECK.is_file():
        void = {r["pos"] for r in csv.DictReader(io.open(CHECK, encoding="utf-8"))
                if r["layer_px_alpha_gt0"] == "0"}
        unk = {str(int(i.split("_")[1])) for i, r in last.items()
               if r.get("value") in ("__none__", "__unknown__")}
        print(f"\n**器具の点検(記述)**: 画素が0 の行 {sorted(void, key=int) or '0'} / "
              f"『どれでもない・分からない』の行 {sorted(unk, key=int) or '0'} → 重なり {len(void & unk)}")

    print("\n**作者の一言(逐語。書き換えない)**")
    for iid in sorted(last, key=lambda s: int(s.split("_")[1])):
        note = (last[iid].get("comment") or "").strip()
        if note:
            print(f"   {iid} [{last[iid].get('value')}] 「{note}」")

    print("\n**形の裁定をどうするかは、数ではなく作者の言葉で決める**"
          "(個別の裁定が形と食い違ったら、形をどうするかを問う)")


main()
