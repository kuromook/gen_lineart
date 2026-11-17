#!/usr/bin/env python3
"""(ウ) の棒を読む —— **答えを読む前に書いた器具**(登録 `validation_hearing_registered_before_images`)。

**棒(登録のまま。決め方は動かさない)** —— 「資料」でない答えの数(絵の一部 + 下絵 + 3D背景 + 切り出し・コラージュ):
- **0〜2/50** → 名前の規則を据え置く(**0/50 は『6%未満』で厳密なゼロではない**)
- **3以上/50** → 名前の規則を撤回し、3,642 を候補に戻す
- **「どれでもない」+「分からない」が10以上** → **判定保留**(棒に届かなかったことと、判定できなかったことは別)
- 群ごとの内訳は並べて印字するが**決め手にしない**(`資料化下絵` は7層しかない)

**登録の数の訂正(2026-11-17。決め方は動かさない)**
- 登録の文には撤回のときの数を「215,162 → 215,961 相当」と書いてあるが、**これは算術の誤り**。
  215,961 は v5(非表示を理由に 4,356 を落としていた版)の数で、**規則を撤回したら落とす数は増えない**
- **正しくは 215,247(v7)− 3,642 = 211,605**(v6 基準でも 215,162 − 3,557 = 211,605 で同じ)。
  ほかの条項(空(定義C)など)が拾う層が在れば**それより多く**なるので、**census に前後を両方印字してもらう**
- **直したのは数の札だけで、棒(0〜2 / 3以上 / 10以上)は動かしていない**(規律10。前後を両方、名前を付けて印字する)

**作者の答えは書き換えない**(規律6)。層ごとに最後の行を採る(押し直しの最後の答え)。
"""
import collections
import csv
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ANS = HERE / "data/role_predictor/shiryou_name_check50c_answers_20261117.jsonl"
LIST = HERE / "data/role_predictor/shiryou_name_check50_v4_20261117.csv"
KEY = HERE / "inbox/image_material_hidden_groups_key_v3_20261117.csv"

A_BEFORE, A_TOTAL = 215247, 437425          # v7(落とす / 全層)
DROPPED_BY_CLAUSE = 3642                    # この条項が落としている層(census v7)
A_IF_WITHDRAWN = A_BEFORE - DROPPED_BY_CLAUSE   # = 211,605(下限。ほかの条項が拾えば増える)
REGISTERED_SLIP = 215961                    # 登録に書いてあった数(算術の誤り。撤回しない。印字する)
NOT_SHIRYOU = ("part_of_picture", "sketch_underdrawing", "background_3d", "cutout_collage")
HELD = ("__none__", "__unknown__")


def main():
    rows = {r["pos"]: r for r in csv.DictReader(io.open(LIST, encoding="utf-8"))}
    key = {(r["page_path"], r["layer_id"]): r
           for r in csv.DictReader(io.open(KEY, encoding="utf-8"))}
    last = {}
    n_lines = 0
    for line in io.open(ANS, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        n_lines += 1
        rec = json.loads(line)
        if rec.get("item_id"):
            last[rec["item_id"]] = rec          # 層ごとに**最後の**行を採る
    print(f"答えの行 {n_lines} → 層 {len(last)}(層ごとに最後の行)")

    cnt = collections.Counter(r.get("value") for r in last.values())
    print("\n**生の数**: " + " / ".join(f"{k} {v}" for k, v in cnt.most_common()))
    not_shiryou = sum(cnt[v] for v in NOT_SHIRYOU)
    held = sum(cnt[v] for v in HELD)
    print(f"\n「資料」でない答え **{not_shiryou} / {len(last)}**"
          f"(絵の一部 {cnt['part_of_picture']} / 下絵 {cnt['sketch_underdrawing']}"
          f" / 3D背景 {cnt['background_3d']} / 切り出し・コラージュ {cnt['cutout_collage']})")
    print(f"「どれでもない」+「分からない」 **{held}**")

    band = "0〜2(据え置き)" if not_shiryou <= 2 else "3以上(撤回)"
    print(f"\n**帯: {band}**")
    if not_shiryou <= 2:
        print(f"   → **名前の規則を据え置く。a は {A_BEFORE} / {A_TOTAL} = {A_BEFORE/A_TOTAL:.4f} のまま**")
        if not_shiryou == 0:
            print("   → **0/50 は『6%未満』で厳密なゼロではない**(条例5)")
    else:
        print(f"   → **名前の規則を撤回し、この条項の {DROPPED_BY_CLAUSE}層を候補に戻す**")
        print(f"   → **a 前 {A_BEFORE} / {A_TOTAL} = {A_BEFORE/A_TOTAL:.4f}"
              f" → 後 {A_IF_WITHDRAWN} / {A_TOTAL} = {A_IF_WITHDRAWN/A_TOTAL:.4f}**"
              "(下限。ほかの条項が拾う層が在れば増える。census に前後を両方印字してもらう)")
        print(f"   → **登録に書いてあった {REGISTERED_SLIP} は算術の誤り**(v5 の数)。"
              "**棒(0〜2 / 3以上 / 10以上)は動かしていない**")
        print("   → **非表示の取り込み画像の層に a が当てられる印は無くなる** —— 4,356 は丸ごと Q と R へ")
    if held >= 10:
        print(f"\n**判定保留**: 「どれでもない」+「分からない」が {held}(10以上)。"
              "棒に届かなかったことと判定できなかったことは別。両方書く")

    print("\n**群ごと(決め手にしない)**")
    by = collections.defaultdict(collections.Counter)
    for iid, rec in last.items():
        pos = str(int(iid.split("_")[1]))
        by[rows[pos]["group"]][rec.get("value")] += 1
    for g, c in by.items():
        ns = sum(c[v] for v in NOT_SHIRYOU)
        print(f"   {g}: 計 {sum(c.values())} / 「資料」でない {ns} / " +
              " ".join(f"{k} {v}" for k, v in c.most_common()))

    print("\n**祖先に資料の印ごと(記述。棒ではない)**")
    anc = collections.defaultdict(collections.Counter)
    for iid, rec in last.items():
        r = rows[str(int(iid.split("_")[1]))]
        k = key[(r["page_path"], r["layer_id"])]["ancestors_has_shiryou"]
        anc[k][rec.get("value")] += 1
    for k, c in sorted(anc.items()):
        print(f"   祖先に資料={k}: 計 {sum(c.values())} / " + " ".join(f"{x} {y}" for x, y in c.most_common()))

    print("\n**作者の一言(逐語。書き換えない)**")
    for iid in sorted(last, key=lambda s: int(s.split("_")[1])):
        note = (last[iid].get("comment") or "").strip()
        if note:
            print(f"   {iid} [{last[iid].get('value')}] 「{note}」")

    # **器具の点検(記述。棒ではない。答えを読んだ後に足した —— 棒は動かしていない)**
    # 抽出側が「描き直しても頁に1画素も出ない」と報告した行と、作者の「分からない」が重なるか
    chkp = HERE / "data/role_predictor/shiryou_name_check50c_check_20261117.csv"
    void = {r["pos"] for r in csv.DictReader(io.open(chkp, encoding="utf-8"))
            if r["layer_px_alpha_gt0"] == "0"}
    unk = {str(int(i.split("_")[1])) for i, r in last.items() if r.get("value") in HELD}
    print(f"\n**器具の点検(記述)**: 画素が0 の行 {sorted(void, key=int)} / "
          f"作者が「どれでもない・分からない」と答えた行 {sorted(unk, key=int)}"
          f" → 重なり {len(void & unk)}/{len(void)}")

    print("\n**この50層が覆っていないもの**: 作者の裁定で落とす側に移った85層"
          "(名前が『下絵 N』などで祖先に『資料』)は、**母集団 3,557 に入っていない** —— 別に確かめる")


main()
