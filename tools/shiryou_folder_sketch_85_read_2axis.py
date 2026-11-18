#!/usr/bin/env python3
"""**2つの軸の裁定を読む** —— 答えを読む前に書いた器具(登録 `two_axis_ruling_registered_before_answers_20261118`)。

**これは測定ではなく裁定**(規律9-5)。棒は無い。**(ウ) の 0/50 も名前の規則も動かない。**
**作者の答えは書き換えない**(規律6)。層ごとに最後の行を採る。

**先に書いてある読み方**
- **a は軸1(用途)だけを読む**(作者 2026-11-18「レイヤの発注時には機械的に除外しつつ、
  そのレイヤが持つ特徴量を参考にすることができる」):
  - `資料` → **落としたまま**
  - `絵の一部` / `下絵として使う` → **候補に戻す**(写しの層も一緒に動く)
  - `切り出し・コラージュ` → **判じかねた扱い**。落としたい対象だが(11-16 の裁定)、
    **決定論の印はまだ一度も測っていない**ので、a では落とせない → 作者に問う
  - `どれでもない` / `分からない` → 作者に問う
- **軸2(見た目)は a の判断に使わない。** 表に出して数えるだけ —— 使うなら規律13
- **初期値を置いた便(1b)では、変えた数を印字する**(`prefill_kept`)。閉じていない証拠になる
- 前後の a を両方、名前を付けて印字する。全部戻れば **215,162 = v6**
"""
import collections
import csv
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LIST = HERE / "data/role_predictor/shiryou_folder_sketch_85_20261117.csv"
D = HERE / "data/role_predictor"
ANS = [D / "shiryou_folder_sketch_85_answers_1b_2axis_20261118.jsonl",
       D / "shiryou_folder_sketch_85_answers_2_2axis_20261118.jsonl"]
R1 = D / "shiryou_folder_sketch_85_answers_r1_20261117.jsonl"   # 1列だった便(用途だけ)

A_NOW, A_TOTAL = 215247, 437425
A_IF_ALL_RETURN = 215162
BACK = ("part_of_picture", "sketch_underdrawing")
STAY = ("reference",)
HELD = ("cutout_collage", "__none__", "__unknown__")


def read(path):
    last = {}
    if not path.is_file():
        return last, 0
    n = 0
    for line in io.open(path, encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            if rec.get("item_id"):
                last[rec["item_id"]] = rec
                n += 1
    return last, n


def main():
    rows = {f"im85_{int(r['pos']):02d}": r
            for r in csv.DictReader(io.open(LIST, encoding="utf-8"))}
    last = {}
    for i, p in enumerate(ANS, 1):
        d, n = read(p)
        print(f"便 {'1b' if i == 1 else '2'}: 行 {n} / 層 {len(d)}" if n else
              f"**便 {'1b' if i == 1 else '2'} の答えがまだ無い**: {p.name}")
        last.update(d)
    old, _ = read(R1)
    print(f"→ 2つの軸で答えのある層 {len(last)} / 68(覆う層は 85)")
    if old:
        print(f"   1列だった便の答えも在る(用途だけ、{len(old)}層)—— **書き換えない。別の版として残す**")

    c1 = collections.Counter(r.get("value") for r in last.values())
    c2 = collections.Counter(r.get("value2") for r in last.values())
    print("\n**軸1 用途(a が読む)**: " + (" / ".join(f"{k} {v}" for k, v in c1.most_common()) or "まだ無い"))
    print("**軸2 見た目(落とすのに使わない)**: " + (" / ".join(f"{k} {v}" for k, v in c2.most_common()) or "まだ無い"))

    print("\n**用途 × 見た目(記述。この表が作者の求めたもの)**")
    cross = collections.Counter((r.get("value"), r.get("value2")) for r in last.values())
    for (a1, a2), n in sorted(cross.items(), key=lambda kv: -kv[1]):
        print(f"   {a1} × {a2}: {n}")

    pre = [r for r in last.values() if r.get("prefill")]
    if pre:
        kept = sum(1 for r in pre if r.get("prefill_kept"))
        print(f"\n**初期値(前の便の用途)を置いた層 {len(pre)}: そのまま {kept} / 変えた {len(pre) - kept}**"
              "(置いた答えは閉じていない。変えた層は下に並べる)")
        for r in pre:
            if not r.get("prefill_kept"):
                print(f"   変えた: {r['item_id']} {r['prefill']} → {r['value']}")

    back = [i for i, r in last.items() if r.get("value") in BACK]
    held = [i for i, r in last.items() if r.get("value") in HELD]
    copies = sum(int(rows[i]["covers_copies"]) for i in back)
    moved = len(back) + copies
    a_after = A_NOW - moved
    print(f"\n**候補に戻す**: 代表の層 {len(back)} + 写しの層 {copies} = **{moved}層**")
    for i in sorted(back):
        print(f"   戻す: {i} [{last[i]['label']} / 見た目 {last[i].get('label2')}]"
              f" 写し {rows[i]['covers_copies']}")
    print(f"**判じかねた答え(切り出し・コラージュ / どれでもない / 分からない)**: {len(held)} → **作者に問う**")
    for i in sorted(held):
        print(f"   問う: {i} [{last[i]['label']} / 見た目 {last[i].get('label2')}]")
    print(f"\n**a 前 {A_NOW} / {A_TOTAL} = {A_NOW/A_TOTAL:.4f}"
          f" → 後 {a_after} / {A_TOTAL} = {a_after/A_TOTAL:.4f}**(両方に名前を付けて残す)")
    print(f"   参考: 85層が全部戻ったときの数は {A_IF_ALL_RETURN} = {A_IF_ALL_RETURN/A_TOTAL:.4f}(= v6)")
    print("   **裁定は点数と独立**(規律9-5)。(ウ) の 0/50 も名前の規則も動かない")
    print("   **軸2(見た目)は a の数を1つも動かしていない** —— 使うなら規律13(的中率と取りこぼしを先に印字)")

    print("\n**作者の一言(逐語。書き換えない)**")
    any_note = False
    for i in sorted(last, key=lambda s: int(s.split("_")[1])):
        note = (last[i].get("comment") or "").strip()
        if note:
            any_note = True
            print(f"   {i} [{last[i]['label']} / {last[i].get('label2')}] 「{note}」")
    if not any_note:
        print("   0件")

    if len(last) < 68:
        print(f"\n**まだ {68 - len(last)}層の答えが無い。** 足りない便を先に出す")


main()
