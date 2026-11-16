#!/usr/bin/env python3
"""(イ) 50枚の二択の棒を読む(登録 `image_material_second_50_check_20261115`。**画を1枚も見る前に決めた**)。

棒(そのまま。動かさない) —— 「絵の一部」の数で:
- **0〜2/50** → この340層は落とし続ける(0/50 は「6%未満」で厳密なゼロではない)
- **3〜9/50** → 理由『出力に画素が0』と『測れない』を外して候補に戻す(**216,301 → 215,961 / 437,425**)
- **10以上/50** → 外した上で『見えない』4,356 の側も50枚で確かめる
- **「どれでもない」は「絵の一部」に数えない。**「どれでもない」+「分からない」が10以上なら**判定保留**

**読み方**: 追記だけなので層ごとに最後の行を採り、直した層を印字する。**作者の答えは書き換えない**(規律6)。
先に登録した点検: **41行と49行は層の画のバイトが同じ**(別のファイル。画面では知らせていない)。
器具の性質(差がやや大きい14行)を外した数も、約束どおり並べて印字する(**主の値ではない**)。
"""
import csv
import io
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ANS = HERE / "data/role_predictor/image_material_check50b_answers_20261116.jsonl"
LIST = HERE / "data/role_predictor/image_material_check50b_20261115.csv"
CHECK = HERE / "data/role_predictor/image_material_check50b_check_20261116.csv"
LABEL = {"part_of_picture": "絵の一部", "reference": "資料",
         "__none__": "どれでもない", "__unknown__": "分からない"}
A_BEFORE, A_TOTAL = 216301, 437425   # 種別 3d の2層を保留に入れた後の v4 の数
DIFF_BAR = 8.0


def main():
    rows = [json.loads(l) for l in io.open(ANS, encoding="utf-8")]
    last, n_rows = {}, Counter()
    for r in rows:
        n_rows[r["item_id"]] += 1
        last[r["item_id"]] = r
    print(f"答えの行 {len(rows)} / 答えた層 {len(last)}(問いは50)")
    redone = {k: v for k, v in n_rows.items() if v > 1}
    if redone:
        print(f"**直した層 {len(redone)}**(最後の行を採る)")
        for k in sorted(redone):
            seq = [x["value"] for x in rows if x["item_id"] == k]
            print(f"   {k}: " + " → ".join(LABEL.get(v, v) for v in seq)
                  + ("  **答えが変わった**" if len(set(seq)) > 1 else "  (同じ答え)"))

    grp = {f"im50b_{int(r['pos']):02d}": r["group"] for r in csv.DictReader(io.open(LIST, encoding="utf-8"))}
    raw = Counter(LABEL.get(r["value"], r["value"]) for r in last.values())
    print("\n## 素の内訳(作者が押したまま)")
    for k, v in raw.most_common():
        print(f"   {k}: {v}")
    print("\n## 群ごと")
    for g in ("出力に画素が0", "測れない"):
        c = Counter(LABEL.get(last[k]["value"], last[k]["value"]) for k in last if grp[k] == g)
        print(f"   {g}({sum(c.values())}層): " + " / ".join(f"{a} {b}" for a, b in c.most_common()))

    pic = raw.get("絵の一部", 0)
    none_, unk = raw.get("どれでもない", 0), raw.get("分からない", 0)
    band = ("0〜2 → この340層は落とし続ける" if pic <= 2 else
            "3〜9 → 理由『出力に画素が0』と『測れない』を外して候補に戻す" if pic <= 9 else
            "10以上 → 外した上で『見えない』4,356 の側も50枚で確かめる")
    print("\n## 棒(先に決めたもの。動かさない)")
    print(f"   「絵の一部」**{pic}/50** → **{band}**")
    print(f"   「どれでもない」{none_} + 「分からない」{unk} = {none_+unk}/50 → "
          + ("**判定保留**(この群は判じない)" if none_ + unk >= 10 else "判定保留には当たらない"))
    print("\n## a の前後(規律9-5。両方に名前を付ける)")
    print(f"   **棒を読む前(v4 = 種別 3d の2層を保留に入れた数)**: {A_BEFORE:,} / {A_TOTAL:,} = {A_BEFORE/A_TOTAL:.4f}")
    if pic >= 3:
        after = A_BEFORE - 340
        print(f"   **理由『出力に画素が0』(267)と『測れない』(73)を外した後**: {after:,} / {A_TOTAL:,} = {after/A_TOTAL:.4f}")
    print("   **過去の数は撤回しない** —— 取り消すのは規則で、数は版として残る")

    chk = {f"im50b_{int(c['pos']):02d}": c for c in csv.DictReader(io.open(CHECK, encoding="utf-8"))}
    keep = [k for k in last if float(chk[k]["diff_vs_preview_255"]) <= DIFF_BAR]
    sub = Counter(LABEL.get(last[k]["value"], last[k]["value"]) for k in keep)
    print(f"\n## 差がやや大きい14行を外した {len(keep)}行(**主の値ではない。並べて印字する約束**)")
    print("   " + " / ".join(f"{k} {v}" for k, v in sub.most_common()))

    a, b = last.get("im50b_41"), last.get("im50b_49")
    if a and b:
        print("\n## 本人内一貫性(41行目と49行目は、層の画のバイトが同じ。画面では知らせていない)")
        print(f"   41行目 {LABEL.get(a['value'])} / 49行目 {LABEL.get(b['value'])} → "
              + ("**一致**" if a["value"] == b["value"] else "**食い違い(書き換えない)**"))
    v12 = last.get("im50b_12")
    if v12:
        print(f"\n## 12行目(描き直しても頁に1画素も出ない層): {LABEL.get(v12['value'])}"
              f" / 一言: {(v12.get('comment') or '(なし)').strip()}")


    # 作者の一言の語で束ねる(**答えの書き換えではない。作者自身の言葉を数えただけ**)
    KEY = (("3D背景", ("3D", "３D")),
           ("切り出し・コラージュ", ("きりだし", "切り出し", "コラージュ", "はりこん")),
           ("ラフ・下絵", ("ラフ", "下絵")),
           ("素材背景", ("素材",)),
           ("何も映っていない", ("何も映ってない", "白紙")))
    print("\n## 作者の一言の語で束ねた表(**答えは書き換えていない。作者自身の言葉**)")
    seen = set()
    for name, keys in KEY:
        hit = [k for k, r in sorted(last.items())
               if any(x in (r.get("comment") or "") for x in keys) and k not in seen]
        seen |= set(hit)
        if not hit:
            continue
        c = Counter(LABEL.get(last[k]["value"], last[k]["value"]) for k in hit)
        print(f"   {name}: {len(hit)}層 —— " + " / ".join(f"{a} {b}" for a, b in c.most_common()))
    rest = [k for k in last if k not in seen]
    c = Counter(LABEL.get(last[k]["value"], last[k]["value"]) for k in rest)
    nocom = [k for k in rest if not (last[k].get("comment") or "").strip()]
    print(f"   上のどれにも当たらない: {len(rest)}層(一言なし {len(nocom)}) —— "
          + " / ".join(f"{a} {b}" for a, b in c.most_common()))
    print("   **これは分類の規則ではない** —— 作者が書いた語で束ねただけで、"
          "機械がこの語を層から引けるとは言っていない(規律13)")

    print("\n## 一言のあった層すべて(逐語)")
    for k, r in sorted(last.items()):
        c = (r.get("comment") or "").strip()
        if c:
            print(f"   {k} [{LABEL.get(r['value'])} / {grp[k]}]: {c}")


main()
