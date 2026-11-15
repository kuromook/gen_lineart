#!/usr/bin/env python3
"""画像素材の50枚の二択の棒を読む(登録 `image_material_50_image_check_20261114` / 第2版 `..._v2_20261115`)。

**棒は画を1枚も見る前に決めてある。ここでは動かさない。**
- 「絵の一部」が **0〜2/50** → 規則を据え置く(0/50 は「6%未満」で厳密なゼロではない)
- **3〜9/50** → 理由『image_material(出力に画素あり)』**だけ**を外して候補に戻す
- **10以上/50** → 外した上で「見えない」「出力に画素が0」の側も50枚で確かめる
- 「分からない」が10以上 → 判定保留

**読み方の決まり**
- 答えの表は**追記だけ**なので、**層ごとに最後の行を採る**(直しは新しい行)。直した行は数えて印字する
- **作者の答えは書き換えない**(規律6)。作者が後から述べた読み(逐語)を当てた数は**別の列**として並べ、
  **素の数と両方印字する**
- **第1版(壊れた画)の11行は版として残すだけで、棒には使わない**
- 器具の性質(原寸の合成と CSP の頁の差が大きい10行)を外した数も、**先に登録したとおり並べて印字する**

**印字するもの**: 答えた層の数 / 素の内訳 / 直した行 / 作者の読みを当てた内訳 / 棒の判定 /
前後の a の数(名前付き) / 本人内一貫性(同じ層を2度) / 「絵の一部」の層の一言。
"""
import csv
import io
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ANS = HERE / "data/role_predictor/image_material_check50_answers_v2_20261115.jsonl"
CHECK = HERE / "data/role_predictor/image_material_check50_check_20261115_v2.csv"
VERDICT = HERE / "inbox/a_verdict_v2_20261114.csv"
LABEL = {"part_of_picture": "絵の一部", "reference": "資料",
         "__none__": "どれでもない", "__unknown__": "分からない"}
# 作者が入力の後に述べた読み(逐語、2026-11-15):
#   「カラー３D背景はラスタライズされていなかったかもしれない 人物紹介カットは資料とも絵ともつかなかったが、
#     どれでもないが適切だったかもしれない」
AUTHORS_LATER_READING = "人物紹介"   # この語を含む一言の行を「どれでもない」として並べる(素の数も残す)
DIFF_BAR = 8.0


def main():
    rows = [json.loads(l) for l in io.open(ANS, encoding="utf-8")]
    last, n_rows = {}, Counter()
    for r in rows:
        n_rows[r["item_id"]] += 1
        last[r["item_id"]] = r
    print(f"答えの行 {len(rows)} / 答えた層 {len(last)}(問いは50)")
    redone = {k: v for k, v in n_rows.items() if v > 1}
    print(f"**直した層 {len(redone)}**(追記だけなので最後の行を採る): "
          + ", ".join(f"{k} {v}回" for k, v in sorted(redone.items())))
    for k in sorted(redone):
        seq = [x["value"] for x in rows if x["item_id"] == k]
        print(f"   {k}: " + " → ".join(LABEL.get(v, v) for v in seq)
              + ("  **答えが変わった**" if len(set(seq)) > 1 else "  (同じ答え)"))

    raw = Counter(LABEL.get(r["value"], r["value"]) for r in last.values())
    print("\n## 素の内訳(作者が押したまま)")
    for k, v in raw.most_common():
        print(f"   {k}: {v}")

    # 作者が後から述べた読みを当てた列(**書き換えではない。並べて印字する**)
    later = Counter()
    moved = []
    for k, r in last.items():
        v = LABEL.get(r["value"], r["value"])
        if AUTHORS_LATER_READING in (r.get("comment") or ""):
            moved.append((k, v))
            v = "どれでもない"
        later[v] += 1
    print(f"\n## 作者が後から述べた読みを当てた内訳(逐語「人物紹介カットは資料とも絵ともつかなかったが、"
          f"どれでもないが適切だったかもしれない」)")
    for k, v in later.most_common():
        print(f"   {k}: {v}")
    print("   当てた行:", ", ".join(f"{k}(素は{v})" for k, v in moved))

    pic_raw, pic_later = raw.get("絵の一部", 0), later.get("絵の一部", 0)
    unk = raw.get("分からない", 0)
    print("\n## 棒(先に決めたもの。動かさない)")
    for name, n in (("素の数", pic_raw), ("作者の読みを当てた数", pic_later)):
        band = ("0〜2 → 規則を据え置く" if n <= 2 else
                "3〜9 → 理由『出力に画素あり』だけを外す" if n <= 9 else
                "10以上 → 外した上で他の側も50枚で確かめる")
        print(f"   「絵の一部」{name} **{n}/50** → **{band}**")
    print(f"   「分からない」{unk}/50 → " + ("**判定保留**" if unk >= 10 else "判定保留には当たらない"))

    # 器具の性質を外した数(先に登録したとおり並べる)
    chk = {f"im50_{int(c['pos']):02d}": c for c in csv.DictReader(io.open(CHECK, encoding="utf-8"))}
    keep = [k for k in last if float(chk[k]["diff_vs_preview_255"]) <= DIFF_BAR]
    sub = Counter(LABEL.get(last[k]["value"], last[k]["value"]) for k in keep)
    print(f"\n## 差が大きい10行を外した {len(keep)}行(**主の値ではない。並べて印字する約束**)")
    print("   " + " / ".join(f"{k} {v}" for k, v in sub.most_common()))

    # 本人内一貫性(同じ中身のファイルの同じ層)
    a, b = last.get("im50_01"), last.get("im50_27")
    print("\n## 本人内一貫性(1行目と27行目は、バイトが同じファイルの同じ層。画面では知らせていない)")
    print(f"   1行目 {LABEL.get(a['value'])} / 27行目 {LABEL.get(b['value'])} → "
          + ("**一致**" if a["value"] == b["value"] else "**食い違い(書き換えない)**"))

    print("\n## 「絵の一部」と答えた層の一言(逐語)")
    for k, r in sorted(last.items()):
        if r["value"] == "part_of_picture":
            print(f"   {k}: {(r.get('comment') or '(一言なし)').strip()}")
    print("\n## 一言のあった層すべて(逐語)")
    for k, r in sorted(last.items()):
        c = (r.get("comment") or "").strip()
        if c:
            print(f"   {k} [{LABEL.get(r['value'])}]: {c}")

    # a の前後(名前を付けて両方印字する)
    if VERDICT.is_file():
        tot = exc = risky = 0
        for r in csv.DictReader(io.open(VERDICT, encoding="utf-8")):
            tot += 1
            if r["a_verdict_v2"] == "template_excluded":
                exc += 1
                if "出力に画素あり" in r["a_rule_v2"]:
                    risky += 1
        print(f"\n## a の前後(規律9-5。両方に名前を付ける)")
        print(f"   **棒を読む前(11-14 の規則)**: {exc:,} / {tot:,} = {exc/tot:.4f}")
        print(f"   理由『image_material(出力に画素あり)』の層: {risky:,}")
        print(f"   **理由を外した後**: {exc-risky:,} / {tot:,} = {(exc-risky)/tot:.4f}")
        print("   **過去の数は撤回しない** —— 取り消すのは規則で、数は版として残る")


main()
