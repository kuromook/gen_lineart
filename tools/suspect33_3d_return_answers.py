#!/usr/bin/env python3
"""**3D の疑い 33層の答えを読み、census に返す表を書く** —— **答えを1つも見る前に書いた器具**。
登録 `suspect33_3d_ruling_registered_before_answers_20261119`。

**E はこの答えを解釈しない。** a はこの便を読まない(名前で引けない 3D は a の外)——
**E の仕事は画面を立てて、答えをそのまま census に返すこと**。census が R(段)・Q(3D背景)・D(対)へ配る。

**先に決めてある書き方**
- 列は `pos` / `page_path` / `layer_id` / **`origin_axis`**(線の出どころ)/ **`draws_axis`**(何を描いたか)/
  `label_origin` / `label_draws` / `comment` / `interview_id` —— **im85 の `use_axis`(用途)とは別の軸なので名前を分ける**
- 層ごとに**最後の行**を採る。**作者の答えは書き換えない**(規律6)
- **数は記述だけ。棒は無い。** 「3D から出した線」が何層でも、**a の数は 215,247 / 437,425 = 0.4921 のまま**
- **答えの表は非公開**(原稿の頁の道と層の番号が入る。作者には見せない)
"""
import collections
import csv
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
D = HERE / "data/role_predictor"
CHECK = D / "succor_3d_suspect33_check_20261118.csv"
ANS = [D / "succor_3d_suspect33_answers_20261119.jsonl"]
OUT = D / "succor_3d_suspect33_answers_for_census_20261119.csv"
A_NOW, A_TOTAL = 215247, 437425


def main():
    chk = {f"d3s_{int(r['pos']):02d}": r for r in csv.DictReader(io.open(CHECK, encoding="utf-8"))}
    last = {}
    for p in ANS:
        if not p.is_file():
            print(f"**答えがまだ無い**: {p.name}")
            continue
        n = 0
        for line in io.open(p, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                if rec.get("item_id"):
                    last[rec["item_id"]] = rec
                    n += 1
        print(f"行 {n} / 層 {len(last)}")
    print(f"→ 答えのある層 {len(last)} / 33")
    if not last:
        return

    c1 = collections.Counter(r.get("label") for r in last.values())
    c2 = collections.Counter(r.get("label2") for r in last.values())
    print("\n**軸1 線の出どころ**: " + " / ".join(f"{k} {v}" for k, v in c1.most_common()))
    print("**軸2 何を描いたか**: " + (" / ".join(f"{k} {v}" for k, v in c2.most_common() if k) or "この便では聞いていない"))
    print("\n**出どころ × 何を描いたか(記述)**")
    for (a1, a2), n in sorted(collections.Counter(
            (r.get("label"), r.get("label2")) for r in last.values()).items(), key=lambda kv: -kv[1]):
        print(f"   {a1} × {a2}: {n}")

    held = [i for i, r in last.items() if r.get("value") in ("__none__", "__unknown__")]
    print(f"\n**判じかねた答え(どれでもない・分からない)**: {len(held)} {sorted(held) or ''}")
    print(f"**a は動かない**: {A_NOW} / {A_TOTAL} = {A_NOW/A_TOTAL:.4f} の前後とも同じ"
          " —— 名前で引けない 3D は a の外(11-15・11-18 の裁定)。**この便に棒は無い**")

    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos", "page_path", "layer_id", "origin_axis", "draws_axis",
                    "label_origin", "label_draws", "comment", "interview_id"])
        for i in sorted(last, key=lambda s: int(s.split("_")[1])):
            r, c = last[i], chk[i]
            w.writerow([int(i.split("_")[1]), c["page_path"], c["layer_id"],
                        r.get("value"), r.get("value2") or "",
                        r.get("label"), r.get("label2") or "",
                        (r.get("comment") or "").strip(), r.get("interview_id")])
    import hashlib
    print(f"\n書いた {OUT}(非公開。作者には見せない)")
    print("   sha256 先頭", hashlib.sha256(OUT.read_bytes()).hexdigest()[:16])

    print("\n**作者の一言(逐語。書き換えない)**")
    any_note = False
    for i in sorted(last, key=lambda s: int(s.split("_")[1])):
        note = (last[i].get("comment") or "").strip()
        if note:
            any_note = True
            print(f"   {i} [{last[i].get('label')} / {last[i].get('label2')}] 「{note}」")
    if not any_note:
        print("   0件")
    if len(last) < 33:
        print(f"\n**まだ {33 - len(last)}層の答えが無い**")


main()
