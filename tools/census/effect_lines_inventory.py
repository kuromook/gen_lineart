#!/usr/bin/env python
"""**効果線(流線・集中線)の棚卸し** —— Track Q の頼み(2026-11-17 `note_q_census_effect_lines_inventory_20261117.md`)。数だけ、点数ではない。

**数を見る前に決めたこと**:
- 母集団: 構造 DB の層(フォルダーを含む)から恒久除外の頁(ポートフォリオの8頁を含む)を外したもの。**種別 text の層は外して数える**
  (セリフに「集中」「スピード」などの語が入る。外した数を印字する)
- **定規の印** = 層の生のブロックに `SpecialRulerManager` の鍵が在る(`raw` は NULL でない列だけを持つ)。
  値は整数(定規の表の番号と見える)で、**定規の種類(集中線・平行線・パース)は構造 DB に無い** —— 種類は抽出側で読む必要がある。
  `RulerRange` / `RulerVectorIndex` の鍵も数える(参考。意味は確かめていない)
- 名前の群(先に当たった1つ):
  1. `effect`: 名前に `effect` を含む(大文字小文字を区別しない)
  2. 道具の名前: 名前に `集中線` / `流線` / `ウニフラ` を含む(CSP の素材の名前 `まばら集中線 …` `まばら流線 …` を含む)。
     `スピード` は入れない(トーンの名前 `グラデスピード…` `かけ網スピード…` に当たるため。census の目)
  3. その他
- 時期: 構造 DB の `year` の先頭4桁 ≥ 2020 を「新」、それより前を「旧」(頁の最も早い札ではない)
- 祖先: `page_order` の祖先の名前(フォルダーでない層だけに在る)。祖先に `effect` を含む層と、`モザイク` かつ `effect` を含む層を数える
- 作者の答え: 段の聞き取り(第1便の鍵と答え)/ 第2便 A・B の答え の作者の答えと一言に `効果線` `集中線` `流線` を含む層。
  (頁, 層) で重なりを除く。役割の表は第1便・第2便の写しなので数えない
- 出力: 表 `data/census_full/effect_lines_inventory_20261117.csv`(層ごと、非公開)と、作者の答えの一覧 `effect_lines_author_answers_20261117.csv`

    ./venv/bin/python tools/census/effect_lines_inventory.py
"""
import collections
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
from layer_role_table import page_order  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = "data/census_full/effect_lines_inventory_20261117.csv"
OUT_ANS = "data/census_full/effect_lines_author_answers_20261117.csv"
WORDS = re.compile("効果線|集中線|流線")
TOOL = re.compile("集中線|流線|ウニフラ")


def group(name):
    if "effect" in name.lower():
        return "effect"
    if TOOL.search(name):
        return "道具の名前"
    return "その他"


def rd(p):
    return list(csv.DictReader(open(os.path.join(ROOT, p), encoding="utf-8")))


def main() -> int:
    print("[効果線の棚卸し(数だけ)]")
    con = db.connect()
    ex = exclusions.load()
    rows = con.execute("""select l.page_path, l.layer_id, coalesce(l.name, ''), l.folder, k.v_layer_kind, p.format, p.year, cast(l.raw as varchar)
                            from layers l join v_layer_kind k using (page_path, layer_id) join pages p using (page_path)""").fetchall()
    n0 = len(rows)
    rows = [r for r in rows if r[0] not in ex]
    n_ex = n0 - len(rows)
    n_text = sum(r[4] == "text" for r in rows)
    rows = [r for r in rows if r[4] != "text"]
    print(f"    層 {n0:,} → 恒久除外の頁の層 {n_ex} を外し → 種別 text {n_text:,} を外して {len(rows):,}")
    po = page_order(con, sorted({r[0] for r in rows}))
    out = []
    for r in rows:
        d = json.loads(r[7]) if r[7] else {}
        o = po.get((r[0], r[1]))
        anc = o[2] if o else ""
        out.append(dict(page_path=r[0], layer_id=r[1], format=r[5], name=r[2], is_folder=int(r[3]), v_layer_kind=r[4],
                        period="新" if int(str(r[6])[:4]) >= 2020 else "旧", name_group=group(r[2]),
                        has_ruler=int("SpecialRulerManager" in d), ruler_value=d.get("SpecialRulerManager", ""),
                        has_ruler_range=int("RulerRange" in d), has_ruler_vector=int("RulerVectorIndex" in d),
                        ancestors=anc, anc_effect=int("effect" in anc.lower()),
                        anc_mosaic_effect=int("effect" in anc.lower() and "モザイク" in anc)))
    print(f"    定規の印(SpecialRulerManager): {sum(x['has_ruler'] for x in out):,} 層"
          f"(.clip {sum(x['has_ruler'] and x['format'] == 'clip' for x in out):,} / .lip {sum(x['has_ruler'] and x['format'] == 'lip' for x in out):,})"
          f" / RulerRange {sum(x['has_ruler_range'] for x in out):,} / RulerVectorIndex {sum(x['has_ruler_vector'] for x in out):,}")
    print("    値の上位: " + " / ".join(f"{v} {n:,}" for v, n in collections.Counter(x["ruler_value"] for x in out if x["has_ruler"]).most_common(6)))
    print("    名前の群 × 定規 × 種別 × 時期(.clip と .lip を分ける。層の数):")
    c = collections.Counter((x["format"], x["name_group"], x["has_ruler"], x["v_layer_kind"], x["period"]) for x in out)
    kinds = ("raster", "vector", "folder")
    for f in ("clip", "lip"):
        for g in ("effect", "道具の名前", "その他"):
            for h in (1, 0):
                for per in ("旧", "新"):
                    line = [c[(f, g, h, k, per)] for k in kinds]
                    other = sum(n for (ff, gg, hh, k, pp), n in c.items() if (ff, gg, hh, pp) == (f, g, h, per) and k not in kinds)
                    if sum(line) + other:
                        print(f"        .{f} {g:6} 定規{'あり' if h else 'なし'} {per}: raster {line[0]:,} / vector {line[1]:,} / folder {line[2]:,} / 他の種別 {other:,}")
    leaf = [x for x in out if not x["is_folder"]]
    print(f"    祖先に effect を含む層(フォルダーでない層): {sum(x['anc_effect'] for x in leaf):,}"
          f"(うち祖先に モザイク も含む {sum(x['anc_mosaic_effect'] for x in leaf):,})")
    print("        祖先の上位: " + " / ".join(f"{a} {n:,}" for a, n in collections.Counter(
        x["ancestors"] for x in leaf if x["anc_effect"]).most_common(6)))
    tool = collections.Counter(x["name"] for x in out if x["name_group"] == "道具の名前")
    print(f"    道具の名前の層 {sum(tool.values())}(名前 {len(tool)} 種): " + " / ".join(f"{a} {n}" for a, n in tool.most_common(12)))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(x for x in out if x["name_group"] != "その他" or x["has_ruler"] or x["anc_effect"])
    # 作者の答え
    ans = {}
    key = {(r["no"], r["pos"]): r for r in rd("data/census_full/hearing_drawing_stages_key_20261109.csv")}
    for r in rd("data/census_full/hearing_drawing_stages_scored_20261109.csv"):
        t = f"{r['author']} {r['comment']}"
        if WORDS.search(t):
            k = key[(r["no"], r["pos"])]
            ans.setdefault((k["page_path"], k["layer_id"]), []).append(("段の聞き取り 第1便", r["author"], r["comment"]))
    for p, a, cm in (("data/census_full/hearing_round2_a_answers.csv", "author_stage", "author_comment"),
                     ("data/census_full/hearing_round2_b_answers.csv", "stage", "author_comment")):
        for r in rd(p):
            if p.endswith("b_answers.csv") and r.get("answer_source") != "author":
                continue
            if WORDS.search(f"{r[a]} {r[cm]}"):
                ans.setdefault((r["page_path"], r["layer_id"]), []).append((os.path.basename(p), r[a], r[cm]))
    info = {(x["page_path"], str(x["layer_id"])): x for x in out}
    n_raw = sum(len(v) for v in ans.values())
    print(f"    作者が効果線・集中線・流線と答えた・書いた層: 行 {n_raw} → 重なりを除いて {len(ans)} 層"
          f"(恒久除外の頁 {sum(k[0] in ex for k in ans)})")
    with open(os.path.join(ROOT, OUT_ANS), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page_path", "layer_id", "name", "v_layer_kind", "period", "name_group", "has_ruler", "ancestors", "sources"])
        for k, v in sorted(ans.items()):
            x = info.get(k, {})
            w.writerow([k[0], k[1], x.get("name", ""), x.get("v_layer_kind", ""), x.get("period", ""), x.get("name_group", ""),
                        x.get("has_ruler", ""), x.get("ancestors", ""), " | ".join("/".join(s) for s in v)])
            print(f"        {x.get('period', '?')} {x.get('v_layer_kind', '?')} 群 {x.get('name_group', '?')} 定規 {x.get('has_ruler', '?')}"
                  f" 出所 {len(v)}: " + " | ".join(s[0] for s in v))
    print(f"    → `{OUT}`(非公開。その他の群で定規も祖先の effect も無い層は書かない)/ `{OUT_ANS}`(非公開)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
