#!/usr/bin/env python
"""**S の492頁のうち、Q の R5 の区画が在る頁の数**(Track S の依頼 `note_census_s_count_r5_panels_on_s_pages_20261126.md`。
ユーザーの指示 11-26「censusに、492頁のうち区画が付く頁の数を数えてもらう」。**数えるだけ。列は作らない**)。

- 母集団 = 対 v6 の頁の名前の規則の表(`s_v6_pairs_page_name_rule_20261120.csv`)で `pair_quality = ok`・線画の側 = 清書・下絵の側 = 下絵(census 11-20/21 で 492 を再現した述語)。鍵は `source_path`(= 構造 DB の頁の道)
- 区画が在る = Q の `data/frames/q22_run_r5/arm_log.json` の q(Q22 の便の点検表で頁の道へ)か、Q の `data/q29/pages.json` の頁
- 区画が無い訳(上から最初に当たったもの):
  1. コマ枠のフォルダー(`v_layer_kind = frame_folder`、モザイクの名前だけのものを除く)が在る頁 —— **R5 の対象の外**(コマの形はフォルダーから。Q11 の道)
  1b. 枠らしさの表(`q_frame_likeness_layers_20261121.csv`、コマ枠のフォルダーの無い .clip 頁の層)に頁が無い(上に当たらないもの。11-26 には9頁とも、コマ枠のフォルダーがモザイクの名前のものだけの頁だった)
  2. 名前の候補の枠の層(`cand_cls` = 名前の候補(コマN・枠型)、`set = main`)が 0 枚
  3. 同じく 2枚以上
  4. 年代札が 2023b 以降(鉛筆漫画の別枠。**鉛筆漫画かどうかの列は無い**ので札だけで分けた。札で落とした数として読む)
  5. 候補がちょうど1枚で Q がまだ当てていない(11-26 には4頁とも Q12〜Q20 の聞き取りで引いた頁で、Q22 の母集団から外された頁だった)
- 作品フォルダー(`work_id`)ごとに印字(プールしない)。内容重複の代表(QC 表 v6 の `is_primary`)だけの数も並べる
"""
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
QX = os.path.join(ROOT, "..", "lineart-layer-exclusion")
LATE = ("2023b", "2024", "2024b", "2024c")
REASONS = ["区画が在る", "コマ枠のフォルダーが在る(R5 の対象外)", "枠らしさの表に頁が無い(モザイクの枠フォルダーだけの頁)", "名前の候補 0枚", "名前の候補 2枚以上", "札 2023b 以降(鉛筆漫画の別枠)", "候補1枚・Q がまだ当てていない"]


def main() -> int:
    pop = [r for r in csv.DictReader(open(os.path.join(D, "s_v6_pairs_page_name_rule_20261120.csv"), encoding="utf-8"))
           if r["pair_quality"] == "ok" and r["line_rules"] == "清書" and r["sketch_rules"] == "下絵"]
    qc = {r["source_path"]: r for r in csv.DictReader(open(os.path.join(ROOT, "inbox", "clip_pairs_qc_v6_20261114.csv"), encoding="utf-8"))}
    q22 = {r["q"]: r["page_path"] for r in csv.DictReader(open(os.path.join(D, "q22", "q22_pages_images_check_20261124.csv"), encoding="utf-8")) if r["kind"] == "page"}
    reg = {q22[x["q"]] for x in json.load(open(os.path.join(QX, "data", "frames", "q22_run_r5", "arm_log.json"), encoding="utf-8"))}
    n22 = len(reg)
    reg |= set(json.load(open(os.path.join(QX, "data", "q29", "pages.json"), encoding="utf-8")))
    print(f"区画の在る頁: Q22 {n22} + Q29 {len(reg) - n22}(重ならない分)= {len(reg)}")
    fl = collections.defaultdict(int); seen = set()
    for r in csv.DictReader(open(os.path.join(QX, "data", "census", "q_frame_likeness_layers_20261121.csv"), encoding="utf-8")):
        if r["set"] == "main":
            seen.add(r["page_path"])
            fl[r["page_path"]] += r["cand_cls"] == "名前の候補(コマN・枠型)"
    year = {r["page_path"]: r["year_label"] for r in csv.DictReader(open(os.path.join(D, "page_file_copies_20261123.csv"), encoding="utf-8"))}
    c = db.connect()
    ff = {r[0] for r in c.execute("select l.page_path, coalesce(l.name,'') from layers l join v_layer_kind v using (page_path, layer_id) "
                                  "where v.v_layer_kind = 'frame_folder'").fetchall() if "モザイク" not in r[1]}
    rows = []
    for r in pop:
        p = r["page_path"]
        if p in reg:
            why = REASONS[0]
        elif p in ff:
            why = REASONS[1]
        elif p not in seen:
            why = REASONS[2]
        elif fl[p] == 0:
            why = REASONS[3]
        elif fl[p] >= 2:
            why = REASONS[4]
        elif year.get(p) in LATE:
            why = REASONS[5]
        else:
            why = REASONS[6]
        rows.append((r["work_id"], p, why, qc.get(p, {}).get("is_primary", "")))
    print(f"母集団 {len(rows)} 頁(QC 表に is_primary の無い頁 {sum(1 for x in rows if x[3] == '')})/ 代表(is_primary = True)"
          f" {sum(1 for x in rows if x[3] in ('True', '1'))}")
    for lab, sel in (("全部", rows), ("代表だけ", [x for x in rows if x[3] in ("True", "1")])):
        c = collections.Counter(x[2] for x in sel)
        print(f"\n[{lab}] " + " / ".join(f"{k} {c[k]}" for k in REASONS))
    print("\n作品フォルダーごと(全部 / 代表だけ。区画が在る頁を先に、0 の訳は省く):")
    for w in sorted({x[0] for x in rows}):
        s = [x for x in rows if x[0] == w]
        sp = [x for x in s if x[3] in ("True", "1")]
        c, cp = collections.Counter(x[2] for x in s), collections.Counter(x[2] for x in sp)
        print(f"  {w}: 頁 {len(s)}/{len(sp)} — " + " / ".join(f"{k} {c[k]}/{cp[k]}" for k in REASONS if c[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
