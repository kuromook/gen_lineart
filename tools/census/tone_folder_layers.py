#!/usr/bin/env python
"""**トーンのフォルダーの中の層(構造 DB だけ)** —— Track E の頼み(`note_census_tone_layer_flag_20261111.md` §4-2)。

**答えの要点(census が DB で確かめた)**: `tone_enabled` が真の層は構造 DB に 91,060層在り、**全部が種別 `mask_fill`**
(`layer_type` の 2 のビット ∧ `has_gradation` = 塗りの層。census `layer_type_values.csv`: 塗り(トーン・ベタ))。
層の役割の表は段の候補(raster と vector)なので、`tone_enabled` が全部 0 なのは**母集団の側**で、列の不具合ではない。

**決めたこと**:
- 作品単位は1つ: `2018\\ako6`(作者が名指した ako の系統で 2018。.clip だけで、表現色モノクロの層を持つ頁が多い)。
  候補の3つ(ako6 / comics4 / china_v2)のうち、census が選んだ。**漫画かイラストかの表のこの行は作者の答えではない(未確認)**
- 層: 祖先に名前が `トーン` か `tone`(大小を問わない)を含むフォルダーを持つ、フォルダーでない層の全部
- 列は構造 DB の列だけ。**画素(`n_px_a_gt0` / `n_px_a_gt0_own_mask`)は無い** —— 抽出側への発注が要り、ユーザーの承認の後

    ./venv/bin/python tools/census/tone_folder_layers.py
"""
import collections
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIT = "\\2018\\ako6\\"
TONE = re.compile(r"トーン|tone", re.I)
OUT = "data/census_full/tone_folder_layers_2018_ako6_20261111.csv"


def main() -> int:
    print("[トーンのフォルダーの中の層(構造 DB だけ)]")
    con = db.connect()
    ex = exclusions.load()
    k = con.execute("""select k.v_layer_kind, coalesce(l.tone_enabled, false), count(*)
                         from layers l join v_layer_kind k using (page_path, layer_id) group by 1, 2""").fetchall()
    on = {a: n for a, t, n in k if t}
    print(f"    全層で tone_enabled が真: {sum(on.values()):,}層 → 種別 " + " / ".join(f"{a} {n:,}" for a, n in on.items()))
    rows = con.execute("""select l.page_path, l.layer_id, l.parent_id, coalesce(l.name, ''), l.folder, k.v_layer_kind, l.layer_type,
                                 l.colour_type_index, l.has_mask, coalesce(l.tone_enabled, false), l.tone_density_pct, l.tone_lpi,
                                 l.visibility & 1, l.opacity
                            from layers l join v_layer_kind k using (page_path, layer_id) join pages p using (page_path)
                           where l.page_path like ? and p.format = 'clip'""", (f"%{UNIT}%",)).fetchall()
    by = {(r[0], r[1]): r for r in rows}
    out = []
    for r in rows:
        if r[4] or r[0] in ex:
            continue
        anc, pid = [], r[2]
        while pid is not None and (r[0], pid) in by:
            anc.append(by[(r[0], pid)][3])
            pid = by[(r[0], pid)][2]
        tf = [a for a in anc if TONE.search(a)]
        if not tf:
            continue
        out.append(dict(page_path=r[0], layer_id=r[1], name=r[3], tone_folder=tf[0], ancestors="/".join(reversed(anc)),
                        v_layer_kind=r[5], layer_type=r[6], colour_type_index=r[7], has_mask=int(bool(r[8])),
                        tone_enabled=int(bool(r[9])), tone_density_pct=r[10], tone_lpi=r[11], self_shown_db=r[12], opacity=r[13]))
    c = collections.Counter((d["v_layer_kind"], d["tone_enabled"], d["has_mask"]) for d in out)
    print(f"    単位 {UNIT.strip(chr(92))}: 頁 {len({r[0] for r in rows})} → トーンのフォルダーの中の層 {len(out):,}(頁 {len({d['page_path'] for d in out})})")
    print("    (種別, tone_enabled, has_mask): " + " / ".join(f"{a} {n:,}" for a, n in c.most_common()))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"    → `{OUT}`(非公開)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
