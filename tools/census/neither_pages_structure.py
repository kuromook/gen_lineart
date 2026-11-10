#!/usr/bin/env python
"""**名前の当て方が何も言わない頁(「どちらも無い」)の構造の列** —— Track E の問い
(`note_census_b_hypotheses_measured_20261110.md` §5-2)、ユーザーの決定(2026-11-10「出す」)。

**なぜ**: 層の役割の表は名前で引いた層なので、名前の死角(線画系・下絵系のどちらの名前も無い頁)が入っていない(E が数えた)。

**決めたこと(作る前に)**:
- 頁: `.clip`、恒久除外を除く。フォルダーでもテキストでもない層の名前に、線画系(`stage_hearing.LINE`)も
  下絵系(`stage_hearing.SKETCH`)も当たらない頁(11-09 の「どちらも無い」と同じ定義。当時 2,191頁)
- 層: その頁の、フォルダーでもテキストでもない層の全部(段の候補の一覧と同じ種別の切り方)
- 列: 層の役割の表の構造の列と同じ定義(`layer_role_table.page_order` と同じ問い合わせ)。**画素・描画色は無い**(構造 DB だけ)
- `holdout_side`: 便A の単位の側(評価 / 学習 / 空)。**便A の評価単位の頁も残し、印で分ける**(落とすかは E が決める)

    ./venv/bin/python tools/census/neither_pages_structure.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import split_units  # noqa: E402
from layer_role_table import page_order  # noqa: E402
from p_split_columns import wd  # noqa: E402
from stage_hearing import LINE, SKETCH, name_rule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOLDOUT = "data/census_full/holdout_units_round2_a_20261110.csv"
OUT = "data/census_full/neither_pages_structure_20261110.csv"


def main() -> int:
    print("[名前の当て方が何も言わない頁の構造の列(Track E の問い、ユーザー承認)]")
    con = db.connect()
    ex = exclusions.load()
    side = {r["split_unit"]: r["side"] for r in csv.DictReader(open(os.path.join(ROOT, HOLDOUT), encoding="utf-8"))}
    rows = con.execute("""select l.page_path, l.layer_id, coalesce(l.name, '')
                            from layers l join pages p using (page_path) join v_layer_kind k using (page_path, layer_id)
                           where p.format = 'clip' and not l.folder and k.v_layer_kind <> 'text'""").fetchall()
    by = collections.defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    n_ex = sum(p in ex for p in by)
    neither = sorted(p for p, rs in by.items() if p not in ex
                     and not any(LINE.search(r[2]) or SKETCH.search(r[2]) for r in rs))
    print(f"    `.clip` の頁(フォルダーでもテキストでもない層を持つ){len(by):,} → 恒久除外 {n_ex} → "
          f"**どちらの名前も無い頁 {len(neither):,}**(11-09 の数 2,191)")
    keys = [(p, r[1]) for p in neither for r in by[p]]
    un = split_units.units(con)
    po = page_order(con, neither)
    q = ",".join("?" * len(neither))
    st = {(r[0], r[1]): r for r in con.execute(f"""
        select l.page_path, l.layer_id, coalesce(l.name, ''), k.v_layer_kind, l.layer_type, l.colour_type_index,
               l.visibility & 1, l.opacity, l.blend_mode, l.clip_raw, l.has_mask, coalesce(l.tone_enabled, false),
               p.v_canvas_px_w, p.v_canvas_px_h, p.year
          from layers l join v_layer_kind k using (page_path, layer_id) join pages p using (page_path)
         where l.page_path in ({q})""", neither).fetchall()}
    head = ["page_path", "layer_id", "work_dir", "split_unit", "holdout_side", "year_label", "name", "ancestors", "name_rule",
            "v_layer_kind", "layer_type", "colour_type_index", "self_shown_db", "ancestors_shown_db", "opacity",
            "blend_mode", "clip_raw", "has_mask", "tone_enabled", "pos_from_top", "n_leaf_layers", "canvas_px_w", "canvas_px_h"]
    out = []
    miss = 0
    for k in keys:
        if k not in po:
            miss += 1
            continue
        s, o = st[k], po[k]
        u = un[wd(k[0])].split("\\", 2)[-1]
        out.append([k[0], k[1], wd(k[0]), u, side.get(u, ""), s[14], s[2], o[2], name_rule(s[2]),
                    s[3], s[4], s[5], s[6], o[3], s[7], s[8], s[9], int(bool(s[10])), int(bool(s[11])), o[0], o[1],
                    s[12], s[13]])
    print(f"    層 {len(keys):,} → 重なり順が引けなかった層 {miss}(印字のみ)→ 書いた {len(out):,}")
    sd = collections.Counter(r[4] or "便A の外" for r in out)
    print("    便A の単位の側(層): " + " / ".join(f"{a} {b:,}" for a, b in sd.most_common()))
    print("    種別(層): " + " / ".join(f"{a} {b:,}" for a, b in collections.Counter(r[9] for r in out).most_common()))
    print(f"    作品フォルダー {len({r[2] for r in out})} / 単位 {len({r[3] for r in out})}")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(out)
    print(f"    → `{OUT}`(非公開、{len(head)}列)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
