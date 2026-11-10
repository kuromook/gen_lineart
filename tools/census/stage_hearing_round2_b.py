#!/usr/bin/env python
"""**便B(学習専用の便。棒を読まない)の頁を引き、20行ずつの束の聞き取りの表を作る** —— Track E の決定
(`note_census_b_hypotheses_measured_20261110.md` §2、`note_census_exclude_axis_and_20row_split_20261110.md` §2・§3)。

**引く前に決めたこと(札に書く)**:
- 母集団: 層の役割の表 v2(`layer_role_table_20261110_v2.csv`、53,289層)のうち、
  **便A の評価12単位を外し、ラベルの在る12頁も外した残り**(E の歩留まりと同じ切り方)
- 層別は3つ。**名前を使わない**(E の決定):
  - **芯**: 覆い率 `n_px_a_gt0 / (canvas_px_w × canvas_px_h)` > 0.5 **かつ** 薄さ `n_px_a_ge26 / n_px_a_gt0` < 0.25 の層を持つ頁 → 4頁
    (**閾値は E がラベル19層を見て決めた —— 当て込み**。便A では使わない)
  - **広**: 祖先フォルダーが非表示(`ancestors_shown_db = 0`)の層を持つ頁 → 4頁
  - **描画色**: `draw_color_enable = 1` の層を持つ頁 → 2頁(**この軸を測れるようにするため**。E)
- **1単位1頁**(`split_units` の単位。10頁で単位が重ならない)。層別の順(芯 → 広 → 描画色)に、
  頁を `random.Random(20261110)` で並べて先頭から取る。前の層別で使った単位は飛ばす(飛ばした数を印字)
- 頁が複数の層別に当たっても、引いた層別の札だけを書き、他に当たる層別も別の列に印字する
- 層: その頁の、フォルダーでない `raster` と `vector` の全部(便A と同じ。名前・層別に関わらず)
- **束**: 頁の順のまま、20行を超えない所で切る。**頁の途中では切らない**。1頁で20行を超える頁は単独の束(例外、印字)
- 問いは軸1(工程の段)だけ。最初の束は試し

    ./venv/bin/python tools/census/stage_hearing_round2_b.py
"""
import collections
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402
from stage_hearing import OUT_KEY, name_rule  # noqa: E402
from stage_hearing_round2_a_sheets import LIMIT, layers_top_down, write_xlsx  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLE = "data/census_full/layer_role_table_20261110_v2.csv"
HOLDOUT = "data/census_full/holdout_units_round2_a_20261110.csv"
SEED = 20261110
STRATA = [("芯", 4), ("広", 4), ("描画色", 2)]
OUT = "data/census_full/hearing_round2_b_pages_20261110.csv"
KEY_OUT = "data/census_full/hearing_round2_b_key_20261110.csv"
XLSX = "data/census_full/hearing_round2_b_bundle{b}_20261110.xlsx"


def f(x):
    return float(x) if x not in ("", None) else None


def main() -> int:
    print(f"[便B の頁(学習専用の便)] seed {SEED}")
    con = db.connect()
    un = split_units.units(con)
    ev = {r["split_unit"] for r in csv.DictReader(open(os.path.join(ROOT, HOLDOUT), encoding="utf-8"))
          if r["side"].startswith("評価")}
    labelled = {r["page_path"] for r in csv.DictReader(open(os.path.join(ROOT, OUT_KEY), encoding="utf-8"))}
    hits = collections.defaultdict(set)
    n_all, drop = 0, collections.Counter()
    pages_all = set()
    for r in csv.DictReader(open(os.path.join(ROOT, TABLE), encoding="utf-8")):
        n_all += 1
        if r["split_unit"] in ev:
            drop["便A の評価単位"] += 1
            continue
        if r["page_path"] in labelled:
            drop["ラベルの在る12頁"] += 1
            continue
        p = r["page_path"]
        pages_all.add(p)
        gt0, ge26, w, h = f(r["n_px_a_gt0"]), f(r["n_px_a_ge26"]), f(r["canvas_px_w"]), f(r["canvas_px_h"])
        if gt0 and w and h and gt0 / (w * h) > 0.5 and ge26 / gt0 < 0.25:
            hits[p].add("芯")
        if r["ancestors_shown_db"] == "0":
            hits[p].add("広")
        if r["draw_color_enable"] == "1":
            hits[p].add("描画色")
    print(f"    表 {n_all:,}層 → 外した " + " / ".join(f"{a} {b:,}" for a, b in drop.items())
          + f" → 残り {n_all - sum(drop.values()):,}層 / {len(pages_all):,}頁")
    for s, _ in STRATA:
        ps = [p for p in hits if s in hits[p]]
        print(f"      {s}: 当たる頁 {len(ps):,} / 単位 {len({un[wd(p)] for p in ps})}")
    rng = random.Random(SEED)
    used, picks = set(), []
    for s, k in STRATA:
        cand = sorted(p for p in hits if s in hits[p])
        rng.shuffle(cand)
        got = skipped = 0
        for p in cand:
            u = un[wd(p)]
            if u in used:
                skipped += 1
                continue
            used.add(u)
            picks.append((s, p, u))
            got += 1
            if got == k:
                break
        print(f"    {s}: 引いた {got}(使用済みの単位で飛ばした頁 {skipped})")
    q = ",".join("?" * len(picks))
    ls = con.execute(f"""select l.page_path, l.layer_id, l.parent_id, l.order_in_parent, coalesce(l.name, ''),
                                l.folder, l.visibility, k.v_layer_kind
                           from layers l join v_layer_kind k using (page_path, layer_id)
                          where l.page_path in ({q})""", [p for _, p, _ in picks]).fetchall()
    by_page = collections.defaultdict(list)
    for r in ls:
        by_page[r[0]].append(r)
    rows, items = [], []
    for i, (s, p, u) in enumerate(picks, 1):
        lay, _ = layers_top_down(by_page[p])
        items.append(dict(item=i, stratum=s, other_strata="/".join(sorted(hits[p] - {s})), page_path=p, split_unit=u,
                          n_layers_raster_vector=len(lay), wip_status=""))
        for x in lay:
            rows.append(dict(item=i, kind_item=s, page_path=p, wip="", rule=name_rule(x["name"]), **x))
    # 束: 頁の順のまま、20行を超えない所で切る
    bundles, cur, n = [], [], 0
    for it in items:
        m = it["n_layers_raster_vector"]
        if cur and n + m > LIMIT:
            bundles.append(cur)
            cur, n = [], 0
        cur.append(it["item"])
        n += m
    bundles.append(cur)
    names = [(f"B-{j}", tuple(b)) for j, b in enumerate(bundles, 1)]
    for it in items:
        print(f"      項目 {it['item']}: {it['stratum']}" + (f"(他に {it['other_strata']})" if it["other_strata"] else "")
              + f" / {it['n_layers_raster_vector']}行")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(items[0]))
        w.writeheader()
        w.writerows(items)
    bmap = {i: b for b, its in names for i in its}
    with open(os.path.join(ROOT, KEY_OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bundle"] + list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(dict(bundle=bmap[r["item"]], **r))
    for b, its in names:
        sub = [r for r in rows if r["item"] in its]
        over = "  **20行を超える(1頁で切れない)**" if len(sub) > LIMIT else ""
        print(f"    {b}: 項目 {', '.join(map(str, its))} → {len(sub)}行{over}")
        write_xlsx(b, its, items, sub, None, label="便B", n_bundles=len(names), path=XLSX, trial=(b == "B-1"))
    print(f"    計 {len(rows)}行 / {len(names)}束 → `{OUT}` / 鍵 `{KEY_OUT}` / 表 `{XLSX.format(b='B-1')}` ほか(非公開)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
