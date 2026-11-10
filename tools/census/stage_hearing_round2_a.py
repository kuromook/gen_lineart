#!/usr/bin/env python
"""**次の聞き取りの便A(無作為の便。棒を読む唯一の便)の頁を引く** —— Track E の発注
(`note_census_hearing_round2_design_20261110.md` §1、ユーザー承認 11-10「発注しよう」)。

**便B より先に引く**(作品は学習と評価のどちらか一方にだけ入れる。E §1)。

- 母集団: 恒久除外を除いた `.clip` の全頁(「どちらも無い」頁を含む)
- **学習側に置く単位**: 既にラベルの在る12頁の作品フォルダーが属する単位(`split_units.py`。編の裁定と中身の共有で束ねた単位)
  —— **単位ごと外す**(同じ中身を共有するフォルダーが評価に入ると学習と評価が繋がる)
- 引き方: 残りの作品フォルダーを `random.Random(SEED)` で並べ、**単位が重ならないように** 12 フォルダー、各フォルダーから1頁を一様に。
  **名前の当て方・画素・予測器を選別に使わない**
- 再掲2頁: 答えた12頁から同じ乱数で2頁、14項目の **6〜9番目**に同じ乱数で置く。位置と直前の項目を記録する
- 層: その頁の、フォルダーでない `raster` と `vector` の全部(名前に関わらず)

    ./venv/bin/python tools/census/stage_hearing_round2_a.py
"""
import collections
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402
from stage_hearing import KINDS, OUT_KEY  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 20261110
N_NEW, N_REPEAT = 12, 2
OUT = "data/census_full/hearing_round2_a_pages_20261110.csv"
HOLDOUT = "data/census_full/holdout_units_round2_a_20261110.csv"
WIP = "data/census_full/wip_pages_20261106.csv"
ORDER_OUT = "outbox/stage_layers_order_round2a_20261110.csv"


def main() -> int:
    print(f"[便A の頁(無作為の便)] seed {SEED}")
    con = db.connect()
    ex = exclusions.load()
    un = split_units.units(con)
    wip = {r["page_path"]: r["wip_status"] for r in csv.DictReader(open(os.path.join(ROOT, WIP), encoding="utf-8"))}
    pages = [p for p, in con.execute("select page_path from pages where format = 'clip' order by page_path").fetchall()]
    n_ex = sum(p in ex for p in pages)
    pages = [p for p in pages if p not in ex]
    answered = sorted({r["page_path"] for r in csv.DictReader(open(os.path.join(ROOT, OUT_KEY), encoding="utf-8"))})
    train_units = {un[wd(p)] for p in answered}
    by_wd = collections.defaultdict(list)
    for p in pages:
        by_wd[wd(p)].append(p)
    cand = sorted(w for w in by_wd if un[w] not in train_units)
    n_train_wd = sum(1 for w in by_wd if un[w] in train_units)
    print(f"    `.clip` {len(pages):,}頁(恒久除外 {n_ex} を除く)/ 作品フォルダー {len(by_wd)}")
    print(f"    学習側の単位 {len(train_units)}(ラベルの在る12頁から)→ 外した作品フォルダー {n_train_wd} / 候補 {len(cand)}")
    rng = random.Random(SEED)
    rng.shuffle(cand)
    used_u, picks = set(), []
    skipped_same_unit = 0
    for w in cand:
        if un[w] in used_u:
            skipped_same_unit += 1
            continue
        used_u.add(un[w])
        picks.append(rng.choice(sorted(by_wd[w])))
        if len(picks) == N_NEW:
            break
    print(f"    引いた {len(picks)}(単位が重なるので飛ばした作品フォルダー {skipped_same_unit})")
    rep = rng.sample(answered, N_REPEAT)
    slots = sorted(rng.sample(range(6, 10), N_REPEAT))
    items = list(picks)
    for s, p in zip(slots, rep):
        items.insert(s - 1, p)
    rows = []
    q = ",".join("?" * len(items))
    nl = dict(con.execute(f"""select l.page_path, count(*) from layers l join v_layer_kind k using (page_path, layer_id)
                              where l.page_path in ({q}) and not l.folder and k.v_layer_kind in ('raster', 'vector')
                              group by 1""", items).fetchall())
    for i, p in enumerate(items, 1):
        kind = "再掲" if p in rep and p in answered else "新"
        rows.append(dict(item=i, kind=kind, page_path=p, work_dir=wd(p), split_unit=un[wd(p)].split("\\", 2)[-1],
                         prev_item=i - 1 if i > 1 else "", prev_page=items[i - 2] if i > 1 else "",
                         n_layers_raster_vector=nl.get(p, 0), wip_status=wip.get(p, "")))
    for r in rows:
        print(f"      {r['item']:2d} {r['kind']} 層 {r['n_layers_raster_vector']:3d} {r['wip_status']}")
    n_new = sum(r["n_layers_raster_vector"] for r in rows if r["kind"] == "新")
    print(f"    **層(通常・ベクター)**: 新 12頁 {n_new} / 再掲 2頁 {sum(r['n_layers_raster_vector'] for r in rows if r['kind'] == '再掲')}"
          f" / 計 {sum(r['n_layers_raster_vector'] for r in rows)}(E の見積り ≈300)")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(ROOT, HOLDOUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split_unit", "side"])
        for u in sorted(used_u):
            w.writerow([u.split("\\", 2)[-1], "評価(便A)"])
        for u in sorted(train_units):
            w.writerow([u.split("\\", 2)[-1], "学習(ラベル済み12頁)"])
    print(f"    → `{OUT}` / 単位の側 `{HOLDOUT}`(非公開)")
    # 画素の数え上げの発注(既に数えた層は除く)
    have = {(r["page_path"], int(r["layer_id"]))
            for r in csv.DictReader(open(os.path.join(ROOT, "outbox/stage_layers_order_20261109.csv"), encoding="utf-8"))}
    order, n_have = [], 0
    for p in items:
        for lid, k in con.execute("""select l.layer_id, k.v_layer_kind from layers l join v_layer_kind k using (page_path, layer_id)
                                      where l.page_path = ? and not l.folder and k.v_layer_kind in ('raster', 'vector')
                                      order by l.layer_id""", [p]).fetchall():
            if (p, lid) in have:
                n_have += 1
            else:
                order.append((p, lid, "", "便A", k))
    with open(os.path.join(ROOT, ORDER_OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["no", "page_path", "layer_id", "name_family", "why_listed", "v_layer_kind"])
        for i, o in enumerate(order, 1):
            w.writerow([i, *o])
    print(f"    画素の発注: {len(order)}層(既に数えた {n_have} を除く)→ `{ORDER_OUT}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
