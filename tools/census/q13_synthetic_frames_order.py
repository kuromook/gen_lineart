#!/usr/bin/env python
"""**Q13(段2)の器具の点検の材料の頁 200**(Track Q の依頼、2026-11-22。census が頁を引き、抽出側へ発注する一覧を作る)。

母集団(述語): 構造 DB(10-26 の断面)の `.clip` の頁のうち `v_layer_kind = frame_folder` の層が1つ以上 − 恒久除外。
**コマの形が1つも無い頁は引かずに数える**: `koma_frames` の、名前に `モザイク` を含まないコマ枠のフォルダーの行の
`n_components` の和が 0(行が無い・誤りの行だけ・形 0 を理由別に)。
**DB の `koma_frames` は 2023b 以降の頁がどれも「マスクが読めない」**(`コマ 1` ほか)—— その頁は DB のマスクで IC1 が引けないので、引けない頁として数える。

引き方(Q の依頼のとおり。読み方は census が決めた):
- 作品フォルダー = `p_split_columns.wd`(路の先頭4段)。**種 20261122 で作品フォルダーを並べ替え、前半 = `tune` / 後半 = `check`**(半々。奇数なら check が1多い)
- 年代札 = DB の `pages.year`。札ごとに頁を種 20261122 で並べ替え、**札を名前の順に1頁ずつ回して**取る。1作品フォルダー3頁まで。200頁で止める
- Q12 の陽性の対照(枠らしさの一覧の `positive_frame_folder` の頁 94)と重なってよい。重なった数を印字する
- 清書の層(`name_rule` = 清書、11-14 の画素の表で `n_px_a_gt0` > 0、フォルダー・テキストでない)の数を頁ごとに付ける(0 なら `art.png` を作らない)

出力 `outbox/q13_synthetic_frames_order_20261122.csv`(頁の路を持つ。追跡しない)。
"""
import collections
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
from p_split_columns import wd  # noqa: E402
from stage_hearing import name_rule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 20261122
N = 200
PER_WORK = 3
MOZA = "モザイク"
PIX = os.path.join(ROOT, "data", "census_full", "full_clip_pixels_layers_merged.csv")
POS = os.path.join(ROOT, "outbox", "q_frame_likeness_order_20261120.csv")
OUT = os.path.join(ROOT, "outbox", "q13_synthetic_frames_order_20261122.csv")


def main() -> int:
    c = db.connect()
    ex = exclusions.load()
    year = {p: y for p, f, y in c.execute("select page_path, format, year from pages").fetchall() if f == "clip"}
    ff = {p for (p,) in c.execute("select distinct page_path from v_layer_kind where v_layer_kind = 'frame_folder'").fetchall()}
    P0 = sorted(p for p in ff if p in year)
    n_ex = sum(p in ex for p in P0)
    P = [p for p in P0 if p not in ex]
    print(f".clip でコマ枠のフォルダーが在る頁 {len(P0):,} − 恒久除外 {n_ex} = {len(P):,}")
    ffl = collections.defaultdict(set)
    for p, i, n in c.execute("select l.page_path, l.layer_id, coalesce(l.name,'') from layers l join v_layer_kind k "
                             "using (page_path, layer_id) where k.v_layer_kind = 'frame_folder'").fetchall():
        if MOZA not in n:
            ffl[p].add(i)
    kf = collections.defaultdict(list)
    for p, i, n, nc, err in c.execute("select page_path, layer_id, coalesce(name,''), n_components, error from koma_frames").fetchall():
        kf[p].append((i, n, nc, err))
    shapes, drop = {}, collections.Counter()
    for p in P:
        rows = [r for r in kf.get(p, []) if r[0] in ffl[p]]
        if not ffl[p]:
            drop["コマ枠のフォルダーが モザイク の名前だけ"] += 1; continue
        if not rows:
            drop["koma_frames に行が無い"] += 1; continue
        ok = [r for r in rows if not r[3]]
        if not ok:
            drop["koma_frames がマスクが読めない行だけ(DB の誤り。枠が空とは別)"] += 1; continue
        s = sum(r[2] or 0 for r in ok)
        if s == 0:
            drop["コマの形が 0(フォルダーはあるが枠が空)"] += 1; continue
        shapes[p] = (s, sum(1 for r in rows if r[3]))
    print("引かずに数えた頁: " + " / ".join(f"{k} {v}" for k, v in drop.items()) + f" → 残り {len(shapes):,}")
    works = sorted({wd(p) for p in shapes})
    rng = random.Random(SEED)
    w2 = works[:]; rng.shuffle(w2)
    half = len(w2) // 2
    split = {w: ("tune" if i < half else "check") for i, w in enumerate(w2)}
    by_year = collections.defaultdict(list)
    for p in sorted(shapes):
        by_year[year[p]].append(p)
    for y in sorted(by_year):
        rng.shuffle(by_year[y])
    chosen, per_w, idx = [], collections.Counter(), {y: 0 for y in by_year}
    while len(chosen) < N and any(idx[y] < len(by_year[y]) for y in by_year):
        for y in sorted(by_year):
            if len(chosen) >= N:
                break
            while idx[y] < len(by_year[y]):
                p = by_year[y][idx[y]]; idx[y] += 1
                if per_w[wd(p)] < PER_WORK:
                    chosen.append(p); per_w[wd(p)] += 1; break
    # 清書の層の数
    cs = set(chosen)
    pix = {}
    for r in csv.DictReader(open(PIX, encoding="utf-8")):
        if r["page_path"] in cs:
            pix[(r["page_path"], int(r["layer_id"]))] = int(r["n_px_a_gt0"] or 0)
    q = ",".join("?" * len(chosen))
    nline = collections.Counter()
    for p, i, n, fo, k in c.execute(f"select l.page_path, l.layer_id, coalesce(l.name,''), l.folder, v.v_layer_kind from layers l "
                                     f"join v_layer_kind v using (page_path, layer_id) where l.page_path in ({q})", chosen).fetchall():
        if fo or k in ("folder", "frame_folder", "text"):
            continue
        if name_rule(n) == "清書" and pix.get((p, i), 0) > 0:
            nline[p] += 1
    pos = {r["page_path"] for r in csv.DictReader(open(POS, encoding="utf-8")) if r["set"] == "positive_frame_folder"}
    rows = []
    for j, p in enumerate(sorted(chosen, key=lambda x: (split[wd(x)], year[x], x)), 1):
        rows.append(dict(q=f"q13_{j:03d}", split=split[wd(p)], page_path=p, work_dir=wd(p), year_label=year[p],
                         n_panels_db=shapes[p][0], n_frame_folder_error_rows=shapes[p][1],
                         n_line_layers_with_px=nline[p], in_q12_positive=int(p in pos)))
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"\n作品フォルダー {len(works)}: tune {sum(v == 'tune' for v in split.values())} / check {sum(v == 'check' for v in split.values())}")
    print(f"引いた頁 {len(rows)}(年代札 {len(by_year)}、1作品フォルダー {PER_WORK} 頁まで)")
    for s in ("tune", "check"):
        rs = [r for r in rows if r["split"] == s]
        print(f"  {s}: 頁 {len(rs)} / 作品フォルダー {len({r['work_dir'] for r in rs})} / 清書の層が無い頁 {sum(r['n_line_layers_with_px'] == 0 for r in rs)}")
    print("  札ごと: " + " / ".join(f"{y} {n}" for y, n in sorted(collections.Counter(r["year_label"] for r in rows).items())))
    print(f"  Q12 の陽性の対照 {len(pos)} 頁と重なった頁 {sum(r['in_q12_positive'] for r in rows)}")
    print(f"  清書の層が無い頁(art.png を作らない) {sum(r['n_line_layers_with_px'] == 0 for r in rows)}")
    print(f"  コマの数(DB): 計 {sum(r['n_panels_db'] for r in rows):,} / 頁の中央 {sorted(r['n_panels_db'] for r in rows)[len(rows)//2]}")
    print(f"出力: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
