"""Does a binary-alpha line layer explain the missing grey edge in a tile?

doc/prereg_line_edge_binary_alpha_20261103.md. The registration is closed;
this file only implements it. No AUC: a join, a rank correlation per work,
and a within-work permutation null.
"""

import argparse
import collections
import csv
import json
import pathlib
import re

import numpy as np
from scipy.stats import rankdata

CENSUS = "inbox/line_binary_alpha_layers_for_n_20261102.csv"
MANIFEST = "data/author_process_20261101/pool_manifest.csv"
PROFILE = "results/process_stage2_20261101/profile_s2.csv"
TILES = "/home/sh1/deepl/lineart/results/clip_pairs_v3_koma_tiles_480_20260826.csv"
MIN_PAGES = 8              # registered: a work needs 8 pages with tiles
READ_BAR = 0.30            # registered: C1
BAND_WIDE = 0.60           # per-work band, printed only (amendment 2)
MIN_WORKS = 6              # registered: C5
DRAWS = 10_000
SEED = 20261103
# IC1 / IC2: values known from outside this run (census table 1, result §5).
IC1_KNOWN = {"2017/ako5": 0.225, "2017/copy": 0.551, "2017/pink": 0.240,
             "2018/ako6": 0.156, "2019/ben": 0.038, "2019/fringe": 0.053,
             "2019/fringe3": 0.063, "2019/succor": 0.029}
IC2_KNOWN = {"2017/copy": 0.63, "2017/pink": 0.68, "2017/ako5": 0.45,
             "2018/ako6": 0.65, "2019/ben": 0.74, "2019/fringe3": 0.55,
             "2019/succor": 0.59, "2019/fringe": 0.40}


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_rhos(x, y, rng, draws):
    """Spearman of y against `draws` within-work permutations of x."""
    rx, ry = rankdata(x), rankdata(y)
    ry = (ry - ry.mean()) / ry.std()
    P = np.array([rng.permutation(rx) for _ in range(draws)])
    P = (P - P.mean(1, keepdims=True)) / P.std(1, keepdims=True)
    return P @ ry / len(ry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/line_edge_20261103/line_edge.txt")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out = []
    p = out.append

    p("# 線の縁 —— 原稿の層のアルファ2値(登録 doc/prereg_line_edge_binary_alpha_20261103.md)")
    p("# 判定の単位はタイル1枚 / この track は絵を判定しない / 縁の有無は良さではない")

    # ---- the census side: per page sums ----
    page = collections.defaultdict(lambda: np.zeros(4))   # bin_px, all_px, partial, layers
    work_px = collections.defaultdict(lambda: np.zeros(2))
    for r in csv.DictReader(open(CENSUS)):
        parts = r["page_path"].split("\\")
        work = f"{parts[2]}/{parts[3]}"
        sub = "_".join(parts[4:-1]) or "root"
        n = float(r["n_px_alpha_gt0"])
        b = n if r["binary_alpha"] == "yes" else 0.0
        page[(work, sub, parts[-1])] += (b, n, float(r["n_px_partial_alpha"]), 1)
        work_px[work] += (b, n)

    p("")
    p("## IC1 x の側 —— 作品ごとの「2値の層の画素の比」を層の表から計算し直す(census 表1、±0.001)")
    ic1 = True
    for w, known in IC1_KNOWN.items():
        got = work_px[w][0] / work_px[w][1]
        ok = abs(got - known) <= 0.001     # registered tolerance
        ic1 &= ok
        p(f"  {w:16s} いま {got:.4f} / 既知 {known:.3f} → {'通った' if ok else '**外れた**'}")

    # ---- the tile side ----
    faint = {}
    for r in csv.DictReader(open(PROFILE)):
        faint[pathlib.Path(r["sample"]).stem] = float(r["faint_of_drawn_ratio"])
    spo = {r["name"].rsplit(".", 1)[0]: float(r["src_per_out"])
           for r in csv.DictReader(open(TILES))}
    man = list(csv.DictReader(open(MANIFEST)))

    p("")
    p("## IC2 y の側 —— 作品ごとの「0 のタイルの割合」(結果 §5、±0.005)")
    ic2 = True
    for w, known in IC2_KNOWN.items():
        v = np.array([faint[r["tile"]] for r in man if r["work"] == w])
        got = float(np.mean(v == 0))
        ok = abs(got - known) <= 0.005     # registered tolerance
        ic2 &= ok
        p(f"  {w:16s} いま {got:.4f} / 既知 {known:.2f} → {'通った' if ok else '**外れた**'}")

    p("")
    p("## IC4 タイルの段の縮小")
    s = np.array([spo[r["tile"]] for r in man])
    p(f"  src_per_out = 1.0 のタイル {int((s == 1.0).sum())} / {len(s)}"
      "(**コマを切る前の段は分からない**)")

    # ---- join ----
    reasons = collections.defaultdict(collections.Counter)
    tiles_of = collections.defaultdict(list)
    for r in man:
        w = r["work"]
        tag, name = w.split("/")
        stem = r["page"].rsplit(".", 1)[0]
        m = re.search(r"_" + re.escape(tag) + "_" + re.escape(name)
                      + r"__(.+)__" + re.escape(stem) + r"_p\d", r["group"])
        if not any(k[0] == w for k in page):
            reasons[w]["層の表に作品が無い"] += 1
            continue
        if not m:
            reasons[w]["群名から頁を引けない"] += 1
            continue
        key = (w, m.group(1), r["page"])
        if key not in page:
            reasons[w]["層の表に頁が無い"] += 1
            continue
        reasons[w]["結合"] += 1
        tiles_of[key].append(faint[r["tile"]])

    p("")
    p("## IC3 結合(作品ごと・理由ごと)")
    for w in sorted(reasons):
        p(f"  {w:20s} " + " / ".join(f"{k} {v}" for k, v in sorted(reasons[w].items())))
    p(f"  計 結合 {sum(c['結合'] for c in reasons.values())} / {len(man)}")

    # ---- per work: pages, x, x2, ink, y ----
    works = collections.defaultdict(list)
    for key, ys in tiles_of.items():
        bin_px, all_px, partial, _n = page[key]
        works[key[0]].append((key[1], key[2], bin_px / all_px, partial / all_px,
                              all_px, float(np.mean(np.array(ys) == 0)), len(ys)))
    pop, dropped = {}, collections.Counter()
    for w, rows in works.items():
        rows.sort(key=lambda t: (t[0], t[1]))
        y = np.array([t[5] for t in rows])
        if len(rows) < MIN_PAGES:
            dropped[f"頁が {MIN_PAGES} 未満"] += 1
            p(f"  外す {w}: 頁 {len(rows)}")
            continue
        x = np.array([t[2] for t in rows])
        if np.ptp(x) == 0:                    # amendment 1
            dropped["x が頁の間で一定(2値の層が無い)"] += 1
            p(f"  外す {w}: x が一定({x[0]:.3f})頁 {len(rows)}")
            continue
        if np.ptp(y) == 0:
            dropped["y が頁の間で一定"] += 1
            p(f"  外す {w}: y が一定({y[0]:.2f})")
            continue
        pop[w] = rows

    def read(col, shift=False):
        rhos = {}
        for w, rows in pop.items():
            x = np.array([t[col] for t in rows])
            if shift:
                x = np.roll(x, -1)
            rhos[w] = spearman(x, np.array([t[5] for t in rows]))
        return rhos

    main_r = read(2)
    nc1 = read(4)
    nc2 = read(2, shift=True)
    nc3 = read(3)

    # within-work permutation null of the median
    perm = {w: perm_rhos(np.array([t[2] for t in rows]),
                         np.array([t[5] for t in rows]), rng, DRAWS)
            for w, rows in pop.items()}
    ws = sorted(pop)
    null = np.median(np.array([perm[w] for w in ws]), axis=0) if ws else np.array([np.nan])
    real = float(np.median([main_r[w] for w in ws])) if ws else float("nan")
    pv = float((np.sum(null >= real) + 1) / (DRAWS + 1))

    p("")
    p(f"## 母集団 {len(ws)}作(外した作品: "
      + (", ".join(f"{k} {v}" for k, v in dropped.items()) or "なし") + ")")
    p("")
    p("## 作品ごと(ρ は頁の間の Spearman。帯は作品の中の置換の 2.5〜97.5%)")
    wide = 0
    for w in ws:
        lo, hi = np.percentile(perm[w], [2.5, 97.5])
        wide += (hi - lo) > BAND_WIDE
        p(f"  {w:20s} 頁 {len(pop[w]):3d} タイル {sum(t[6] for t in pop[w]):4d}"
          f"  **ρ {main_r[w]:+.3f}** 帯 [{lo:+.3f}, {hi:+.3f}]"
          f"  NC1 墨の量 {nc1[w]:+.3f} / NC2 ずらす {nc2[w]:+.3f} / NC3 縁 {nc3[w]:+.3f}"
          + ("   **帯が広い**" if (hi - lo) > BAND_WIDE else ""))

    def med(d):
        return float(np.median([d[w] for w in ws])) if ws else float("nan")

    p("")
    p("## 帰無(作品の中で x を置換、10,000回)")
    p(f"  中央 {np.median(null):+.4f} / 97.5% {np.percentile(null, 97.5):+.4f}")
    p("")
    p("## 主値")
    p(f"  **読む値 = 中央 ρ = {real:+.4f}**  片側 p = {pv:.4f}")
    p(f"  NC1 墨の量 中央 {med(nc1):+.4f} / NC2 ずらす 中央 {med(nc2):+.4f}"
      f" / NC3 縁(期待 負) 中央 {med(nc3):+.4f}")

    c4 = (not ic1) or (not ic2) or (med(nc3) >= 0 and real > 0)
    c3 = abs(med(nc1)) >= abs(real)
    # amendment 2: the per-work band is set by the page count alone, so C5
    # reads the null of the read value instead (per-work bands still printed).
    c5 = len(ws) < MIN_WORKS or float(np.percentile(null, 97.5)) >= READ_BAR
    c1 = (real >= READ_BAR and real > np.percentile(null, 97.5)
          and not c3 and not c4)
    p("")
    p("## 判定(登録 §Pass criteria)")
    p(f"  C4 器具: IC1 {'通った' if ic1 else '外れた'} / IC2 {'通った' if ic2 else '外れた'}"
      f" / NC3 の向き {med(nc3):+.4f} → **{'壊れている —— 何も読まない' if c4 else '壊れていない'}**")
    p(f"  C3 墨の量: |NC1| {abs(med(nc1)):.4f} 対 |主| {abs(real):.4f}"
      f" → **{'墨の量だった' if c3 else '墨の量ではない'}**")
    p(f"  NC2 ずらす: {med(nc2):+.4f} 対 主 {real:+.4f}"
      f" → {'主と同じ大きさ —— 頁の単位では名乗らない' if abs(med(nc2)) >= abs(real) else '主より小さい'}")
    p(f"  C5 判定保留(修正2): 母集団 {len(ws)}作(要 {MIN_WORKS}) / 読む値の帰無 97.5%"
      f" {np.percentile(null, 97.5):+.4f}(棒 {READ_BAR:+.2f} 以上で保留) / 帯が広い作品 {wide}(印字のみ)"
      f" → **{'判定保留' if c5 else '読める'}**")
    p(f"  C1 対応: 読む値 {real:+.4f} 棒 {READ_BAR:+.2f} 帰無 97.5% {np.percentile(null, 97.5):+.4f}"
      f" p {pv:.4f} → **{'C1 対応が見える' if c1 else 'C2 届かない'}**")

    txt = "\n".join(out) + "\n"
    pathlib.Path(args.out).write_text(txt)
    pathlib.Path(args.out).with_suffix(".json").write_text(json.dumps({
        "rho": main_r, "nc1": nc1, "nc2": nc2, "nc3": nc3, "read": real, "p": pv,
        "null_975": float(np.percentile(null, 97.5)), "works": ws,
        "dropped": dict(dropped), "join": {w: dict(c) for w, c in reasons.items()},
        "verdict": {"C1": bool(c1), "C3": bool(c3), "C4": bool(c4), "C5": bool(c5)},
    }, indent=1, ensure_ascii=False))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
