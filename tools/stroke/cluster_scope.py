#!/usr/bin/env python
"""Does a stroke's own cluster say where it is?

This is what decides whether a loose grouping is good enough to carry the infill
question. The first infill run had no unit, so it was told which 64px cell the
missing stroke sat in -- an artificial scope whose own error is 24.6px (measured
2026-09-19, results/infill_eval_20260919/). If the centroid of a stroke's
cluster, computed WITHOUT that stroke, lands closer than that, the cluster is a
real scope and B can be re-posed inside it; if not, the grouping buys nothing.

Same held-out panels and same per-instance seeds as train_infill's eval, so the
numbers sit beside the ones already recorded.
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stroke_graph import build_edges, cap_degree, communities, load_panel
from train_infill import CELL, POINTS

MAX_STROKES = 600


def bbox_diag(pts):
    if len(pts) == 0:
        return float("nan")
    f = pts.reshape(-1, 2)
    return float(np.linalg.norm(f.max(0) - f.min(0)))


def work(t):
    pack, row, rule, resolution, i, seed = t
    arr = np.load(Path(pack) / "strokes.npy", mmap_mode="r")
    pts, meta = load_panel(arr, int(row["start"]), int(row["n"]))
    rng = np.random.default_rng(seed + i)
    if len(pts) > MAX_STROKES:
        k = rng.choice(len(pts), MAX_STROKES, replace=False)
        pts, meta = pts[k], meta[k]
    n = len(pts)
    tgt = int(rng.integers(n))
    t0 = time.time()
    lab = communities(cap_degree(build_edges(pts, meta, rule), n), n, resolution, 0)
    cent = pts.mean(1)
    true_c = cent[tgt]
    mates = np.flatnonzero((lab == lab[tgt]) & (np.arange(n) != tgt))
    others = np.flatnonzero(np.arange(n) != tgt)
    cell = (np.floor(true_c / CELL) + 0.5) * CELL
    out = {"name": row["name"], "group": row["group"], "rule": rule, "resolution": resolution,
           "strokes": n, "target": tgt, "arc": float(meta[tgt, 3]),
           "n_mates": int(len(mates)),
           "cell_err": float(np.linalg.norm(cell - true_c)),
           "panel_err": float(np.linalg.norm(cent[others].mean(0) - true_c)),
           "panel_extent": bbox_diag(pts[others]),
           "sec": round(time.time() - t0, 2)}
    if len(mates):
        out["cluster_err"] = float(np.linalg.norm(cent[mates].mean(0) - true_c))
        out["cluster_extent"] = bbox_diag(pts[mates])
        # a centroid is a poor summary of a 300px-wide cluster, so also record how
        # close the cluster actually reaches: the nearest mate, by centroid and by
        # ink. A model conditioned on the mates can use these; it cannot use the
        # query cell, which is the question-setter handing over the answer.
        out["near_mate_err"] = float(np.linalg.norm(cent[mates] - true_c, axis=1).min())
        d = np.linalg.norm(pts[mates].reshape(-1, 2) - true_c, axis=1)
        out["near_mate_ink"] = float(d.min())
    else:
        out["cluster_err"] = float("nan")
        out["cluster_extent"] = float("nan")
        out["near_mate_err"] = float("nan")
        out["near_mate_ink"] = float("nan")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="results/panel_pack_20260919")
    p.add_argument("--out", default="results/stroke_clusters_20260919/cluster_scope.csv")
    p.add_argument("--rules", default="width:0.5,arc:0.5,knn:2.0")
    p.add_argument("--panels", type=int, default=300)
    p.add_argument("--min-strokes", type=int, default=40)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260919)
    a = p.parse_args()
    rows = [r for r in csv.DictReader(open(Path(a.pack) / "panels.csv"))
            if int(r["n"]) >= a.min_strokes]
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(20260918); rng.shuffle(groups)
    test = set(groups[: int(len(groups) * 0.25)])
    te = [r for r in rows if r["group"] in test][: a.panels]
    print(f"held-out panels {len(te)}", flush=True)
    allrows = []
    for spec in a.rules.split(","):
        rule, res = spec.split(":"); res = float(res)
        tasks = [(a.pack, r, rule, res, i, a.seed) for i, r in enumerate(te)]
        t0 = time.time()
        with Pool(a.workers) as pool:
            got = list(pool.imap_unordered(work, tasks, chunksize=4))
        allrows += got
        g = [x for x in got if np.isfinite(x["cluster_err"])]
        g3 = [x for x in got if x["n_mates"] >= 3]
        med = lambda rows_, k: float(np.median([x[k] for x in rows_])) if rows_ else float("nan")
        print(f"{rule}:{res}  塊あり {len(g)/len(got):.0%}  3本以上の仲間 {len(g3)/len(got):.0%}  "
              f"塊の重心誤差 中央値 {med(g, 'cluster_err'):>5.1f}px "
              f"(仲間3本以上なら {med(g3, 'cluster_err'):>5.1f}px)  "
              f"最寄りの仲間 {med(g, 'near_mate_err'):>5.1f}px  "
              f"塊の広がり {med(g, 'cluster_extent'):>6.0f}px  "
              f"マス中心 {med(got, 'cell_err'):>5.1f}px  コマ全体 {med(got, 'panel_err'):>6.1f}px  "
              f"{time.time()-t0:.0f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(allrows[0])); w.writeheader(); w.writerows(allrows)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
