#!/usr/bin/env python
"""Baselines the infill model has to beat, on exactly the same masked instances.

Without these a chamfer number means nothing: the query already says which 64px
cell the stroke is in, and most missing strokes sit in a visible gap.

  B0     the corpus-median stroke (its average normalised shape at the median
         arc length), centred on the query cell
  B_near copy the nearest visible stroke, translated to the cell centre --
         "strokes look like their neighbours"
  B_gap  join the two nearest free stroke ends around the cell with a straight
         segment; falls back to B_near when there are not two ends nearby

The same dataset object and seed as `train_infill.Panels(eval_mode=True)`, so
every instance is the one the model is scored on.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_infill import CELL, POINTS, Panels, chamfer, to16


def median_stroke(ds, n=200):
    shapes, arcs = [], []
    for i in range(min(n, len(ds.rows))):
        pts, meta = ds.load(i)
        c = pts.mean(1, keepdims=True)
        span = np.linalg.norm(pts[:, -1] - pts[:, 0], axis=1) + 1e-6
        shapes.append(((pts - c) / span[:, None, None]).reshape(len(pts), -1))
        arcs.append(meta[:, 3])
    S = np.concatenate(shapes); A = np.concatenate(arcs)
    return np.median(S, 0).reshape(POINTS, 2), float(np.median(A))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="results/panel_tokens_v2_20260918")
    p.add_argument("--out", default="results/infill_20260918/baselines.csv")
    p.add_argument("--eval-panels", type=int, default=300)
    p.add_argument("--min-strokes", type=int, default=40)
    p.add_argument("--max-strokes", type=int, default=600)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260918)
    a = p.parse_args()
    rows = [r for r in csv.DictReader(open(Path(a.tokens) / "index.csv"))
            if int(r["strokes"]) >= a.min_strokes]
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(a.seed); rng.shuffle(groups)
    test_g = set(groups[: int(len(groups) * a.test_frac)])
    te = [r for r in rows if r["group"] in test_g][: a.eval_panels]
    tr = [r for r in rows if r["group"] not in test_g]
    ds_te = Panels(a.tokens, te, a.max_strokes, a.seed + 1, eval_mode=True); ds_te.f_in = 69
    ds_tr = Panels(a.tokens, tr, a.max_strokes, a.seed); ds_tr.f_in = 69
    shape, arc = median_stroke(ds_tr)
    print(f"corpus median stroke: arc {arc:.0f}px", flush=True)
    res = {"B0": [], "B_near": [], "B_gap": []}
    for i in range(len(te)):
        rng_i = np.random.default_rng(a.seed + 1 + i)
        pts, meta = ds_te.load(i)
        if len(pts) > a.max_strokes:
            k = rng_i.choice(len(pts), a.max_strokes, replace=False)
            pts, meta = pts[k], meta[k]
        t = int(rng_i.integers(len(pts)))
        target = pts[t]
        ctx = np.delete(pts, t, 0)
        mid = target.mean(0)
        cell = (np.floor(mid / CELL) + 0.5) * CELL
        res["B0"].append(chamfer(shape * arc + cell, target))
        if len(ctx) == 0:
            continue
        d = np.linalg.norm(ctx.mean(1) - cell, axis=1)
        near = ctx[int(d.argmin())]
        res["B_near"].append(chamfer(near - near.mean(0) + cell, target))
        ends = np.concatenate([ctx[:, 0], ctx[:, -1]])
        de = np.linalg.norm(ends - cell, axis=1)
        o = np.argsort(de)[:2]
        if len(o) == 2 and de[o[1]] < 3 * CELL:
            g = np.stack([np.linspace(ends[o[0]][0], ends[o[1]][0], POINTS),
                          np.linspace(ends[o[0]][1], ends[o[1]][1], POINTS)], 1)
            res["B_gap"].append(chamfer(g, target))
        else:
            res["B_gap"].append(res["B_near"][-1])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["baseline", "n", "chamfer_med", "within3", "within8"])
        print(f"{'ベースライン':<10}{'n':>6}{'chamfer中央値':>16}{'3px以内':>10}{'8px以内':>10}")
        for k, v in res.items():
            v = np.array(v)
            if not len(v):
                continue
            w.writerow([k, len(v), round(float(np.median(v)), 2), round(float((v <= 3).mean()), 4),
                        round(float((v <= 8).mean()), 4)])
            print(f"{k:<10}{len(v):>6}{np.median(v):>16.1f}{(v<=3).mean():>10.1%}{(v<=8).mean():>10.1%}")


if __name__ == "__main__":
    main()
