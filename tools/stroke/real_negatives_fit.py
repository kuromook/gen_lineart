#!/usr/bin/env python
"""Can the extra strokes be told from the real ones, inside one model output?

Real negatives (real_negatives.py), same three feature tiers as gate 1, split by
TILE so no drawing is in both halves. Reported per snapshot as well as overall,
because a snapshot where almost nothing lands on the GT is a different problem
from one that mostly works. "太さ濃さのみ" is the trivial baseline: an extra
stroke might simply be fainter or thinner, which would say nothing about how it
sits among its neighbours.
"""
import argparse, csv

import numpy as np
from scipy.optimize import minimize


def auc(y, s):
    o = np.argsort(s); y = np.asarray(y)[o]
    p, n = y.sum(), len(y) - y.sum()
    if not p or not n:
        return float("nan")
    r = np.arange(1, len(y) + 1)
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def fit(X, y, l2=1e-2):
    Xb = np.c_[X, np.ones(len(X))]

    def f(w):
        z = Xb @ w
        return (np.logaddexp(0, z) - y * z).mean() + l2 * (w[:-1] ** 2).sum(), \
               (Xb * ((1 / (1 + np.exp(-z))) - y)[:, None]).mean(0) + np.r_[2 * l2 * w[:-1], 0]

    r = minimize(f, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B")
    return lambda Z: np.c_[Z, np.ones(len(Z))] @ r.x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/real_negatives_20260918/candidates.csv")
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--min-rows", type=int, default=400)
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    cols = [c for c in rows[0] if c[:2] in ("a_", "b_", "c_")]
    SETS = {"A 単独": [c for c in cols if c.startswith("a_")],
            "B 粗い関係": [c for c in cols if c.startswith("b_")],
            "C 鋭い関係": [c for c in cols if c.startswith("c_")],
            "BC": [c for c in cols if c[:2] in ("b_", "c_")],
            "ABC": cols,
            "太さ濃さのみ": ["a_width", "a_width_var", "a_fill"]}
    idx = {c: i for i, c in enumerate(cols)}
    X = np.array([[float(r[c]) for c in cols] for r in rows])
    y = np.array([int(r["label"]) for r in rows], float)
    tiles = sorted({r["tile"] for r in rows})
    rng = np.random.default_rng(a.seed); rng.shuffle(tiles)
    test = set(tiles[: int(len(tiles) * a.test_frac)])
    intest = np.array([r["tile"] in test for r in rows])
    snap = np.array([r["snapshot"] for r in rows])

    def run(sel, name):
        if sel.sum() < a.min_rows:
            return
        Xs, ys, ts = X[sel], y[sel], intest[sel]
        if min(ys[ts].sum(), (1 - ys[ts]).sum(), ys[~ts].sum(), (1 - ys[~ts]).sum()) < 50:
            return
        mu, sd = Xs[~ts].mean(0), Xs[~ts].std(0) + 1e-9
        Z = (Xs - mu) / sd
        line = f"{name:<44}"
        for _k, fs in SETS.items():
            j = [idx[c] for c in fs]
            line += f"{auc(ys[ts], fit(Z[~ts][:, j], ys[~ts])(Z[ts][:, j])):>12.3f}"
        best = max(((abs(auc(ys[ts], Z[ts][:, idx[c]]) - 0.5) + 0.5, c) for c in cols))
        print(line + f"{best[1] + ' ' + format(best[0], '.3f'):>24}  (n={int(sel.sum())}, 本物{ys.mean():.0%})")

    print(f"rows {len(rows)}  tiles {len(tiles)}  本物 {int(y.sum())} / 余計 {int((1 - y).sum())}")
    print(f"{'群':<44}" + "".join(f"{s:>12}" for s in SETS) + f"{'最良の単一特徴':>24}")
    run(np.ones(len(rows), bool), "全体")
    for s in sorted(set(snap)):
        run(snap == s, s.replace("results/controlnet_lora_manga_consistency_", "cnet/"))


if __name__ == "__main__":
    main()
