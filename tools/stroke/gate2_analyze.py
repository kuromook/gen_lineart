#!/usr/bin/env python
"""Gate 2 readout: stroke axes vs the existing pixel axes, on the same images.

For every model snapshot, AUC separating GT line art from that snapshot's 192
tiles, computed for (a) each single stroke axis, (b) each single pixel axis from
measure_lineart_profile, (c) a logistic model over each family, fitted on half
the tiles and read on the other half (split by tile id, shared across groups).

A measure that only separates is not enough: a snapshot at step 10,000 should
also sit closer to GT than one at step 1,000 if the measure tracks quality, so
the mean model score per snapshot is printed as well.
"""
import argparse, csv, re
from collections import defaultdict
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
    p.add_argument("--csv", default="results/gate2_20260918/profiles.csv")
    p.add_argument("--seed", type=int, default=20260918)
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    scols = [c for c in rows[0] if c[:2] in ("s_", "r_")]
    pcols = [c for c in rows[0] if c.startswith("p_")]

    def val(r, c):
        v = r[c]
        try:
            f = float(v)
        except ValueError:
            return np.nan
        return f if np.isfinite(f) else np.nan

    tile = lambda r: re.sub(r"_out\.png$|\.jpg$|\.png$", "", r["image"].split("/")[-1])
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    gt = by["GT"]
    tiles = sorted({tile(r) for r in gt})
    rng = np.random.default_rng(a.seed); rng.shuffle(tiles)
    test = set(tiles[: len(tiles) // 2])

    def matrix(rs, cols):
        X = np.array([[val(r, c) for c in cols] for r in rs])
        return X

    Xg_s, Xg_p = matrix(gt, scols), matrix(gt, pcols)
    tg = np.array([tile(r) in test for r in gt])
    print(f"{'snapshot':<46}{'stroke best':>22}{'pixel best':>24}{'stroke fit':>12}{'pixel fit':>11}{'both':>8}")
    summary = []
    for g in sorted(by):
        if g == "GT":
            continue
        rs = by[g]
        Xs, Xp = matrix(rs, scols), matrix(rs, pcols)
        ts = np.array([tile(r) in test for r in rs])
        X = {"s": (np.r_[Xg_s, Xs], scols), "p": (np.r_[Xg_p, Xp], pcols)}
        y = np.r_[np.ones(len(gt)), np.zeros(len(rs))]
        intest = np.r_[tg, ts]
        best = {}
        scores = {}
        for tag, (M, cols) in X.items():
            M = np.where(np.isfinite(M), M, np.nan)
            mu = np.nanmean(M[~intest], 0); sd = np.nanstd(M[~intest], 0) + 1e-9
            Z = np.where(np.isfinite(M), (M - mu) / sd, 0.0)
            b = max(((abs(auc(y, Z[:, i]) - 0.5) + 0.5, cols[i]) for i in range(len(cols))))
            best[tag] = b
            sc = fit(Z[~intest], y[~intest])
            scores[tag] = (auc(y[intest], sc(Z[intest])), float(sc(Z[len(gt):]).mean()))
        Zb = np.c_[X["s"][0], X["p"][0]]
        Zb = np.where(np.isfinite(Zb), Zb, np.nan)
        mu = np.nanmean(Zb[~intest], 0); sd = np.nanstd(Zb[~intest], 0) + 1e-9
        Zb = np.where(np.isfinite(Zb), (Zb - mu) / sd, 0.0)
        both = auc(y[intest], fit(Zb[~intest], y[~intest])(Zb[intest]))
        name = g.replace("results/controlnet_lora_manga_consistency_", "cnet/")
        print(f"{name:<46}{best['s'][1]+' '+format(best['s'][0],'.3f'):>22}"
              f"{best['p'][1]+' '+format(best['p'][0],'.3f'):>24}"
              f"{scores['s'][0]:>12.3f}{scores['p'][0]:>11.3f}{both:>8.3f}")
        summary.append((name, scores["s"], scores["p"]))
    print("\nmean model score per snapshot (higher = more GT-like)")
    print(f"{'snapshot':<46}{'stroke':>10}{'pixel':>10}")
    for name, s, pp in summary:
        print(f"{name:<46}{s[1]:>10.2f}{pp[1]:>10.2f}")


if __name__ == "__main__":
    main()
