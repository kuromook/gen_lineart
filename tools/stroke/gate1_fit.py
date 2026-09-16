#!/usr/bin/env python
"""Gate 1: can a stroke's fit be told from its relations to the drawing?

Logistic regression (L-BFGS, L2), split BY TILE -- the same drawing must never
appear in both halves, or the context leaks.

Reported per negative kind, never pooled into one headline number, because the
kinds are not equally hard. Feature sets are ablated:

  A    the stroke alone      a_*
  B    its relations         b_*
  AB   both
  ART  a_curv + a_len only -- the two columns with residual generation
       artifacts (re-rasterizing a rotated polyline adds staircase jitter and
       can shorten it). Whatever ART scores on a kind is the floor that kind's
       other numbers must be read against.

DISPLACED is the gate: its a_* columns are distributionally identical to the
positives by construction (same pixels, shifted), so anything above 0.5 there
comes from relations. Gate 1 asks for AUC >= 0.80 on displaced.
"""
import argparse, csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

KINDS = ("displaced", "rotated", "foreign")


def auc(y, s):
    r = rankdata(s)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def fit(X, y, l2=1.0):
    X = np.hstack([X, np.ones((len(X), 1))])

    def obj(w):
        z = X @ w
        ll = np.logaddexp(0, z).sum() - (y * z).sum()
        g = X.T @ (1 / (1 + np.exp(-z)) - y)
        pen = np.r_[w[:-1], 0.0]
        return ll + 0.5 * l2 * (pen @ pen), g + l2 * pen

    w = minimize(obj, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
    return lambda Z: np.hstack([Z, np.ones((len(Z), 1))]) @ w


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="results/gate1_20260917/candidates.csv")
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--drop-on-other", type=float, default=0.5,
                   help="drop negatives sitting mostly on ANOTHER real stroke (false negatives)")
    a = p.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    cols = [c for c in rows[0] if c[:2] in ("a_", "b_")]
    A = [c for c in cols if c.startswith("a_")]
    B = [c for c in cols if c.startswith("b_")]
    SETS = {"A (単独)": A, "B (関係)": B, "AB": cols, "ART (副産物のみ)": ["a_curv", "a_len"]}

    tiles = sorted({r["tile"] for r in rows})
    rng = np.random.default_rng(a.seed)
    rng.shuffle(tiles)
    ntest = int(len(tiles) * a.test_frac)
    test = set(tiles[:ntest])
    print(f"rows {len(rows)}  tiles {len(tiles)}  (test {len(test)} / train {len(tiles)-len(test)})")

    X = np.array([[float(r[c]) for c in cols] for r in rows])
    kind = np.array([r["kind"] for r in rows])
    oo = np.array([float(r["on_other"]) for r in rows])
    intest = np.array([r["tile"] in test for r in rows])
    idx = {c: i for i, c in enumerate(cols)}

    for filt in (False, True):
        keep = np.ones(len(rows), bool) if not filt else ((kind == "true") | (oo <= a.drop_on_other))
        tag = "全候補" if not filt else f"on_other<={a.drop_on_other} に限定"
        print(f"\n===== {tag} =====")
        print(f"{'負例の種類':<14}" + "".join(f"{s:>18}" for s in SETS) + f"{'最良の単一特徴':>26}")
        for k in KINDS:
            sel = keep & ((kind == "true") | (kind == k))
            y = (kind[sel] == "true").astype(float)
            Xs, ts = X[sel], intest[sel]
            mu, sd = Xs[~ts].mean(0), Xs[~ts].std(0) + 1e-9
            Z = (Xs - mu) / sd
            line = f"{k:<14}"
            for name, fs in SETS.items():
                j = [idx[c] for c in fs]
                sc = fit(Z[~ts][:, j], y[~ts])(Z[ts][:, j])
                line += f"{auc(y[ts], sc):>18.3f}"
            best = max(((abs(auc(y[ts], Z[ts][:, idx[c]]) - 0.5) + 0.5, c) for c in cols))
            line += f"{best[1] + ' ' + format(best[0], '.3f'):>26}"
            print(line + f"   (n={int(sel.sum())})")


if __name__ == "__main__":
    main()
