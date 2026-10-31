"""Twelve axes at once, read by transfer.

doc/prereg_multivariate_transfer_20261031.md. The registration is closed; this
file only implements it.

The question is whether a discriminator trained on A1 (the author's
analogue-dominant pool) against A3 (external SDXL) puts A2 (the author's own
digital line art) on the author's side. One axis at a time could not answer it
because the author axis and the process axis rode on the same difference.

Fixed by the registration, not chosen here:
  F9  = the nine axes the bars are read on (the two gate axes and ink_ratio out)
  F10 = F9 + ink_ratio          F12 = all twelve
  A3 is split in half by SHA-1 of the group name: H1 trains, H2 tests, always.
  Standardization uses the TRAINING side's mean and sd only -- re-standardizing
  per pool would erase the very shift being measured.
  C comes from {0.01, 0.1, 1, 10} by group CV inside the training side.
  No transforms, no class weights, no random splits.

Logistic regression is written out here with scipy's L-BFGS-B because this venv
has no sklearn and the shared venv is not ours to change.
"""

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compute_auc import AXES, auc  # noqa: E402
from compute_auc_three_arms import group_of, row_of  # noqa: E402

GATE_AXES = ("line_width_p50", "width_consistency")   # the one-sided gate (条例4)
NULL_AXIS = "ink_ratio"                                # the null arm (条例3)

F9 = [a for a in AXES if a not in GATE_AXES and a != NULL_AXIS]
F10 = [a for a in AXES if a not in GATE_AXES]
F12 = list(AXES)
FEATURE_SETS = {"F9": F9, "F10": F10, "F12": F12}

C_GRID = (0.01, 0.1, 1.0, 10.0)
CV_FOLDS = 5
PERM_SEED = 20261031
A1, A2 = "a1_author_analogue", "a2_author_digital"

# IC-1: the known values, written into the registration before this ran.
KNOWN_AUC13 = {
    "line_width_p50": 0.9788, "background_ratio": 0.7644,
    "faint_mean_dist_to_ink": 0.8635, "faint_of_drawn_ratio": 0.7718,
    "long_line_ratio": 0.7415, "width_consistency": 0.5409,
    "deep_black_ratio": 0.0451, "orientation_entropy": 0.3989,
    "faint_near_ink_ratio": 0.2087, "long_component_ratio": 0.2397,
    "components_per_1k_ink_px": 0.1816, "ink_ratio": 0.1799,
}
KNOWN_AUC12_FAINT = 0.9900          # IC-5, faint_of_drawn_ratio


def digest_mod(name, m):
    """The only splitter in this file: SHA-1 of the group name, mod m."""
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % m


class Pool:
    """Tiles with their arm label, their printed row, and their group."""

    def __init__(self, X, y, g, r):
        self.X, self.y, self.g, self.r = X, y, g, r

    def __len__(self):
        return len(self.y)

    def take(self, mask):
        return Pool(self.X[mask], self.y[mask], self.g[mask], self.r[mask])


def load(path):
    arms = {}
    rows = collections.defaultdict(list)
    for rec in csv.DictReader(open(path)):
        rows[rec["label"]].append(rec)
    for label, recs in rows.items():
        X = np.array([[float(r[a]) for a in AXES] for r in recs], dtype=float)
        g = np.array([group_of(label, r["sample"]) for r in recs])
        r = np.array([row_of(label, r["sample"]) for r in recs])
        arms[label] = {"X": X, "g": g, "r": r}
    return arms


def arm_pool(arm, y, keep=None):
    """One arm as a Pool with label y; keep is a boolean mask over its tiles."""
    X, g, r = arm["X"], arm["g"], arm["r"]
    if keep is None:
        keep = np.ones(len(g), dtype=bool)
    return Pool(X[keep], np.full(keep.sum(), y, dtype=float), g[keep], r[keep])


def join(*pools):
    return Pool(
        np.concatenate([p.X for p in pools]),
        np.concatenate([p.y for p in pools]),
        np.concatenate([p.g for p in pools]),
        np.concatenate([p.r for p in pools]),
    )


def columns(pool, feats):
    idx = [AXES.index(a) for a in feats]
    return pool.X[:, idx]


def standardize(train, test):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd == 0] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def fit_logreg(X, y, C):
    """L2 logistic regression; the intercept is not penalised."""
    n, d = X.shape

    def obj(theta):
        w, b = theta[:d], theta[d]
        z = X @ w + b
        # log(1 + exp(-y'z)) with y' in {-1, +1}, written stably
        ys = 2.0 * y - 1.0
        m = -ys * z
        loss = np.sum(np.where(m > 0, m + np.log1p(np.exp(-np.abs(m))),
                               np.log1p(np.exp(m))))
        p = 1.0 / (1.0 + np.exp(-z))
        grad_z = p - y
        grad = np.empty(d + 1)
        grad[:d] = X.T @ grad_z + w / C
        grad[d] = grad_z.sum()
        return loss + 0.5 * np.dot(w, w) / C, grad

    res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    return res.x[:d], res.x[d]


def score_auc(scores, y):
    return auc(scores[y == 1], scores[y == 0])


def select_C(pool, feats):
    """Group CV inside the training side only, over the registered grid."""
    folds = np.array([digest_mod(g, CV_FOLDS) for g in pool.g])
    best, best_auc = None, -1.0
    per_C = {}
    for C in C_GRID:
        aucs = []
        for f in range(CV_FOLDS):
            tr, te = folds != f, folds == f
            if len(set(pool.y[tr])) < 2 or len(set(pool.y[te])) < 2:
                continue
            Xtr, Xte = standardize(columns(pool.take(tr), feats),
                                   columns(pool.take(te), feats))
            w, b = fit_logreg(Xtr, pool.y[tr], C)
            aucs.append(score_auc(Xte @ w + b, pool.y[te]))
        per_C[C] = float(np.mean(aucs)) if aucs else float("nan")
        if aucs and per_C[C] > best_auc:
            best, best_auc = C, per_C[C]
    return best, per_C


def train_test(train, test, feats, C=None):
    """The main arm M3/M4: fit on train, score test. Returns the full record."""
    if C is None:
        C, per_C = select_C(train, feats)
    else:
        per_C = None
    Xtr, Xte = standardize(columns(train, feats), columns(test, feats))
    w, b = fit_logreg(Xtr, train.y, C)
    s = Xte @ w + b
    return {
        "C": C, "C_cv": per_C, "auc": score_auc(s, test.y),
        "coef": dict(zip(feats, w.round(4).tolist())),
        "n_train": [int((train.y == 1).sum()), int((train.y == 0).sum())],
        "n_test": [int((test.y == 1).sum()), int((test.y == 0).sum())],
        "scores": s, "test": test,
    }


def best_single_axis(train, test, feats):
    """M2, the existing tool with no model. The axis is chosen on train only."""
    pick, pick_d = None, -1.0
    for a in feats:
        v = columns(train, [a])[:, 0]
        d = abs(score_auc(v, train.y) - 0.5)
        if d > pick_d:
            pick, pick_d = a, d
    tr = columns(train, [pick])[:, 0]
    sign = 1.0 if score_auc(tr, train.y) >= 0.5 else -1.0
    te = columns(test, [pick])[:, 0]
    return {"axis": pick, "train_auc": score_auc(tr, train.y),
            "auc": score_auc(sign * te, test.y)}


def frequency_table(train, test, axis=NULL_AXIS, bins=10):
    """M1, no training: deciles of one cheap descriptor on the training side."""
    tr = columns(train, [axis])[:, 0]
    edges = np.quantile(tr, np.linspace(0, 1, bins + 1))[1:-1]
    b_tr = np.digitize(tr, edges)
    rate = np.array([train.y[b_tr == b].mean() if (b_tr == b).any() else 0.5
                     for b in range(bins)])
    b_te = np.digitize(columns(test, [axis])[:, 0], edges)
    return {"axis": axis, "auc": score_auc(rate[b_te], test.y)}


def group_bootstrap(rec, draws=1000, rng=None):
    """Band on the test AUC, resampling the TEST side's groups (never tiles)."""
    rng = rng or np.random.default_rng(PERM_SEED)
    test, s = rec["test"], rec["scores"]
    by_group = collections.defaultdict(list)
    for i, g in enumerate(test.g):
        by_group[g].append(i)
    groups = np.array(sorted(by_group))
    out = []
    for _ in range(draws):
        pick = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        if len(set(test.y[idx])) < 2:
            continue
        out.append(score_auc(s[idx], test.y[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def permute_labels_by_group(pool, rng):
    """NC1/null_label: swap the ARM label of whole groups in the training side."""
    groups = sorted(set(pool.g))
    labels = np.array([pool.y[pool.g == g][0] for g in groups])
    labels = rng.permutation(labels)
    table = dict(zip(groups, labels))
    y = np.array([table[g] for g in pool.g], dtype=float)
    return Pool(pool.X, y, pool.g, pool.r)


def shuffle_columns_within_arm(pool, feats, rng):
    """NC5: break the within-row correspondence, keep each arm's marginals."""
    X = pool.X.copy()
    for a in feats:
        j = AXES.index(a)
        for y in (0.0, 1.0):
            m = pool.y == y
            X[m, j] = rng.permutation(X[m, j])
    return Pool(X, pool.y, pool.g, pool.r)


# ---------------------------------------------------------------- evaluations

def halves(a3):
    """A3's groups split in two by SHA-1: H1 trains, H2 tests. Never swapped."""
    h = np.array([digest_mod(g, 2) for g in a3["g"]])
    return h == 0, h == 1


def works_of(a1):
    return sorted(set(a1["r"]))


def eval_E1(arms, a3key, feats, C=None):
    """In-domain: leave one A1 work out. Says whether anything was learned."""
    a1, a3 = arms[A1], arms[a3key]
    h1, h2 = halves(a3)
    out = []
    for w in works_of(a1):
        train = join(arm_pool(a1, 1, a1["r"] != w), arm_pool(a3, 0, h1))
        test = join(arm_pool(a1, 1, a1["r"] == w), arm_pool(a3, 0, h2))
        rec = train_test(train, test, feats, C)
        rec["fold"] = w
        out.append(rec)
    return out


def eval_E2(arms, a3key, feats, C=None):
    """The main transfer: trained on A1 vs A3, tested on A2 vs A3."""
    a1, a2, a3 = arms[A1], arms[A2], arms[a3key]
    h1, h2 = halves(a3)
    train = join(arm_pool(a1, 1), arm_pool(a3, 0, h1))
    test = join(arm_pool(a2, 1), arm_pool(a3, 0, h2))
    return train_test(train, test, feats, C)


def eval_E3(arms, a3key, feats, C=None):
    """The transfer the other way round: trained on A2 vs A3, tested on A1."""
    a1, a2, a3 = arms[A1], arms[A2], arms[a3key]
    h1, h2 = halves(a3)
    train = join(arm_pool(a2, 1), arm_pool(a3, 0, h1))
    test = join(arm_pool(a1, 1), arm_pool(a3, 0, h2))
    return train_test(train, test, feats, C)


def eval_E4(arms, feats, C=None):
    """Inside the author: A1 vs A2, grouped folds. The size of the process gap."""
    a1, a2 = arms[A1], arms[A2]
    wks = works_of(a1)
    f2 = np.array([digest_mod(g, CV_FOLDS) for g in a2["g"]])
    out = []
    for i, w in enumerate(wks):
        train = join(arm_pool(a1, 1, a1["r"] != w), arm_pool(a2, 0, f2 != i))
        test = join(arm_pool(a1, 1, a1["r"] == w), arm_pool(a2, 0, f2 == i))
        rec = train_test(train, test, feats, C)
        rec["fold"] = w
        out.append(rec)
    return out


def group_cv(pool, feats, C=None):
    """Grouped 5-fold CV on one pool, used by NC2 and NC4."""
    folds = np.array([digest_mod(g, CV_FOLDS) for g in pool.g])
    out = []
    for f in range(CV_FOLDS):
        tr, te = folds != f, folds == f
        if len(set(pool.y[tr])) < 2 or len(set(pool.y[te])) < 2:
            continue
        rec = train_test(pool.take(tr), pool.take(te), feats, C)
        rec["fold"] = f
        out.append(rec)
    return out


def row_breakdown(rec, side=1):
    """E2/E3 per printed row: that row's tiles against the whole other side."""
    test, s = rec["test"], rec["scores"]
    other = test.y != side
    out = {}
    for row in sorted(set(test.r[test.y == side])):
        m = (test.r == row) & (test.y == side)
        idx = np.where(m | other)[0]
        out[row] = {"auc": score_auc(s[idx], test.y[idx]), "n": int(m.sum()),
                    "groups": len(set(test.g[m]))}
    return out


# --------------------------------------------------------------------- nulls

def null_label(arms, a3key, feats, which, draws, rng, C=None):
    """null_label: permute the training side's arm labels at group level.

    C is the one the real data chose for this evaluation (amendment 2026-10-31
    11:5x, written before the main scores were read): re-selecting C inside every
    draw costs 2.8 s per fit-set and would run for hours. Handing the null the
    tuned C makes the null stronger, not weaker.
    """
    a1, a2, a3 = arms[A1], arms[A2], arms[a3key]
    h1, h2 = halves(a3)
    out = []
    for _ in range(draws):
        if which == "E2":
            train = join(arm_pool(a1, 1), arm_pool(a3, 0, h1))
            test = join(arm_pool(a2, 1), arm_pool(a3, 0, h2))
            out.append(train_test(permute_labels_by_group(train, rng),
                                  test, feats, C)["auc"])
        elif which == "E3":
            train = join(arm_pool(a2, 1), arm_pool(a3, 0, h1))
            test = join(arm_pool(a1, 1), arm_pool(a3, 0, h2))
            out.append(train_test(permute_labels_by_group(train, rng),
                                  test, feats, C)["auc"])
        elif which == "E1":
            per = []
            for w in works_of(a1):
                train = join(arm_pool(a1, 1, a1["r"] != w), arm_pool(a3, 0, h1))
                test = join(arm_pool(a1, 1, a1["r"] == w), arm_pool(a3, 0, h2))
                per.append(train_test(permute_labels_by_group(train, rng),
                                      test, feats, C)["auc"])
            out.append(float(np.mean(per)))
        elif which == "E4":
            f2 = np.array([digest_mod(g, CV_FOLDS) for g in a2["g"]])
            per = []
            for i, w in enumerate(works_of(a1)):
                train = join(arm_pool(a1, 1, a1["r"] != w),
                             arm_pool(a2, 0, f2 != i))
                test = join(arm_pool(a1, 1, a1["r"] == w),
                            arm_pool(a2, 0, f2 == i))
                per.append(train_test(permute_labels_by_group(train, rng),
                                      test, feats, C)["auc"])
            out.append(float(np.mean(per)))
    a = np.array(out)
    return {"median": float(np.median(a)), "lo": float(np.percentile(a, 2.5)),
            "hi": float(np.percentile(a, 97.5)), "draws": len(a)}


# ------------------------------------------------------- instrument checks

def instrument_checks(arms, a3key, out):
    a1, a2, a3 = arms[A1], arms[A2], arms[a3key]
    h1, h2 = halves(a3)
    p = out.append

    p("## IC-1 単軸 12本の AUC13(既知の値との一致。合わなければ何も読まない)")
    worst = 0.0
    for ax in AXES:
        j = AXES.index(ax)
        got = auc(a1["X"][:, j], a3["X"][:, j])
        want = KNOWN_AUC13[ax]
        d = abs(got - want)
        worst = max(worst, d)
        p(f"  {ax:26s} 既知 {want:.4f}  今回 {got:.4f}  差 {d:.4f}"
          + ("" if d <= 0.0001 else "   <<< 一致しない"))
    p(f"  最大のずれ {worst:.5f} —— {'一致' if worst <= 0.0001 else '不一致'}")

    p("")
    p("## IC-2 1軸だけ入れたロジスティック回帰 = その軸の素の AUC(単調変換)")
    train = join(arm_pool(a1, 1), arm_pool(a3, 0, h1))
    test = join(arm_pool(a2, 1), arm_pool(a3, 0, h2))
    worst2 = 0.0
    for ax in AXES:
        rec = train_test(train, test, [ax], C=1.0)
        raw = auc(columns(test, [ax])[test.y == 1, 0],
                  columns(test, [ax])[test.y == 0, 0])
        d = min(abs(rec["auc"] - raw), abs(rec["auc"] - (1 - raw)))
        worst2 = max(worst2, d)
        p(f"  {ax:26s} LR {rec['auc']:.4f}  素 {raw:.4f} / 鏡像 {1-raw:.4f}  差 {d:.6f}")
    p(f"  最大のずれ {worst2:.6f} —— {'合う' if worst2 < 1e-6 else '器具が壊れている'}")

    p("")
    p("## IC-3 定数予測器(恒等)")
    const = np.zeros(len(test))
    p(f"  AUC = {score_auc(const, test.y):.4f}(期待 0.5000 ちょうど)")

    p("")
    p("## IC-4 A3 の半分(H1 学習専用 / H2 試し専用)")
    g1, g2 = set(a3["g"][h1]), set(a3["g"][h2])
    p(f"  H1 群 {len(g1)} / タイル {int(h1.sum())}    H2 群 {len(g2)} / タイル {int(h2.sum())}")
    p(f"  群の重なり {len(g1 & g2)}(期待 0)   タイルの重なり 0(排他マスク)")

    p("")
    p("## IC-5 faint_of_drawn_ratio の AUC12(外部から既知 0.9900)")
    j = AXES.index("faint_of_drawn_ratio")
    got = auc(a1["X"][:, j], a2["X"][:, j])
    p(f"  今回 {got:.4f}   既知 {KNOWN_AUC12_FAINT:.4f}   差 {abs(got-KNOWN_AUC12_FAINT):.4f}")
    return worst <= 0.0001 and worst2 < 1e-6


# ------------------------------------------------------------------ report

def fmt(rec):
    return (f"AUC {rec['auc']:.4f}  C={rec['C']}  "
            f"学習 {rec['n_train'][0]}/{rec['n_train'][1]}  "
            f"試し {rec['n_test'][0]}/{rec['n_test'][1]}")


def run_model(arms, a3key, draws, out, bands=True):
    rng = np.random.default_rng(PERM_SEED)
    p = out.append
    a1, a2, a3 = arms[A1], arms[A2], arms[a3key]
    h1, h2 = halves(a3)
    summary = {}

    for name, feats in FEATURE_SETS.items():
        p("")
        p(f"# 特徴集合 {name}({len(feats)}軸: {', '.join(feats)})")
        rec = {}

        e1 = eval_E1(arms, a3key, feats)
        m1 = float(np.mean([r["auc"] for r in e1]))
        p("")
        p(f"## E1 内部(A1 の作品を1つ残す。5つの折)  平均 AUC **{m1:.4f}**")
        for r in e1:
            p(f"  折 {r['fold']:16s} {fmt(r)}")
        rec["E1"] = {"mean": m1, "folds": {r["fold"]: r["auc"] for r in e1}}

        e2 = eval_E2(arms, a3key, feats)
        p("")
        p(f"## E2 移植(A1 対 A3 で学習 → **A2 対 A3**)  AUC **{e2['auc']:.4f}**")
        p(f"  {fmt(e2)}")
        if bands:
            lo, hi = group_bootstrap(e2, rng=rng)
            p(f"  群単位 bootstrap 95% 帯 [{lo:.4f}, {hi:.4f}]")
            rec["E2_band"] = [lo, hi]
        p("  A2 の札ごと(畳まない):")
        for row, v in row_breakdown(e2).items():
            flag = "  (群2個。判定しない)" if v["groups"] <= 2 else ""
            p(f"    {row:14s} AUC {v['auc']:.4f}  タイル {v['n']:4d}  群 {v['groups']:3d}{flag}")
        rec["E2"] = e2["auc"]
        rec["E2_rows"] = {k: v["auc"] for k, v in row_breakdown(e2).items()}
        rec["E2_coef"] = e2["coef"]

        e3 = eval_E3(arms, a3key, feats)
        p("")
        p(f"## E3 逆向きの移植(A2 対 A3 で学習 → **A1 対 A3**)  AUC **{e3['auc']:.4f}**")
        p(f"  {fmt(e3)}")
        if bands:
            lo, hi = group_bootstrap(e3, rng=rng)
            p(f"  群単位 bootstrap 95% 帯 [{lo:.4f}, {hi:.4f}]")
            rec["E3_band"] = [lo, hi]
        p("  A1 の作品ごと:")
        for row, v in row_breakdown(e3).items():
            p(f"    {row:14s} AUC {v['auc']:.4f}  タイル {v['n']:4d}  群 {v['groups']:3d}")
        rec["E3"] = e3["auc"]

        e4 = eval_E4(arms, feats)
        m4 = float(np.mean([r["auc"] for r in e4]))
        p("")
        p(f"## E4 作者の中(A1 対 A2、群分割5折)  平均 AUC **{m4:.4f}**")
        for r in e4:
            p(f"  折 {r['fold']:16s} {fmt(r)}")
        rec["E4"] = m4

        p("")
        p("## 腕の比較(同じ試し側で)")
        train2 = join(arm_pool(a1, 1), arm_pool(a3, 0, h1))
        test2 = join(arm_pool(a2, 1), arm_pool(a3, 0, h2))
        m0 = score_auc(np.zeros(len(test2)), test2.y)
        freq = frequency_table(train2, test2)
        single = best_single_axis(train2, test2, feats)
        p(f"  M0 何もしない(定数)        AUC {m0:.4f}   恒等")
        p(f"  M1 頻度表 {freq['axis']}(学習なし) AUC {freq['auc']:.4f}   帰無値の腕の棒")
        p(f"  M2 単軸の最良 {single['axis']}  AUC {single['auc']:.4f}"
          f"(学習側で選んだ。学習側 {single['train_auc']:.4f})")
        p(f"  M3/M4 {name} のロジスティック回帰    AUC {e2['auc']:.4f}")
        rec["M0"], rec["M1"], rec["M2"] = m0, freq["auc"], single["auc"]
        rec["M2_axis"] = single["axis"]

        p("")
        n_draws = draws if name == "F9" else max(1, draws // 4)
        chosen = {"E1": collections.Counter(r["C"] for r in e1).most_common(1)[0][0],
                  "E2": e2["C"], "E3": e3["C"],
                  "E4": collections.Counter(r["C"] for r in e4).most_common(1)[0][0]}
        p(f"## null_label(学習側の札を群単位で置換。{n_draws}回。**棒より先に出す**)")
        p(f"  C は実データが選んだものを渡す(修正 2026-10-31。帰無値を強くする向き): {chosen}")
        for which in ("E1", "E2", "E3", "E4"):
            n = null_label(arms, a3key, feats, which, n_draws, rng, chosen[which])
            p(f"  {which}  中央 {n['median']:.4f}  帯 [{n['lo']:.4f}, {n['hi']:.4f}]"
              f"  ({n['draws']}回)")
            rec[f"null_{which}"] = n

        p("")
        p("## 係数(標準化した尺度の上。**重要度としては読まない**)")
        for a, v in sorted(e2["coef"].items(), key=lambda kv: -abs(kv[1])):
            p(f"  {a:26s} {v:+.4f}")

        summary[name] = rec

    # Negative controls that do not depend on the feature set loop
    p("")
    p("# Negative controls")
    nc2_pool = Pool(a3["X"], np.where(h1, 1.0, 0.0), a3["g"], a3["r"])
    nc2 = group_cv(nc2_pool, F9)
    p(f"  NC2 A3 の H1 対 H2(F9、群分割5折)  平均 AUC "
      f"**{np.mean([r['auc'] for r in nc2]):.4f}**(期待 0.5 付近)")
    early = {"2018/ako7", "2019/ako7", "2019b/ako8"}
    y2 = np.array([1.0 if r in early else 0.0 for r in a2["r"]])
    nc4 = group_cv(Pool(a2["X"], y2, a2["g"], a2["r"]), F9)
    p(f"  NC4 A2 の札で2分(早 {int(y2.sum())} / 遅 {int((1-y2).sum())}、F9)  平均 AUC "
      f"**{np.mean([r['auc'] for r in nc4]):.4f}**(期待 0.75 以上)")
    sh_rng = np.random.default_rng(PERM_SEED)
    nc5 = []
    for w in works_of(a1):
        train = join(arm_pool(a1, 1, a1["r"] != w), arm_pool(a3, 0, h1))
        test = join(arm_pool(a1, 1, a1["r"] == w), arm_pool(a3, 0, h2))
        nc5.append(train_test(shuffle_columns_within_arm(train, F9, sh_rng),
                             shuffle_columns_within_arm(test, F9, sh_rng),
                             F9)["auc"])
    nc5m = float(np.mean(nc5))
    p(f"  NC5 列ごとに独立シャッフル(F9・E1)  平均 AUC **{nc5m:.4f}** "
      f"対 素の E1 {summary['F9']['E1']['mean']:.4f}  "
      f"落差 {summary['F9']['E1']['mean']-nc5m:+.4f}(期待 0.05 以上の低下)")
    summary["NC2"] = float(np.mean([r["auc"] for r in nc2]))
    summary["NC4"] = float(np.mean([r["auc"] for r in nc4]))
    summary["NC5"] = nc5m
    return summary


def verdict(rec, out):
    """Pass criteria, exactly as registered. The bars are read on F9 only."""
    p = out.append
    f9 = rec["F9"]
    e1, e2, e3, e4 = f9["E1"]["mean"], f9["E2"], f9["E3"], f9["E4"]
    p("")
    p("# 判定(**F9 にだけ当てる**。登録 §Pass criteria)")
    p(f"  前提の棒 P: E1 {e1:.4f} ≥ 0.75 → {'満たす' if e1 >= 0.75 else '**満たさない**'}")
    for which, bar in (("E2", 0.60), ("E3", 0.60)):
        n = f9[f"null_{which}"]
        p(f"  棒の妥当性 {which}: null_label 97.5% {n['hi']:.4f} 対 棒 {bar}"
          + ("  → 帯の中。**帯が足りない**" if n["hi"] >= bar else "  → 棒は帯の外"))
    if e1 < 0.75:
        p("  **判定: この道を閉じる** —— 9軸を同時に入れても内部で 0.75 に届かない")
        return "closed"
    if e2 <= 0.60:
        p(f"  **判定: C1 工程で説明できる**(E2 {e2:.4f} ≤ 0.60)")
        return "C1"
    if e2 >= 0.75 and e3 >= 0.75 and e4 <= 0.70:
        p(f"  **判定: C2 作者由来の候補**(E2 {e2:.4f} / E3 {e3:.4f} / E4 {e4:.4f})")
        return "C2"
    if (e2 >= 0.75) != (e3 >= 0.75):
        side = "E2(A1→A2)" if e2 >= 0.75 else "E3(A2→A1)"
        p(f"  **判定: C3 片側だけ立つ** —— {side} のみ。判定保留")
        return "C3"
    p(f"  **判定: C4 判定保留**(E2 {e2:.4f} は 0.60〜0.75 の間、または E4 {e4:.4f} > 0.70)")
    return "C4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/third_pool_20261031/profile_three_arms.csv")
    ap.add_argument("--model", default="animagine", choices=["animagine", "illustrious"])
    ap.add_argument("--mode", default="main", choices=["smoke", "ic", "main"])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    arms = load(args.csv)
    a3key = f"a3_{args.model}"
    out = [f"# 多変量の移植({args.model})  mode={args.mode}  "
           f"登録 doc/prereg_multivariate_transfer_20261031.md",
           f"# 材料 {args.csv} / 置換の seed {PERM_SEED} / C の格子 {C_GRID}", ""]

    if args.mode == "smoke":
        h1, h2 = halves(arms[a3key])
        w = works_of(arms[A1])[0]
        train = join(arm_pool(arms[A1], 1, arms[A1]["r"] != w), arm_pool(arms[a3key], 0, h1))
        test = join(arm_pool(arms[A1], 1, arms[A1]["r"] == w), arm_pool(arms[a3key], 0, h2))
        t = time.time()
        Xtr, Xte = standardize(columns(train, F9), columns(test, F9))
        wv, b = fit_logreg(Xtr, train.y, 1.0)
        dt = time.time() - t
        out.append(f"1回の当てはめ {dt*1000:.1f} ms(F9、学習 {len(train)} 行)")
        rec = train_test(train, test, F9)
        out.append(f"E1 の1折({w}) {fmt(rec)}")
        out.append(f"定数 AUC {score_auc(np.zeros(len(test)), test.y):.4f}")
        ok = True
    elif args.mode == "ic":
        ok = instrument_checks(arms, a3key, out)
        out.append("")
        out.append(f"IC の総合: {'通過' if ok else '**不通過 —— 主値に進まない**'}")
    else:
        summary = run_model(arms, a3key, args.draws, out)
        v = verdict(summary, out)
        summary["verdict"] = v
        if args.out:
            pathlib.Path(args.out).with_suffix(".json").write_text(
                json.dumps(summary, indent=1, ensure_ascii=False, default=float))
        ok = True

    out.append("")
    out.append(f"# 所要 {time.time()-t0:.1f} 秒")
    text = "\n".join(out)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
