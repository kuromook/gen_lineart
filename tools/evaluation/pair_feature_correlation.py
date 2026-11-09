#!/usr/bin/env python
"""Correlation tables from pair_feature_tiles.py output (Track D reopen, 2026-11-08).

Registration: doc/track_d_correlation_register_20261108.md (section 6 fixes the
null and the bar before any number). Reads tiles.csv / segs.csv / pages.csv.

U1 (page). Each page's value of an axis = median over its kept tiles. Sides:
line (128) against sketch160 (main) and sketch200 (beside).
  * within-work Spearman: every axis is ranked inside its work (work_id, which
    carries the year label) and centred, then one correlation is taken over all
    pages. Work-to-work differences cannot enter. Printed for all pages, per
    year label, and per aligned stratum.
  * null: the sketch-line pairing is permuted inside each work x year cell
    (work_id already fixes the year), N_PERM times, the same statistic
    recomputed. Works with one page cannot be permuted: they are dropped and
    counted.
  * bar: is the observed r outside the null's 2.5/97.5 band. The number of
    cells outside is printed beside the 5% expected by chance.
  * the cross-work (raw) Spearman is printed in its own labelled column; it
    mixes work differences in and is not the main number.
  * CCA on the within-work ranks, first canonical correlations, beside their
    permutation distribution.
U2 (segment). Length-weighted shares of sketch segments at <=3 / 3-8 / >8 px
from the GT skeleton, real vs GT rotated 180 deg, per year label and aligned
stratum; and per class the median segment features, real and rotated.
Exposure: ink_ratio per side is printed per year label (not a variable).
"""
import argparse, csv, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SIDES_SK = ("sketch160", "sketch200")
DROP_AXES = {"ink_ratio"}  # exposure only (register section 5)
N_PERM = 1000


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def rank_within(x, groups):
    """Average ranks inside each group, centred and scaled to unit variance per group; NaN kept."""
    out = np.full(len(x), np.nan)
    for g in np.unique(groups):
        m = (groups == g) & ~np.isnan(x)
        if m.sum() < 2:
            continue
        v = x[m]
        order = v.argsort(kind="mergesort")
        r = np.empty(len(v)); r[order] = np.arange(len(v), dtype=float)
        for u in np.unique(v):  # ties -> average rank
            t = v == u
            if t.sum() > 1:
                r[t] = r[t].mean()
        r -= r.mean()
        sd = r.std()
        out[np.where(m)[0]] = r / sd if sd > 0 else 0.0
    return out


def corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 5:
        return np.nan
    a, b = a[m], b[m]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def raw_spearman(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 5:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float); rb = np.argsort(np.argsort(b[m])).astype(float)
    return corr(ra, rb)


def perm_index(groups, rng):
    idx = np.arange(len(groups))
    for g in np.unique(groups):
        w = np.where(groups == g)[0]
        idx[w] = rng.permutation(w)
    return idx


def cca_first(X, Y, k=3):
    """First k canonical correlations of complete rows (ridge 1e-6)."""
    m = ~np.isnan(X).any(1) & ~np.isnan(Y).any(1)
    X, Y = X[m] - X[m].mean(0), Y[m] - Y[m].mean(0)
    n = len(X)
    if n < X.shape[1] + Y.shape[1] + 5:
        return np.full(k, np.nan), n

    def isqrt(C):
        w, v = np.linalg.eigh(C + 1e-6 * np.eye(len(C)))
        return v @ np.diag(1 / np.sqrt(np.clip(w, 1e-12, None))) @ v.T
    Cxx, Cyy, Cxy = X.T @ X / n, Y.T @ Y / n, X.T @ Y / n
    s = np.linalg.svd(isqrt(Cxx) @ Cxy @ isqrt(Cyy), compute_uv=False)
    return s[:k], n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/pair_feature_correlation_20261108/measure")
    ap.add_argument("--qc", default="data/clip_pairs_v4/clip_pairs/clip_pairs_qc_v4.csv")
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    d = Path(a.dir)
    rng = np.random.default_rng(a.seed)

    qc = {f"{x['work_id']}|{x['slug']}|{x['page']}": x for x in csv.DictReader(open(a.qc, encoding="utf-8"))}
    pages = list(csv.DictReader(open(d / "pages.csv", encoding="utf-8")))
    print(f"== pages measured: {len(pages)}")
    tot = Counter()
    for p in pages:
        for k in ("edge", "empty", "sketch_only", "line_only", "kept", "shape_mismatch"):
            tot[k] += int(p[k])
    print("grid cells by reason (all pages):", dict(tot))

    tiles = list(csv.DictReader(open(d / "tiles.csv", encoding="utf-8")))
    axes = [k for k in tiles[0] if k not in ("key", "tile", "side", "stroke_status")]
    st = Counter((t["side"], t["stroke_status"]) for t in tiles)
    print("stroke axes status by side (non-ok tiles keep pixel axes, stroke axes NaN):", dict(st))
    ai = Counter(t["side"] for t in tiles if fnum(t["ink_ratio"]) >= 1.0)
    print("all-ink tiles by side (MLP skeleton axes and line_width_p50 NaN; fix of 2026-11-09):", dict(ai))
    # page medians
    agg = defaultdict(lambda: defaultdict(list))
    for t in tiles:
        for k in axes:
            agg[(t["key"], t["side"])][k].append(fnum(t[k]))
    keys = sorted({t["key"] for t in tiles})
    work = np.array([k.split("|")[0] for k in keys])
    year = np.array([re.match(r"\d+_(\d{4}\w?)_", w).group(1) for w in work])
    aligned = np.array([qc[k]["aligned"] for k in keys])
    V = {side: {ax: np.array([np.nanmedian(agg[(k, side)][ax]) if agg[(k, side)][ax] and
                              not np.all(np.isnan(agg[(k, side)][ax])) else np.nan for k in keys])
                for ax in axes} for side in ("line",) + SIDES_SK}
    print(f"pages with kept tiles: {len(keys)}; works {len(set(work))}; year labels {len(set(year))}")

    # exposure
    print("\n== exposure: page-median ink_ratio by year label (not a variable)")
    for y in sorted(set(year)):
        m = year == y
        print(f"  {y:6s} n={m.sum():4d}  line {np.nanmedian(V['line']['ink_ratio'][m]):.3f}  "
              f"sketch160 {np.nanmedian(V['sketch160']['ink_ratio'][m]):.3f}  "
              f"sketch200 {np.nanmedian(V['sketch200']['ink_ratio'][m]):.3f}")

    # permutable pages
    wc = Counter(work)
    single = [w for w, n in wc.items() if n < 2]
    perm_ok = np.array([wc[w] >= 2 for w in work])
    print(f"\n== null: pairing permuted within work (work_id carries the year label), {a.perm} draws")
    print(f"   works with 1 page (cannot permute, dropped from U1): {len(single)} works / "
          f"{sum(wc[w] for w in single)} pages")

    vars_ = [ax for ax in axes if ax not in DROP_AXES]
    out_rows = []
    for sk_side in SIDES_SK:
        for label, sel in [("all", perm_ok)] + [(f"year={y}", perm_ok & (year == y)) for y in sorted(set(year))] + \
                [(f"aligned={v}", perm_ok & (aligned == v)) for v in ("True", "False")]:
            if sel.sum() < 10:
                print(f"\n-- {sk_side} {label}: n={sel.sum()} pages, too few (<10), not computed")
                continue
            g = work[sel]
            R_sk = {ax: rank_within(V[sk_side][ax][sel], g) for ax in vars_}
            R_ln = {ax: rank_within(V["line"][ax][sel], g) for ax in vars_}
            obs = {(x, y): corr(R_sk[x], R_ln[y]) for x in vars_ for y in vars_}
            raw = {(x, y): raw_spearman(V[sk_side][x][sel], V["line"][y][sel]) for x in vars_ for y in vars_}
            null = defaultdict(list)
            for _ in range(a.perm):
                pi = perm_index(g, rng)
                for x in vars_:
                    sx = R_sk[x][pi]
                    for y in vars_:
                        null[(x, y)].append(corr(sx, R_ln[y]))
            outside = cells = 0
            for (x, y), r in obs.items():
                nl = np.array(null[(x, y)]); nl = nl[~np.isnan(nl)]
                lo, hi = (np.percentile(nl, 2.5), np.percentile(nl, 97.5)) if len(nl) else (np.nan, np.nan)
                out = (not np.isnan(r)) and (r < lo or r > hi)
                cells += not np.isnan(r); outside += out
                out_rows.append({"sketch_side": sk_side, "stratum": label, "n_pages": int(sel.sum()),
                                 "n_works": len(set(g)), "sketch_axis": x, "line_axis": y,
                                 "r_within_work": r, "null_lo": lo, "null_hi": hi, "outside": int(out),
                                 "r_raw_cross_work": raw[(x, y)]})
            print(f"\n-- {sk_side} {label}: n={sel.sum()} pages / {len(set(g))} works; "
                  f"cells outside the null band {outside}/{cells} (chance 5% = {0.05 * cells:.1f})")
            # same-axis diagonal printed in full
            print(f"   {'axis':26s} {'r_within':>9s} {'null 2.5/97.5':>17s} {'r_raw(cross-work)':>18s}")
            for ax in vars_:
                r = next(o for o in out_rows[::-1] if o["sketch_side"] == sk_side and o["stratum"] == label
                         and o["sketch_axis"] == ax and o["line_axis"] == ax)
                flag = "*" if r["outside"] else " "
                print(f"   {ax:26s} {r['r_within_work']:9.3f}{flag} [{r['null_lo']:7.3f},{r['null_hi']:7.3f}] "
                      f"{r['r_raw_cross_work']:18.3f}")
            # CCA on within-work ranks (main stratum only)
            if label == "all":
                X = np.stack([R_sk[x] for x in vars_], 1); Y = np.stack([R_ln[y] for y in vars_], 1)
                good = ~np.all(np.isnan(X), 0) & ~np.all(np.isnan(Y), 0)
                cc, ncc = cca_first(X, Y)
                nulls = []
                for _ in range(min(a.perm, 200)):
                    pi = perm_index(g, rng)
                    nulls.append(cca_first(X[pi], Y)[0])
                nulls = np.array(nulls)
                print(f"   CCA (within-work ranks, {len(vars_)} x {len(vars_)} axes, complete rows n={ncc}): "
                      + ", ".join(f"cc{i + 1} {c:.3f} [null 97.5% {np.nanpercentile(nulls[:, i], 97.5):.3f}]"
                                  for i, c in enumerate(cc)) + f"  ({len(nulls)} permutations)")
    with open(d.parent / "u1_correlation_cells.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    print(f"\nall cells -> {d.parent / 'u1_correlation_cells.csv'}")

    # U2
    print("\n== U2: sketch segments by median distance to GT skeleton (length-weighted shares)")
    segs = list(csv.DictReader(open(d / "segs.csv", encoding="utf-8")))
    feats = ["len_p50", "straight_p50", "curv_p50", "width_p50", "gray_p50", "sharp_p50",
             "dir_0", "dir_45", "dir_90", "dir_135"]
    key_year = {k: y for k, y in zip(keys, year)}
    key_al = {k: v for k, v in zip(keys, aligned)}
    for thr in SIDES_SK:
        for label, f in [("all", lambda k: True)] + \
                [(f"aligned={v}", (lambda v: lambda k: key_al.get(k) == v)(v)) for v in ("True", "False")] + \
                [(f"year={y}", (lambda y: lambda k: key_year.get(k) == y)(y)) for y in sorted(set(year))]:
            L = defaultdict(float); F = defaultdict(list)
            for s in segs:
                if s["arm_thr"] != thr or not f(s["key"]):
                    continue
                L[(s["arm"], s["cls"])] += fnum(s["len_sum"])
                if int(s["n"]) > 0:
                    for ft in feats:
                        F[(s["arm"], s["cls"], ft)].append(fnum(s[ft]))
            line = f"  {thr} {label:14s}"
            for arm in ("real", "rot180"):
                t = sum(L[(arm, c)] for c in ("le3", "3to8", "gt8"))
                if t:
                    line += f" | {arm}: " + " ".join(f"{c} {L[(arm, c)] / t:.3f}" for c in ("le3", "3to8", "gt8"))
            print(line)
            if label in ("all", "aligned=True"):
                print(f"    per-class median of tile medians ({label}):")
                for ft in feats:
                    print(f"      {ft:13s} " + "  ".join(
                        f"{arm[:4]}/{c} {np.nanmedian(F[(arm, c, ft)]) if F[(arm, c, ft)] else np.nan:8.3f}"
                        for arm in ("real", "rot180") for c in ("le3", "gt8")))


if __name__ == "__main__":
    main()
