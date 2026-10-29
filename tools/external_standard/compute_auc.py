"""The primary value: per-axis AUC, as registered.

doc/prereg_external_standard_20261030.md, "Primary value" and "Pass criteria".

Ties count as 0.5 (the Mann-Whitney form), fixed for every axis, both sides
and every arm, because line_width_p50 takes only 12 distinct values with
50.5% of the author pool on one of them -- see section 0-3. That axis and
width_consistency print their tie structure next to the AUC and are never
read as "no author information" when they land in C1.

Bands are bootstrapped over SOURCES, not tiles: 1,489 tiles come from 457
panels, so a tile-level band overstates the precision by about 1.81x.
"""

import argparse
import collections
import csv

import numpy as np

AXES = [
    "ink_ratio",
    "long_component_ratio",
    "components_per_1k_ink_px",
    "line_width_p50",
    "width_consistency",
    "faint_of_drawn_ratio",
    "faint_near_ink_ratio",
    "faint_mean_dist_to_ink",
    "long_line_ratio",
    "orientation_entropy",
    "deep_black_ratio",
    "background_ratio",
]
NULL_AXIS = "ink_ratio"  # arm A3, the null
DISCRETE_AXES = {"line_width_p50", "width_consistency"}


def auc(pos, neg):
    """AUC with ties at 0.5. pos = author side (label 1), neg = external (0)."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    ranks = np.empty(len(both))
    order = np.argsort(both, kind="mergesort")
    sorted_vals = both[order]
    i = 0
    while i < len(both):
        j = i
        while j + 1 < len(both) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def tie_structure(values):
    counts = collections.Counter(values)
    biggest = counts.most_common(1)[0][1] if counts else 0
    return len(counts), biggest, biggest / max(len(values), 1)


def band(pos_by_source, neg_by_source, draws=2000, seed=0):
    """Bootstrap over sources, as registered."""
    rng = np.random.default_rng(seed)
    pos_keys = list(pos_by_source)
    neg_keys = list(neg_by_source)
    out = []
    for _ in range(draws):
        p = np.concatenate([pos_by_source[pos_keys[i]] for i in rng.integers(0, len(pos_keys), len(pos_keys))])
        n = np.concatenate([neg_by_source[neg_keys[i]] for i in rng.integers(0, len(neg_keys), len(neg_keys))])
        out.append(auc(p, n))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def read(path, source_of):
    rows = list(csv.DictReader(open(path)))
    by_axis = {a: np.array([float(r[a]) for r in rows]) for a in AXES if a in rows[0]}
    sources = [source_of(r["sample"]) for r in rows]
    return rows, by_axis, sources


def group(values, sources):
    out = collections.defaultdict(list)
    for v, s in zip(values, sources):
        out[s].append(v)
    return {k: np.array(v) for k, v in out.items()}


def author_source(name):
    return "_".join(name.split("_")[:2])


def external_source(name):
    return name.split("_w")[0]


def ic3(by_axis):
    """A pool against itself is 0.5 exactly on every axis -- every pair ties."""
    bad = []
    for a, v in by_axis.items():
        got = auc(v, v)
        if abs(got - 0.5) > 1e-12:
            bad.append((a, got))
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-csv", required=True)
    parser.add_argument("--external-csv", help="omit to run IC3 only")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()

    _, a_axes, a_src = read(args.author_csv, author_source)

    print("## IC3 -- a pool against itself is 0.5 on every axis")
    bad = ic3(a_axes)
    if bad:
        for a, got in bad:
            print(f"  **FAIL** {a}: {got!r}")
        raise SystemExit("IC3 failed: the split or the aggregation is leaking")
    print(f"  PASS on all {len(a_axes)} axes (|AUC - 0.5| <= 1e-12)")
    print(f"  author tiles: {len(a_src)}  sources: {len(set(a_src))}")

    if not args.external_csv:
        print("\n(no external csv given -- primary value not computed)")
        return

    _, e_axes, e_src = read(args.external_csv, external_source)
    print(f"  external tiles: {len(e_src)}  sources: {len(set(e_src))}")

    null = auc(a_axes[NULL_AXIS], e_axes[NULL_AXIS])
    print(f"\n## null (arm A3, {NULL_AXIS} alone): AUC = {null:.4f}")
    print(f"   C1 at <= {null + 0.05:.4f}   C2 at >= {null + 0.15:.4f}")

    print(f"\n{'axis':28s}{'AUC':>8s}{'band(2.5-97.5)':>22s}{'median A':>12s}{'median B':>12s}  note")
    for a in AXES:
        if a not in a_axes or a not in e_axes:
            continue
        value = auc(a_axes[a], e_axes[a])
        lo, hi = band(group(a_axes[a], a_src), group(e_axes[a], e_src), args.draws)
        note = ""
        if a in DISCRETE_AXES:
            d1, b1, f1 = tie_structure(a_axes[a])
            d2, b2, f2 = tie_structure(e_axes[a])
            note = f"distinct {d1}/{d2}, biggest tie {f1:.1%}/{f2:.1%} -- never read as 'no author info'"
        verdict = "C2" if value >= null + 0.15 else ("C1" if value <= null + 0.05 else "判定保留")
        print(
            f"{a:28s}{value:8.4f}{f'[{lo:.4f}, {hi:.4f}]':>22s}"
            f"{np.median(a_axes[a]):12.4f}{np.median(e_axes[a]):12.4f}  {verdict} {note}"
        )


if __name__ == "__main__":
    main()
