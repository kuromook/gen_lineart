"""The third pool's primary value: per-axis AUC for three pairs, as registered.

doc/prereg_third_pool_20261030.md, sections "Primary value", "帰無値" and
"Pass criteria".

Three arms: A1 (the author's analogue-dominant pool), A2 (the author's digital
line art) and A3 (the external SDXL pools). Three pairs per axis:

    AUC13 = A1 vs A3    AUC23 = A2 vs A3    AUC12 = A1 vs A2

Both the value and |AUC - 0.5| are printed, because the first version could
not be judged: its criteria could not handle the orientation.

Two null values, printed before any bar is read:
  null_shuffle -- labels permuted at GROUP level, 1000 draws. Its own band is
                  what says whether a row can be judged at all.
  null_ink     -- ink_ratio's own AUC for that pair and row. ink_ratio is the
                  null arm, so it is never read as a C2 candidate.

Bands and permutations work over GROUPS, never tiles: A1 groups by koma, A2 by
sub-region, A3 by generated image. Tiles from one source are not independent.
"""

import argparse
import collections
import csv
import re
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from compute_auc import AXES, DISCRETE_AXES, NULL_AXIS, auc, tie_structure  # noqa: E402

C1_NEAR, C2_FAR, C12_FLAT = 0.05, 0.15, 0.10   # registered, written before the run
DRAWS = 1000


def group_of(label, sample):
    """The bootstrap unit for each arm (registered: never the tile)."""
    if label == "a1_author_analogue":
        return "_".join(sample.split("_")[:2])            # work + koma index
    if label == "a2_author_digital":
        return sample.split("__")[1]                      # the sub-region
    return sample.split("_w")[0]                          # the generated image


def row_of(label, sample):
    """The stratum printed on its own line (registered: never folded)."""
    if label == "a1_author_analogue":
        return sample.split("_")[0]                       # the work
    if label == "a2_author_digital":
        m = re.match(r"a2_(2\d{3}[a-z]?)_(ako\d+)__", sample)
        return f"{m.group(1)}/{m.group(2)}" if m else "?"
    return label


def load(path):
    arms = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        label, sample = r["label"], r["sample"]
        arms[label][(row_of(label, sample), group_of(label, sample))].append(r)
    return arms


def cells(arm, rows=None):
    """[(group, {axis: values})] for one arm, optionally one row only."""
    out = collections.defaultdict(list)
    for (row, grp), items in arm.items():
        if rows is not None and row not in rows:
            continue
        out[grp].extend(items)
    return out


def axis_values(cell, axis):
    return {g: np.array([float(r[axis]) for r in items]) for g, items in cell.items()}


def flat(by_group):
    return np.concatenate([v for v in by_group.values()]) if by_group else np.array([])


def bootstrap(pos, neg, draws=DRAWS, seed=0):
    rng = np.random.default_rng(seed)
    pk, nk = list(pos), list(neg)
    out = []
    for _ in range(draws):
        p = np.concatenate([pos[pk[i]] for i in rng.integers(0, len(pk), len(pk))])
        n = np.concatenate([neg[nk[i]] for i in rng.integers(0, len(nk), len(nk))])
        out.append(auc(p, n))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def shuffle_null(pos, neg, draws=DRAWS, seed=0):
    """Permute the side labels across groups, keeping each group whole."""
    rng = np.random.default_rng(seed + 1)
    groups = list(pos.values()) + list(neg.values())
    n_pos = len(pos)
    out = []
    for _ in range(draws):
        order = rng.permutation(len(groups))
        p = np.concatenate([groups[i] for i in order[:n_pos]])
        n = np.concatenate([groups[i] for i in order[n_pos:]])
        out.append(auc(p, n))
    return (float(np.median(out)), float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)))


def verdict(d13, d23, d12, judgeable, discrete, axis=None):
    """The registered pass criteria. d* are signed (AUC - 0.5)."""
    if axis == NULL_AXIS:
        return "帰無値の腕(C2 の候補として読まない)"
    if discrete:
        return "判定保留(離散・門の片側性)"
    if not judgeable:
        return "判定保留(帯が足りない)"
    if abs(d12) >= C2_FAR:
        return "C3 作者の中で動く"
    if abs(d13) >= C2_FAR and abs(d23) <= C1_NEAR:
        return "C1 工程で説明できる"
    if abs(d23) >= C2_FAR and d23 * d13 > 0 and abs(d12) <= C12_FLAT:
        return "C2 作者由来の候補"
    return "C4 判定保留"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-csv", required=True)
    ap.add_argument("--external", default="a3_animagine")
    ap.add_argument("--a2-rows", default="", help="comma list; default every row")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args()

    arms = load(args.profile_csv)
    a1, a2, a3 = arms["a1_author_analogue"], arms["a2_author_digital"], arms[args.external]
    a2_rows = sorted({row for row, _ in a2})
    if args.a2_rows:
        a2_rows = args.a2_rows.split(",")

    print(f"# 主値: 3対の AUC  |  外部側の腕 = {args.external}  |  draws = {args.draws}")
    for name, arm in (("A1", a1), ("A2", a2), (args.external, a3)):
        tiles = sum(len(v) for v in arm.values())
        groups = len({g for _, g in arm})
        print(f"  {name:14} タイル={tiles:5d}  群={groups:4d}")
    print()

    print("## IC: プールを自分自身と比べると、全軸ちょうど 0.5 か")
    for name, arm in (("A1", a1), ("A2", a2), (args.external, a3)):
        bad = []
        for axis in AXES:
            v = flat(axis_values(cells(arm), axis))
            if abs(auc(v, v) - 0.5) > 1e-12:
                bad.append(axis)
        print(f"  {name:14} 0.5 から外れた軸: {bad if bad else 'なし'}")
    print()

    print("## 帰無値(棒を読む前に印字する。登録どおり**軸ごと**に回す)")
    scopes = [("全体", None)] + [(r, [r]) for r in a2_rows]
    c3_cell = cells(a3)
    null23 = {}
    print(f"  {'scope':14} {'axis':26} {'群':>4} {'med':>8} {'帯 lo':>8} {'帯 hi':>8}  判定可能")
    for label, rows in scopes:
        c2 = cells(a2, rows)
        for axis in AXES:
            med, lo, hi = shuffle_null(axis_values(c2, axis),
                                       axis_values(c3_cell, axis), args.draws)
            judgeable = (hi - 0.5) < C2_FAR and (0.5 - lo) < C2_FAR
            null23[(label, axis)] = judgeable
            print(f"  {label:14} {axis:26} {len(c2):4d} {med:8.4f} {lo:8.4f} {hi:8.4f}  "
                  f"{'はい' if judgeable else '**いいえ(帯 > 0.15)**'}")
    print()
    print("  同じ置換を A1 側の2対にも当てる(全体のみ)")
    c1_cell, c2_cell = cells(a1), cells(a2)
    for name, x, y in (("13 (A1:A3)", c1_cell, c3_cell), ("12 (A1:A2)", c1_cell, c2_cell)):
        worst = max(
            (shuffle_null(axis_values(x, a), axis_values(y, a), args.draws), a)
            for a in AXES
        )
        (med, lo, hi), axis = worst
        print(f"    {name}: 帯がいちばん広い軸 = {axis} med={med:.4f} 帯=[{lo:.4f},{hi:.4f}]")
    print()

    print("## 軸ごと(全体)")
    hdr = (f"{'axis':26} {'AUC13':>8} {'AUC23':>8} {'AUC12':>8} | {'|d13|':>6} "
           f"{'|d23|':>6} {'|d12|':>6} | {'null_ink13':>10} {'null_ink23':>10} | 仕分け")
    print(hdr)
    c1, c2all, c3 = c1_cell, c2_cell, c3_cell
    ink13 = auc(flat(axis_values(c1, "ink_ratio")), flat(axis_values(c3, "ink_ratio")))
    ink23 = auc(flat(axis_values(c2all, "ink_ratio")), flat(axis_values(c3, "ink_ratio")))
    lines = []
    for axis in AXES:
        p1 = axis_values(c1, axis)
        p2 = axis_values(c2all, axis)
        p3 = axis_values(c3, axis)
        a13 = auc(flat(p1), flat(p3))
        a23 = auc(flat(p2), flat(p3))
        a12 = auc(flat(p1), flat(p2))
        d13, d23, d12 = a13 - 0.5, a23 - 0.5, a12 - 0.5
        v = verdict(d13, d23, d12, null23[("全体", axis)], axis in DISCRETE_AXES, axis)
        print(f"{axis:26} {a13:8.4f} {a23:8.4f} {a12:8.4f} | {abs(d13):6.3f} "
              f"{abs(d23):6.3f} {abs(d12):6.3f} | {ink13:10.4f} {ink23:10.4f} | {v}")
        lines.append((axis, a13, a23, a12, v))
        if axis in DISCRETE_AXES:
            for nm, vals in (("A1", flat(p1)), ("A2", flat(p2)), ("A3", flat(p3))):
                k, big, frac = tie_structure([round(x, 12) for x in vals])
                print(f"{'':26}   {nm}: 相異なる値={k:3d} 最大の同値塊={big:5d}({100*frac:4.1f}%)")
    print()

    print("## 帯(bootstrap、群単位。全体のみ)")
    for axis in AXES:
        p1, p2, p3 = (axis_values(c, axis) for c in (c1, c2all, c3))
        b13, b23, b12 = (bootstrap(x, y, args.draws) for x, y in ((p1, p3), (p2, p3), (p1, p2)))
        print(f"{axis:26} 13=[{b13[0]:.4f},{b13[1]:.4f}] 23=[{b23[0]:.4f},{b23[1]:.4f}] "
              f"12=[{b12[0]:.4f},{b12[1]:.4f}]")
    print()

    print("## NC1 A2 を札で2分する(期待: 0.5 付近。離れたらその軸は作者の中の時期を読んでいる)")
    big = sorted(a2_rows, key=lambda r: -sum(len(v) for v in cells(a2, [r]).values()))[:2]
    x, y = (cells(a2, [r]) for r in big)
    for axis in AXES:
        a = auc(flat(axis_values(x, axis)), flat(axis_values(y, axis)))
        flag = "  ← 0.15 超" if abs(a - 0.5) >= C2_FAR else ""
        print(f"  {axis:26} {big[0]} 対 {big[1]}: AUC={a:.4f} |d|={abs(a-0.5):.3f}{flag}")
    print()

    print("## NC4 A1 の作品間 AUC(作者の中の作品差を、工程差 |d12| と同じ尺度に置く)")
    works = sorted({row for row, _ in a1})
    for axis in AXES:
        ds = []
        for i in range(len(works)):
            for j in range(i + 1, len(works)):
                wi, wj = cells(a1, [works[i]]), cells(a1, [works[j]])
                ds.append(abs(auc(flat(axis_values(wi, axis)),
                                  flat(axis_values(wj, axis))) - 0.5))
        print(f"  {axis:26} 作品間 |d| 中央={np.median(ds):.3f} 最大={max(ds):.3f}")
    print(f"  (作品 {len(works)} 個 = {works})")
    print()

    print("## A2 の行ごと(畳まない。AUC23 と AUC12)")
    for row in a2_rows:
        cr = cells(a2, [row])
        n_judgeable = sum(1 for a in AXES if null23[(row, a)])
        print(f"  {row:14} タイル={sum(len(v) for v in cr.values()):4d} 群={len(cr):3d} "
              f"| 帯が足りる軸 {n_judgeable}/{len(AXES)}")
        for axis in AXES:
            pr = axis_values(cr, axis)
            a23 = auc(flat(pr), flat(axis_values(c3, axis)))
            a12 = auc(flat(axis_values(c1, axis)), flat(pr))
            mark = "" if null23[(row, axis)] else "  ← 判定できない(帯)"
            print(f"    {axis:24} AUC23={a23:7.4f} AUC12={a12:7.4f}{mark}")


if __name__ == "__main__":
    main()
