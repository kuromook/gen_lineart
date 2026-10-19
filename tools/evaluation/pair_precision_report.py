"""Track D reopen (2026-10-19): read the precision arm against its null.

`pair_alignment_strata.py` now scores both directions per pair. What counts as
the null moved once before any number was read, and
`doc/track_d_reopen_precision_20261019.md` records both steps:

Section 4 froze a reference value, predicted precision = recall x (L_gt /
L_rough), on the reasoning that the rough skeleton is 1.4-2.5x longer than the
GT skeleton so a low precision is bought by that length alone. In pixel units
that identity reduces exactly to M_prec / M_rec -- the matched length measured
from the rough side over the matched length measured from the GT side, every
length cancelling -- so it does not hold quantity fixed and vary position. It
measures how many rough strokes crowd one GT stroke.

The null that does hold quantity fixed and destroy position is the arm this
instrument already computes: the GT skeleton rotated 180 degrees, and Track
D's own convention `aligned = measured - rotated`, which is 0 by derivation
when position carries nothing. That is the headline here. The frozen reference
stays as a secondary, under its corrected name -- the asymmetry of the
correspondence, not position beyond quantity.

Print order is fixed to avoid circularity: the instrument's distortion first,
then the null, then the ratios.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Frozen 2026-10-19, doc/track_d_reopen_precision_20261019.md section 4.
# L_gt = median gt_len, L_rough = median rough_density x 230400 (the WHOLE
# skeleton, not the min_seg survivors), both from
# results/pair_alignment_strata_20260915/per_pair.csv (commit 6f0bb1b).
FROZEN = {
    #                 L_gt  L_rough  ratio  recall_le3  predicted_precision
    "clippairskoma": (1601, 4063, 0.394, 0.3537, 0.139),
    "ako5ver2koma": (1127, 2677, 0.421, 0.2005, 0.084),
    "fitnesskoma": (981, 1741, 0.563, 0.2007, 0.113),
    "gakuenkoma": (1216, 1709, 0.712, 0.2417, 0.172),
    "hamlabikoma": (1300, 3004, 0.433, 0.2026, 0.088),
    "houseikoma": (864, 1474, 0.586, 0.2450, 0.143),
}
# Section 4.5, declared before the numbers were read. The window applies to
# the secondary only, and names the asymmetry of the correspondence.
NEAR = (0.80, 1.25)
FAR_HIGH, FAR_LOW = 1.5, 0.67
MAJOR = "clippairskoma"      # 82.4% of the pairs: a majority without it is "mixed"


def read_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out = {"name": r["name"], "source": r["source"]}
            for k, v in r.items():
                if k not in ("name", "source"):
                    out[k] = float(v) if v not in ("", "nan") else np.nan
            rows.append(out)
    return rows


def crowding(ratios, label):
    """How far the matched-length asymmetry sits from one-to-one. Not the
    three-branch reading of section 4: that wording assumed the reference was
    a quantity-only null, which it is not."""
    n = len(ratios)
    near = [p for p, v in ratios.items() if NEAR[0] <= v <= NEAR[1]]
    high = [p for p, v in ratios.items() if v >= FAR_HIGH]
    low = [p for p, v in ratios.items() if v <= FAR_LOW]
    print(f"  {label}: min {min(ratios.values()):.2f} max {max(ratios.values()):.2f}; "
          f"in window {len(near)}/{n}, >={FAR_HIGH} {len(high)}/{n}, <={FAR_LOW} {len(low)}/{n}")
    if len(near) == n:
        return "one-to-one in every pool"
    if len(high) >= 4 and MAJOR in high:
        return "more rough length matches than GT length: many rough strokes per GT stroke"
    if len(low) >= 4 and MAJOR in low:
        return "less rough length matches than GT length: GT strokes share few rough strokes"
    if len(high) >= 4 or len(low) >= 4:
        return f"mixed: a majority without {MAJOR}, i.e. the small pools only"
    return "mixed: per pool"


def main():
    new_csv = Path(sys.argv[1] if len(sys.argv) > 1 else "results/pair_alignment_strata_20261019/per_pair.csv")
    old_csv = Path("results/pair_alignment_strata_20260915/per_pair.csv")
    rows = read_rows(new_csv)
    valid = [r for r in rows if r["gt_len"] > 0 and not np.isnan(r["rough_le3"])]
    print(f"{new_csv}: {len(rows)} pairs, {len(valid)} with GT strokes")

    by = defaultdict(list)
    for r in valid:
        by[r["source"]].append(r)
    med = {pool: (lambda rs: lambda c: float(np.median([r[c] for r in rs])))(rs) for pool, rs in by.items()}
    pools = [p for p in FROZEN if p in by]
    for p in FROZEN:
        if p not in by:
            print(f"  MISSING POOL: {p}")

    if old_csv.exists():
        old = {r["name"]: r for r in read_rows(old_csv)}
        both = [(r, old[r["name"]]) for r in valid if r["name"] in old]
        worst = {c: max(abs(a[c] - b[c]) for a, b in both)
                 for c in ("rough_le3", "rough_3to8", "rough_gt8", "rough_chance_le3", "rough_aligned", "gt_len")}
        print(f"\nrecall-side cross-check vs {old_csv.parent.name}: {len(both)}/{len(valid)} pairs matched by name, "
              + "  ".join(f"max|d {c}| {v:.2e}" for c, v in worst.items()))

    print("\n(1) the instrument's own distortion, per pool: k = segment-median share / pixel share.")
    print("    It cancels in the secondary (k_prec / k_rec is what survives) and does not enter the null.")
    print(f"  {'pool':<15} {'n':>5} | {'k_rec':>6} {'k_prec':>6} {'k_p/k_r':>8} | "
          f"{'n_seg_gt':>8} {'n_seg_rgh':>9} {'mean_gt':>7} {'mean_rgh':>8} | "
          f"{'L_gt_seg':>8} {'L_rough_seg':>11} {'L_rough_frozen':>14} {'len_def':>7}")
    for pool in pools:
        m = med[pool]
        k_rec = m("rough_le3") / m("rough_pix_le3")
        k_prec = m("rough_prec_le3") / m("rough_prec_pix_le3")
        l_seg, l_frozen = m("rough_seg_len"), FROZEN[pool][1]
        print(f"  {pool:<15} {len(by[pool]):>5} | {k_rec:>6.3f} {k_prec:>6.3f} {k_prec / k_rec:>8.3f} | "
              f"{m('gt_n_seg'):>8.0f} {m('rough_n_seg'):>9.0f} {m('gt_len') / m('gt_n_seg'):>7.1f} "
              f"{l_seg / m('rough_n_seg'):>8.1f} | "
              f"{m('gt_len'):>8.0f} {l_seg:>11.0f} {l_frozen:>14d} {l_frozen / l_seg:>7.3f}")
    print("    populations: every share and every length below is the min_seg survivors' segment-length sum")
    print("    (the exact cancellation in the secondary holds only when they agree). L_rough_frozen is the")
    print("    whole skeleton, density x 230400, so len_def is the definition gap, printed not corrected.")

    print("    mean_gt / mean_rgh is the registered prediction's mechanism (doc, before the numbers):")
    print("    if the rough's segments are shorter, a short segment meets a median <= 3px by chance more")
    print("    easily, so prec_chance is expected ABOVE the recall side's 0.000-0.012. Read prec_ALIGNED.")

    print("\n(2) MAIN VALUE -- rough, against the null that keeps quantity and destroys position:")
    print("    the GT skeleton rotated 180 degrees. aligned = measured - rotated, 0 by derivation.")
    print(f"  {'pool':<15} {'n':>5} | {'prec<=3':>8} {'prec_chance':>11} {'prec_ALIGNED':>12} | "
          f"{'recall<=3':>9} {'r_chance':>8} {'recall_ALIGNED':>14} | {'p_3to8':>7} {'p_gt8':>7}")
    for pool in pools:
        m = med[pool]
        print(f"  {pool:<15} {len(by[pool]):>5} | {m('rough_prec_le3'):>8.4f} {m('rough_prec_chance_le3'):>11.4f} "
              f"{m('rough_prec_aligned'):>12.4f} | {m('rough_le3'):>9.4f} {m('rough_chance_le3'):>8.4f} "
              f"{m('rough_aligned'):>14.4f} | {m('rough_prec_3to8'):>7.4f} {m('rough_prec_gt8'):>7.4f}")

    print("\n(3) secondary -- the frozen reference under its corrected name: the asymmetry of the")
    print("    correspondence (M_prec / M_rec), NOT position beyond quantity. Window centre 1.")
    print(f"  {'pool':<15} | {'pred prec':>9} {'measured':>9} {'(a) seg':>8} {'(b) pix':>8} | "
          f"{'M_rec':>7} {'M_prec':>7}")
    seg_ratio, pix_ratio = {}, {}
    for pool in pools:
        m = med[pool]
        pred = FROZEN[pool][4]
        got = m("rough_prec_le3")
        seg_ratio[pool] = got / pred
        m_rec = m("rough_pix_le3") * m("gt_len")
        m_prec = m("rough_prec_pix_le3") * m("rough_seg_len")
        # identical to M_prec / M_rec: every length cancels when the populations agree
        pix_ratio[pool] = m("rough_prec_pix_le3") / (m("rough_pix_le3") * m("gt_len") / m("rough_seg_len"))
        print(f"  {pool:<15} | {pred:>9.3f} {got:>9.4f} {seg_ratio[pool]:>8.2f} {pix_ratio[pool]:>8.2f} | "
              f"{m_rec:>7.0f} {m_prec:>7.0f}")
    a = crowding(seg_ratio, "(a) segment-median ratio vs frozen prediction")
    b = crowding(pix_ratio, "(b) pixel units, = M_prec / M_rec exactly")
    print(f"  (a) -> {a}")
    print(f"  (b) -> {b}")

    print("\n(4) the two condition maps (registered as a separate object: no reading attached)")
    for key in ("manga_line", "lineart_coarse"):
        print(f"  {key}")
        for pool in pools:
            m = med[pool]
            print(f"    {pool:<15} recall<=3 {m(f'{key}_le3'):.4f}  recall_algn {m(f'{key}_aligned'):.4f}  "
                  f"prec<=3 {m(f'{key}_prec_le3'):.4f}  prec_chance {m(f'{key}_prec_chance_le3'):.4f}  "
                  f"prec_algn {m(f'{key}_prec_aligned'):.4f}  seg_len {m(f'{key}_seg_len'):.0f}")


if __name__ == "__main__":
    main()
