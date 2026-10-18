"""IC7 follow-up: why is `npred/ngt` ABOVE 1 at every radius?

The IC7 registration (doc/work_log.md, commit 324b2a4) wrote:

    "Splitting cannot occur under dilation (it is monotone in the mask), so
     `n_pred > n_gt` would be a bug, not an effect"

The run measured 1.081 at d = 1 rising to 1.777 at d = 8.485. Monotonicity is
not in doubt, so the registered sentence is wrong about something else: it
forgot the 500px filter. `regions()` counts only components >= MIN_REGION_PX,
and a GT blob BELOW that threshold grows ABOVE it when dilated, appearing as a
predicted region with no counted GT region behind it. IC6's census put 2.0% of
GT fill area in sub-500px components.

This decomposes n_pred instead of assuming which effect it is:

    carriers  pred components containing >= 1 counted GT region
    merges    pred components containing >= 2 counted GT regions (the effect
              C5 predicted and the run did not establish)
    crossers  pred components containing NO counted GT region (the filter
              effect the registration missed)

Mechanical spread: every 20th held panel, the rule IC-j10's exact-EDT leg
used. No verdict is attached to the numbers; they identify a mechanism.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_mask as fm  # noqa: E402
import region_metrics as rm  # noqa: E402
from check_region_f1_overpaint import dilate_mask  # noqa: E402
from koma_split import ROOT, load as load_split  # noqa: E402

OUT = Path("results/fill_koma_20261010/ic7_npred_decomposition.txt")
RADII = (1.0, 4.0, 8.485)
NEAR_BLANK_INK = 0.002
STEP = 20


def decompose(pred, gt):
    """(n_gt, n_pred, carriers, merges, crossers) -- all on the 500px filter."""
    glab, gids, _ = rm.regions(gt, rm.MIN_REGION_PX)
    plab, pids, _ = rm.regions(pred, rm.MIN_REGION_PX)
    carriers = merges = crossers = 0
    for i in pids:
        hit = np.unique(glab[plab == i])
        hit = [h for h in hit if h in gids]
        if len(hit) == 0:
            crossers += 1
        else:
            carriers += 1
            if len(hit) >= 2:
                merges += 1
    return len(gids), len(pids), carriers, merges, crossers


def _self_check():
    """Hand-computed: one counted square, one sub-threshold square that the
    filter admits once dilated; and two squares that dilation merges."""
    g = np.zeros((200, 300), bool)
    g[20:60, 20:60] = True          # 1600px, counted
    g[20:40, 150:170] = True        # 400px, BELOW the 500px filter
    n_gt, n_pred, car, mer, cro = decompose(dilate_mask(g, 2.0), g)
    # 20x20 dilated by 2 is about 24x24 = 576px, above the filter
    assert (n_gt, n_pred, car, mer, cro) == (1, 2, 1, 0, 1), \
        (n_gt, n_pred, car, mer, cro)
    h = np.zeros((200, 300), bool)
    h[20:60, 20:60] = True
    h[20:60, 64:104] = True         # 4px apart, both counted
    n_gt, n_pred, car, mer, cro = decompose(dilate_mask(h, 3.0), h)
    assert (n_gt, n_pred, car, mer, cro) == (2, 1, 1, 1, 0), \
        (n_gt, n_pred, car, mer, cro)
    return True


_self_check()


def main():
    print("self-check: PASS (hand-computed: a 400px blob dilated by 2 crosses "
          "the 500px filter as a crosser; two squares 4px apart dilated by 3 "
          "become one carrier holding 2 GT regions)")
    names = [n for n, _, _ in load_split()["held"]][::STEP]
    acc = {d: np.zeros(5, dtype=np.int64) for d in RADII}
    used = 0
    for idx, name in enumerate(names):
        g = cv2.imread(str(Path(ROOT) / "line" / name), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        ink = fm.ink_of(g)
        if ink.mean() < NEAR_BLANK_INK:
            continue
        gt = fm.fill_mask(ink)
        if len(rm.regions(gt, rm.MIN_REGION_PX)[1]) == 0:
            continue
        used += 1
        for d in RADII:
            acc[d] += np.array(decompose(dilate_mask(gt, d), gt))
        if used % 10 == 0:
            print(f"  {used} panels", flush=True)
    print(f"\npanels: {used} (every {STEP}th held panel, the IC-j10 spread)")
    print(f"{'d':>8}{'n_gt':>9}{'n_pred':>9}{'carriers':>10}{'merges':>8}"
          f"{'crossers':>10}{'crossers/npred':>16}{'merges/ngt':>12}")
    for d in RADII:
        n_gt, n_pred, car, mer, cro = acc[d]
        print(f"{d:>8.3f}{n_gt:>9}{n_pred:>9}{car:>10}{mer:>8}{cro:>10}"
              f"{cro / max(n_pred, 1):>16.3f}{mer / max(n_gt, 1):>12.3f}")
    print("\ncarriers + crossers = n_pred by construction; merges is a subset "
          "of carriers. No verdict is attached here.")


if __name__ == "__main__":
    main()
