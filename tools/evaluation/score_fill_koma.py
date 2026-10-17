"""Stage 2 non-learned arms and negative controls, on the koma panel.

Registered 2026-10-10 in doc/work_log.md. These are printed BEFORE any model
arm is read; an arm that falls to a control is not read at all (C4).

Arms
  blank              paint nothing.      IDENTITY: region-F1 0 unless the GT
                     is also empty, pixel IoU 0 unless the GT is empty
  all_black          paint everything.   IDENTITY: pixel IoU = |G|/|P|
  rough_dark         the rough's own dark ink through `fill_mask`. **The arm
                     that matters**: if roughs already mark hair, no model is
                     needed and the honest report says so
  rough_dark@budget  the same, handed the GT's fill pixel count -- placement
                     only, the reading that caught `preproc@budget` in stage 1
  freq_table         lesson 14's required no-training control: 16 bins of
                     normalised y-centre x 4 bins of log area over each
                     `rough_dark` component, fitted on every 10th train panel
                     by sorted name, predicting filled / not

Negative controls, one named error each
  NC1 polarity       the rough read as white-on-black. Expected fill area ~1.0
                     and indistinguishable from all_black
  NC2 offbyone       panel i's prediction against panel i+1's GT
  NC3 hflip          the prediction mirrored left-right. Panels are not
                     square, so a transpose cannot pass silently here, but a
                     flip can -- and manga is read right to left, so flipped
                     copies are the error this material actually invites
  NC5 shift50        the prediction rolled 50px. Sensitivity, no bar
  (NC4 is the context shuffle and belongs to M1; it lives in the trainer)

Every number carries both readings (area_frac and the ink-normalised
fill_ratio) and the near-blank / normal strata stay apart, as C5 requires.

Usage:
  score_fill_koma.py --side held --limit 0 --out held.csv
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_mask, fill_ratio_cv2, ink_of  # noqa: E402
from koma_split import ROOT, load as load_split  # noqa: E402
from region_metrics import MIN_REGION_PX, regions, score_panel  # noqa: E402

OUT = Path("results/fill_koma_20261010")
NEAR_BLANK_INK = 0.002
NC2_MIN_OVERLAP = 0.25   # H1 guard: below this the overlap is not a panel
FREQ_Y_BINS, FREQ_A_BINS = 16, 4
FREQ_EVERY = 10          # every 10th train panel by sorted name


def budget_mask(gray, n_px):
    """The n_px darkest pixels -- the arm picks the places, the GT supplies
    the amount."""
    if n_px <= 0:
        return np.zeros(gray.shape, bool)
    n_px = min(int(n_px), gray.size)
    cut = np.partition(gray.ravel(), n_px - 1)[n_px - 1]
    m = gray <= cut
    if m.sum() > n_px * 1.5:
        return np.zeros(gray.shape, bool)
    return m


def comp_features(mask, shape):
    """(labels, [(id, y_bin, a_bin, area)]) for components above MIN_REGION_PX."""
    lab, ids, stats = regions(mask, MIN_REGION_PX)
    out = []
    h = shape[0]
    for i in ids:
        area = int(stats[i, cv2.CC_STAT_AREA])
        cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2
        yb = min(FREQ_Y_BINS - 1, int(FREQ_Y_BINS * cy / max(h, 1)))
        ab = min(FREQ_A_BINS - 1, int(max(0.0, math.log10(area) - 2.0)))
        out.append((i, yb, ab, area))
    return lab, out


def fit_freq_table(names, root=ROOT, every=FREQ_EVERY, verbose=True):
    """P(component is filled | y bin, area bin), from the train side only."""
    num = np.zeros((FREQ_Y_BINS, FREQ_A_BINS))
    den = np.zeros((FREQ_Y_BINS, FREQ_A_BINS))
    used = 0
    for name in sorted(names)[::every]:
        r = cv2.imread(str(Path(root) / "rough" / name), cv2.IMREAD_GRAYSCALE)
        g = cv2.imread(str(Path(root) / "line" / name), cv2.IMREAD_GRAYSCALE)
        if r is None or g is None or r.shape != g.shape:
            continue
        gt = fill_mask(ink_of(g))
        lab, feats = comp_features(fill_mask(ink_of(r)), r.shape)
        for i, yb, ab, area in feats:
            den[yb, ab] += 1
            num[yb, ab] += float(gt[lab == i].mean() >= 0.5)
        used += 1
    table = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    if verbose:
        print(f"freq_table fitted on {used} train panels "
              f"(every {every}th by sorted name); "
              f"{int(den.sum())} components, overall filled rate "
              f"{num.sum() / max(den.sum(), 1):.3f}")
    return table, den


def arms_for(name, table, root=ROOT):
    r = cv2.imread(str(Path(root) / "rough" / name), cv2.IMREAD_GRAYSCALE)
    g = cv2.imread(str(Path(root) / "line" / name), cv2.IMREAD_GRAYSCALE)
    if r is None or g is None or r.shape != g.shape:
        return None
    gt_ink = ink_of(g)
    gt = fill_mask(gt_ink)
    r_ink = ink_of(r)
    rough_fill = fill_mask(r_ink)
    lab, feats = comp_features(rough_fill, r.shape)
    freq = np.zeros(r.shape, bool)
    for i, yb, ab, _ in feats:
        if table[yb, ab] >= 0.5:
            freq |= (lab == i)
    preds = {
        "blank": np.zeros(r.shape, bool),
        "all_black": fill_mask(np.ones(r.shape, bool)),
        "rough_dark": rough_fill,
        "rough_dark@budget": fill_mask(budget_mask(r, int(gt.sum()))),
        "freq_table": freq,
        "nc1_polarity": fill_mask(ink_of(255 - r)),
    }
    return preds, gt, gt_ink, r_ink


def nc2_census(names):
    """NC2's repair, second attempt. AMENDMENT H1, 2026-10-17.

    The first repair -- "pair with the nearest same-shape panel by sorted
    name" -- was registered on 2026-10-16 and is REFUTED by measurement
    before it was ever read as a control: over the 969 held panels there are
    **920 distinct shapes**, 894 panels are the only one of their shape, and
    only 75 panels (7.7%) sit in a pairable group. It would have produced
    **49 NC2 rows per arm instead of 0** -- an improvement from void to
    useless. Measured, printed here, and recorded in doc/work_log.md rather
    than discovered later.

    So the registration's OTHER option is taken: **crop to the common
    overlap**. Consecutive panels in sorted name order are paired, which is
    the original intent (a neighbouring panel of the same work is the most
    similar wrong answer available), and both are centre-cropped to
    `(min(h), min(w))`.

    Centre, not top-left: a top-left crop would systematically keep one
    corner of every panel, and manga panels carry content throughout.

    The objection that made cropping the second choice -- "an arm's NC2 row
    would no longer be comparable to the arm's own row" -- is answered by
    scoring the arm on the SAME crop as well, emitted as `arm@nc2crop`. The
    comparison is then within one geometry and NC2 is read against that row,
    never against the full-panel row.

    Guard, fixed before the run: if the overlap is less than 25% of the
    smaller panel's area the pair is skipped and counted, because a
    tall-by-wide pair can overlap in a sliver that is not a panel.
    """
    shapes = []
    for n in sorted(names):
        g = cv2.imread(str(Path(ROOT) / "line" / n), cv2.IMREAD_GRAYSCALE)
        shapes.append(None if g is None else g.shape)
    fr, skip = [], 0
    for a, b in zip(shapes, shapes[1:]):
        if a is None or b is None:
            skip += 1
            continue
        ov = min(a[0], b[0]) * min(a[1], b[1])
        f = ov / float(min(a[0] * a[1], b[0] * b[1]))
        fr.append(f)
        if f < NC2_MIN_OVERLAP:
            skip += 1
    fr.sort()
    print(f"NC2 pairing (H1, common-overlap crop): {len(names)} panels, "
          f"{len(set(shapes))} distinct shapes -- the same-shape repair would "
          f"have given only {sum(1 for c in set(shapes) if shapes.count(c) >= 2)} "
          f"groups, so it is not used")
    if fr:
        print(f"  overlap as a share of the smaller panel: min {fr[0]:.3f}  "
              f"q25 {fr[len(fr) // 4]:.3f}  median {fr[len(fr) // 2]:.3f}  "
              f"max {fr[-1]:.3f}")
    print(f"  pairs skipped by the {NC2_MIN_OVERLAP:.0%} overlap guard or a "
          f"missing image: {skip}; expected NC2 rows per arm "
          f"{len(names) - 1 - skip}")


def centre_crop(m, h, w):
    y = (m.shape[0] - h) // 2
    x = (m.shape[1] - w) // 2
    return m[y:y + h, x:x + w]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", default="held", choices=("held", "train"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    split = load_split()
    names = [n for n, _, _ in split[args.side]]
    fam = {n: f for n, _, f in split["train"] + split["held"]}
    if args.limit:
        names = names[:args.limit]
    nc2_census(names)
    table, _ = fit_freq_table([n for n, _, _ in split["train"]])

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out or OUT / f"per_panel_{args.side}.csv")
    rows = []
    pending = None
    first = None
    for k, name in enumerate(names):
        got = arms_for(name, table)
        if got is None:
            continue
        preds, gt, gt_ink, r_ink = got
        meta = {"panel": name, "family": fam.get(name, "?"),
                "gt_ink_ratio": round(float(gt_ink.mean()), 6),
                "near_blank": int(gt_ink.mean() < NEAR_BLANK_INK)}
        if first is None:
            first = (gt, gt_ink)
        for arm, p in preds.items():
            rows.append({**meta, "arm": arm,
                         **score_panel(p, gt, r_ink, gt_ink)})
        # controls derived from the SAME prediction mask, so only the named
        # error differs between an arm and its control
        for arm in ("rough_dark", "rough_dark@budget", "freq_table", "all_black"):
            p = preds[arm]
            for nc, mutated in (("nc3_hflip", p[:, ::-1]),
                                ("nc5_shift50", np.roll(p, (50, 50), (0, 1)))):
                rows.append({**meta, "arm": f"{nc}:{arm}",
                             **score_panel(mutated, gt, r_ink, gt_ink)})
        if pending is not None:
            pmeta, ppreds, pshape = pending
            ch, cw = min(pshape[0], gt.shape[0]), min(pshape[1], gt.shape[1])
            ov = (ch * cw) / float(min(pshape[0] * pshape[1], gt.size))
            if ov >= NC2_MIN_OVERLAP:
                gtc = centre_crop(gt, ch, cw)
                gic = centre_crop(gt_ink, ch, cw)
                ric = centre_crop(r_ink, ch, cw)
                for arm in ("rough_dark", "rough_dark@budget", "freq_table",
                            "all_black"):
                    # the arm on the SAME crop, so NC2 is read against one
                    # geometry and never against the full-panel row (H1)
                    rows.append({**meta, "arm": f"{arm}@nc2crop",
                                 **score_panel(centre_crop(preds[arm], ch, cw),
                                               gtc, ric, gic)})
                    rows.append({**meta, "arm": f"nc2_offbyone:{arm}",
                                 **score_panel(centre_crop(ppreds[arm], ch, cw),
                                               gtc, ric, gic)})
        pending = (meta, {a: preds[a] for a in
                          ("rough_dark", "rough_dark@budget", "freq_table",
                           "all_black")}, gt.shape)
        if (k + 1) % 25 == 0:
            print(f"{k + 1}/{len(names)}", flush=True)

    fields = sorted({k for r in rows for k in r})
    head = ["panel", "family", "arm", "near_blank", "gt_ink_ratio"]
    fields = head + [f for f in fields if f not in head]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} rows for {len(names)} panels: {out_path}")


if __name__ == "__main__":
    main()
