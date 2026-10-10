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
            pmeta, ppreds = pending
            for arm in ("rough_dark", "rough_dark@budget", "freq_table", "all_black"):
                if ppreds[arm].shape != gt.shape:
                    continue           # panels differ in size; skip, do not pad
                rows.append({**meta, "arm": f"nc2_offbyone:{arm}",
                             **score_panel(ppreds[arm], gt, r_ink, gt_ink)})
        pending = (meta, {a: preds[a] for a in
                          ("rough_dark", "rough_dark@budget", "freq_table", "all_black")})
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
