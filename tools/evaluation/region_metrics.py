"""Region-wise scoring for stage 2, and its instrument checks.

Registered 2026-10-10 in doc/work_log.md:

    region := 8-connected component of the solid-fill mask (radius 4, the
              stage-1 definition unchanged) with area >= MIN_REGION_PX
    match  := IoU >= MATCH_IOU, assigned greedily one-to-one in descending IoU
    region-F1 := harmonic mean of matched/n_pred and matched/n_gt

    MIN_REGION_PX = 500, MATCH_IOU = 0.5 primary; the sensitivity grid
    {0, 100, 500} x {0.25, 0.5, 0.75} is computed every run and printed, and
    is never used to pick a verdict.

A panel with no GT regions and no predicted regions scores 1.0; with no GT and
some predicted, 0.0. Panels are never pooled: the per-panel value is averaged
per family, with near-blank and normal strata kept apart (criterion C5).

Every number carries both readings, as the foundation's second notice
requires: `area_frac` (fill area / panel area) and `fill_ratio` (the
ink-normalised value, the share of ink whose distance transform exceeds 4).

shortcut: the matcher is greedy, not Hungarian; see doc/shortcuts.md.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import FILL_HALF_WIDTH_PX, fill_mask, fill_ratio_cv2, ink_of  # noqa: E402

MIN_REGION_PX = 500
MATCH_IOU = 0.5
SENS_MIN = (0, 100, 500)
SENS_IOU = (0.25, 0.5, 0.75)


def regions(mask, min_px=MIN_REGION_PX):
    """8-connected components of a binary mask, as (labels, [ids]) keeping
    only those at or above min_px."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]
    return lab, keep, stats


def match(pred, gt, min_px=MIN_REGION_PX, iou_thr=MATCH_IOU):
    """Greedy one-to-one matching in descending IoU. Returns a dict."""
    plab, pids, pstat = regions(pred, min_px)
    glab, gids, gstat = regions(gt, min_px)
    np_, ng = len(pids), len(gids)
    if np_ == 0 and ng == 0:
        return {"n_pred": 0, "n_gt": 0, "matched": 0, "precision": 1.0,
                "recall": 1.0, "region_f1": 1.0}
    cand = []
    if np_ and ng:
        # only pairs that actually overlap are candidates
        both = (plab > 0) & (glab > 0)
        if both.any():
            pairs, counts = np.unique(
                np.stack([plab[both], glab[both]]), axis=1, return_counts=True)
            pset, gset = set(pids), set(gids)
            for (pi, gi), inter in zip(pairs.T, counts):
                pi, gi = int(pi), int(gi)
                if pi not in pset or gi not in gset:
                    continue
                union = (pstat[pi, cv2.CC_STAT_AREA] + gstat[gi, cv2.CC_STAT_AREA]
                         - int(inter))
                iou = int(inter) / union if union else 0.0
                if iou >= iou_thr:
                    cand.append((iou, pi, gi))
    cand.sort(reverse=True)
    used_p, used_g, matched = set(), set(), 0
    for iou, pi, gi in cand:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched += 1
    prec = matched / np_ if np_ else (1.0 if ng == 0 else 0.0)
    rec = matched / ng if ng else (1.0 if np_ == 0 else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n_pred": np_, "n_gt": ng, "matched": matched,
            "precision": prec, "recall": rec, "region_f1": f1}


def pixel_iou(pred, gt):
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    return {"inter_px": inter, "union_px": union,
            "pred_px": int(pred.sum()), "gt_px": int(gt.sum()),
            "iou": inter / union if union else 1.0}


def readings(mask, ink):
    """Both readings, always together (foundation notice 2)."""
    return {"area_frac": round(float(mask.mean()), 6),
            "fill_ratio": round(fill_ratio_cv2(ink), 6) if ink is not None else ""}


def score_panel(pred_fill, gt_fill, pred_ink=None, gt_ink=None):
    row = {}
    row.update({f"gt_{k}": v for k, v in readings(gt_fill, gt_ink).items()})
    row.update({f"pred_{k}": v for k, v in readings(pred_fill, pred_ink).items()})
    row.update(pixel_iou(pred_fill, gt_fill))
    row.update(match(pred_fill, gt_fill))
    for m in SENS_MIN:
        for t in SENS_IOU:
            row[f"rf1_m{m}_t{t}"] = round(
                match(pred_fill, gt_fill, m, t)["region_f1"], 4)
    return row


# --------------------------------------------------------------------------
# instrument checks

def disc(shape, cy, cx, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def ic5(verbose=True):
    """Two discs, one displaced by a known amount, expected matches written
    down BEFORE running.

    r = 40, so a disc is pi r^2 = 5027px, above MIN_REGION_PX = 500.
    For two discs of radius r at centre distance d,
        inter = 2 r^2 acos(d/2r) - (d/2) sqrt(4r^2 - d^2)
    d = 16 -> IoU 0.596 -> matches        -> 2 matched, region-F1 1.0
    d = 30 -> IoU 0.364 -> does not match -> 1 matched, region-F1 0.5
    """
    expected = {16: (2, 1.0, 0.596), 30: (1, 0.5, 0.364)}
    ok = True
    for d, (exp_m, exp_f1, exp_iou) in expected.items():
        shape = (400, 500)
        gt = disc(shape, 150, 120, 40) | disc(shape, 150, 340, 40)
        pr = disc(shape, 150, 120, 40) | disc(shape, 150, 340 + d, 40)
        r = match(pr, gt)
        a = disc(shape, 150, 340, 40)
        b = disc(shape, 150, 340 + d, 40)
        got_iou = (a & b).sum() / (a | b).sum()
        good = (r["matched"] == exp_m and abs(r["region_f1"] - exp_f1) < 1e-9
                and abs(got_iou - exp_iou) < 0.01)
        ok &= good
        if verbose:
            print(f"IC5  displacement {d:2d}px: matched {r['matched']} "
                  f"(expected {exp_m})  region-F1 {r['region_f1']:.3f} "
                  f"(expected {exp_f1})  disc IoU {got_iou:.3f} "
                  f"(analytic {exp_iou})  -> {'PASS' if good else 'FAIL'}")
    return ok


def ic3(paths, verbose=True):
    """GT fed in as the prediction: region-F1 and pixel IoU must both be 1.0,
    through the matcher AND through the stitcher."""
    from koma_windows import cut, stitch
    ok = 0
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        ink = ink_of(g)
        gt = fill_mask(ink)
        wins, coords, orig = cut(gt.astype(np.float32))
        back = stitch(wins, coords, orig, reduce="mean") >= 0.5
        r = score_panel(back, gt, ink, ink)
        good = (abs(r["region_f1"] - 1.0) < 1e-9 and abs(r["iou"] - 1.0) < 1e-9
                and np.array_equal(back, gt))
        ok += good
        if verbose and not good:
            print(f"  IC3 MISMATCH {p.name}: region-F1 {r['region_f1']:.4f} "
                  f"pixel IoU {r['iou']:.4f}")
    print(f"IC3  GT as prediction, through the stitcher: {ok}/{len(paths)} "
          f"panels at region-F1 1.0 and pixel IoU 1.0 "
          f"-> {'PASS' if ok == len(paths) else 'FAIL'}")
    return ok == len(paths)


def _self_check():
    """One assert cell with hand-computed values."""
    shape = (200, 200)
    a = disc(shape, 100, 100, 30)
    r = match(a, a)
    assert r["region_f1"] == 1.0 and r["matched"] == 1, r
    # a region under MIN_REGION_PX is invisible to the matcher: r=10 is 317px
    small = disc(shape, 50, 50, 10)
    assert small.sum() < MIN_REGION_PX, small.sum()
    assert match(a | small, a)["n_pred"] == 1, "small region must be dropped"
    assert match(a | small, a, min_px=100)["n_pred"] == 2, "min_px must bite"
    # empty against empty is 1.0; empty GT with a prediction is 0.0
    z = np.zeros(shape, bool)
    assert match(z, z)["region_f1"] == 1.0
    assert match(a, z)["region_f1"] == 0.0
    assert match(z, a)["region_f1"] == 0.0
    return True


_self_check()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="panel dir, for IC3")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()
    print("self-check: PASS (matcher assert cell)")
    ok = ic5()
    if args.root:
        paths = sorted(p for p in (Path(args.root) / "line").iterdir()
                       if p.suffix.lower() in (".png", ".jpg"))[:args.limit]
        ok &= ic3(paths)
    sys.exit(0 if ok else 1)
