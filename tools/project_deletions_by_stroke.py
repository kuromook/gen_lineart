"""Aggregate Track C's per-pixel deletion to whole stroke segments.

The pixel-level version of this material (`project_deletions_to_rough.py`) was
judged for 77 pairs on 2026-10-04 and the judge rejected a whole class of
images on sight: white specks scattered over the drawing, "like falling snow".
Measured, that is strokes broken into dashes -- the classifier decides one
contour pixel at a time, so any stroke it half-decides comes out dotted. It
breaks 38.6% of the rough's stroke segments; the placebo breaks 52.1%, and the
pixel oracle, which is the project's own definition of a correct deletion,
breaks 34.0%. The preference that was being recorded tracked breakage exactly:
the chosen side had fewer broken strokes in 23 of the 24 decided pairs, the
same count as the headline preference. The set was measuring which arm damaged
fewer strokes, not which strokes it chose.

Here the same decision is aggregated to the unit the question is asked in: the
rough's ink is cut into stroke segments (1px skeleton, split at crossing-number
junctions, per the standing rule against the shared skeletonize()), and a
segment is removed whole when the decision erased most of it and kept whole
otherwise. No arm can then dash a stroke, and what separates two arms is only
which strokes went.

The placebos pick whole segments at random, matched to the arm they control for
on both the erased ink and the size distribution of the removed segments.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
sys.path.insert(0, "/home/sh1/deepl/lineart-stroke-selection/scripts")
from tile_region_manifest_480 import edge_map  # noqa: E402
from train_stroke_selection import load_gray01  # noqa: E402

COND = Path("/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning")
CLS = Path("/home/sh1/deepl/lineart-stroke-selection/results/signal_rescore_20261003/outputs/classifier")
TILES = Path("/home/sh1/deepl/lineart/dataset/pairs_480/holdout_lineart_family.txt")
ROOT = Path(__file__).resolve().parents[1] / "results/trackc_judge_20261004"

INK_THRESHOLD = 24
PROPAGATE_PX = 4.0
REMOVE_ABOVE = 0.5      # a segment goes when more than half its ink was erased
ARMS = ["rough", "classifier", "placebo", "oracle", "placebo_oracle"]


def stroke_labels(ink):
    """Cut the ink into stroke segments and give every ink pixel one.

    skimage's 1px thinning plus a crossing number, never the shared
    evaluate_stroke_stability.skeletonize(), which leaves 2px ridges and breaks
    junction detection.
    """
    sk = skeletonize(ink)
    p = np.pad(sk.astype(np.int8), 1)
    nb = [p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:],
          p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2]]
    crossing = sum(np.abs(nb[i] - nb[(i + 1) % 8]) for i in range(8)) // 2
    lab, n = ndimage.label(sk & ~(sk & (crossing >= 3)), structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(ink.shape, dtype=np.int32), 0
    # every ink pixel adopts the nearest segment's label, junctions included
    _, idx = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    full = np.where(ink, lab[idx[0], idx[1]], 0)
    return full, n


def erased_pixels(ink, edge, delete_edge):
    """Which ink the per-pixel decision implicates (the earlier projection)."""
    keep_edge = edge & ~delete_edge
    d_del = ndimage.distance_transform_edt(~delete_edge)
    d_keep = ndimage.distance_transform_edt(~keep_edge)
    return ink & (d_del < d_keep) & (d_del <= PROPAGATE_PX)


def segments_to_drop(labels, n, erased):
    """Majority vote per segment: the whole stroke goes, or none of it does."""
    total = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    hit = np.bincount(labels.ravel(), weights=erased.ravel(), minlength=n + 1)[1:]
    frac = np.divide(hit, total, out=np.zeros_like(hit, dtype=float), where=total > 0)
    return np.nonzero(frac > REMOVE_ABOVE)[0] + 1, total


def random_segments_like(dropped, total, rng):
    """Whole segments at random, matched on count, sizes and so on ink.

    The draw is over every segment, not over the ones the real decision left
    alone: excluding those would make the control the real set's complement
    rather than a random set, and there are not enough segments left to match
    an arm like the oracle, which removes most of them. Overlapping the real
    set by chance is what being random means here.
    """
    available = set(np.nonzero(total > 0)[0] + 1)
    picked = []
    for size in sorted(total[dropped - 1], reverse=True):
        if not available:
            break
        ids = np.fromiter(available, dtype=int)
        gap = np.abs(total[ids - 1] - size)
        near = ids[np.argsort(gap, kind="stable")[:4]]       # break ties at random
        choice = int(rng.choice(near))
        picked.append(choice)
        available.discard(choice)
    # top up: matching segment for segment runs short when the real decision
    # removes most of them and the large ones are used up, so close the
    # remaining ink gap with whatever single segment fits it best.
    target = int(total[dropped - 1].sum())
    while available:
        gap = target - int(total[np.array(picked, dtype=int) - 1].sum()) if picked else target
        if gap <= 0:
            break
        ids = np.fromiter(available, dtype=int)
        best = int(ids[np.argmin(np.abs(total[ids - 1] - gap))])
        if abs(total[best - 1] - gap) >= gap:
            break
        picked.append(best)
        available.discard(best)
    return np.array(picked, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-labels", default=str(ROOT / "holdout_keep_labels"))
    ap.add_argument("--out", default=str(ROOT / "stroke_projected"))
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    tiles = [l.strip() for l in open(TILES) if l.strip()]
    oracle_dir = Path(args.oracle_labels)
    out_root = Path(args.out)
    for a in ARMS:
        (out_root / a).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for tile in tiles:
        stem = Path(tile).stem
        cond_gray = (load_gray01(COND / tile) * 255).astype(np.uint8)
        ink = cond_gray >= INK_THRESHOLD
        cond_edge = edge_map(cond_gray)
        labels, n = stroke_labels(ink)

        keep_cls = np.asarray(Image.open(CLS / f"{stem}_out.png").convert("L")) < 128
        drop_cls, total = segments_to_drop(
            labels, n, erased_pixels(ink, cond_edge, cond_edge & ~keep_cls))

        masks = {"rough": np.zeros_like(ink),
                 "classifier": np.isin(labels, drop_cls) & ink,
                 "placebo": np.isin(labels, random_segments_like(drop_cls, total, rng)) & ink}

        label_path = oracle_dir / f"{stem}.png"
        if label_path.exists():
            packed = np.asarray(Image.open(label_path))
            oe = (packed & 1).astype(bool)
            drop_or, _ = segments_to_drop(
                labels, n, erased_pixels(ink, oe, oe & ~((packed >> 1) & 1).astype(bool)))
            masks["oracle"] = np.isin(labels, drop_or) & ink
            masks["placebo_oracle"] = np.isin(
                labels, random_segments_like(drop_or, total, rng)) & ink

        row = {"tile": tile, "n_segments": n, "ink_px": int(ink.sum()),
               "dropped_segments_classifier": len(drop_cls)}
        for arm, erased in masks.items():
            out = cond_gray.copy()
            out[erased] = 0
            Image.fromarray(255 - out).save(out_root / arm / f"{stem}_out.png")
            row[f"erased_{arm}"] = int(erased.sum())
            row[f"ink_ratio_{arm}"] = round(float((out >= INK_THRESHOLD).mean()), 5)
            row[f"near_white_{arm}"] = round(float((out < 32).mean()), 5)
        rows.append(row)

    with open(ROOT / "stroke_projected_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"tiles {len(rows)}  segments/tile {np.mean([r['n_segments'] for r in rows]):.0f}")
    for arm in ARMS:
        k = f"erased_{arm}"
        if k in rows[0]:
            e = np.array([r[k] for r in rows], dtype=float)
            ik = np.array([r[f"ink_ratio_{arm}"] for r in rows])
            nw = np.array([r[f"near_white_{arm}"] for r in rows])
            print(f"  {arm:15s} erased_ink {e.mean():8.1f}  ink_ratio {ik.mean():.4f}  near_white {nw.mean():.4f}")


if __name__ == "__main__":
    main()
