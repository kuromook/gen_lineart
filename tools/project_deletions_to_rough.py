"""Render Track C's deletion decision back onto the rough, so a human can see it.

Track C's classifier is a keep-mask over `edge_map(conditioning)` -- the Canny
contour map -- and its raw output is that mask drawn as black pixels. A contour
map is not a drawing: every stroke appears as its own two-sided outline, so
neither the before nor the after is something a person can judge as line art.
That was exactly this track's 2026-09-17 stopping condition.

The decision itself is still interpretable if it is projected back onto the
conditioning: each ink pixel of the rough adopts the keep/delete label of its
nearest contour pixel, and the deleted ink is erased. The result is the same
drawing with some strokes gone -- which is the question this track was reopened
to answer.

Arms written (all identical in rendering, differing only in which ink is gone):
  rough      -- the conditioning untouched
  classifier -- rough minus the ink the classifier deleted
  placebo    -- rough minus the SAME number of ink pixels in deletions of the
                SAME fragment-size distribution, relocated at random along the
                contour map. Ink-matched and granularity-matched, so a judge who
                prefers `classifier` over it is responding to *which* strokes
                went, not to how many.
  oracle     -- rough minus the ink the pixel oracle deletes (built from the
                holdout keep labels), when those labels are present.
"""
import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
sys.path.insert(0, "/home/sh1/deepl/lineart-stroke-selection/scripts")
from tile_region_manifest_480 import edge_map  # noqa: E402
from train_stroke_selection import load_gray01  # noqa: E402

COND = Path("/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning")
CLS = Path("/home/sh1/deepl/lineart-stroke-selection/results/signal_rescore_20261003/outputs/classifier")
TILES = Path("/home/sh1/deepl/lineart/dataset/pairs_480/holdout_lineart_family.txt")
ROOT = Path(__file__).resolve().parents[1] / "results/trackc_judge_20261004"

INK_THRESHOLD = 24      # 0-255 on the inverted rough; below this is paper
PROPAGATE_PX = 4.0      # an ink pixel further than this from any contour keeps
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def project(ink, cond_edge, delete_edge):
    """Label ink pixels by their nearest contour pixel's decision."""
    keep_edge = cond_edge & ~delete_edge
    d_del = ndimage.distance_transform_edt(~delete_edge)
    d_keep = ndimage.distance_transform_edt(~keep_edge)
    return ink & (d_del < d_keep) & (d_del <= PROPAGATE_PX)


def relocate(delete_edge, cond_edge, rng, scale=1.0):
    """Same fragment sizes, same contour map, random positions."""
    lab, n = ndimage.label(delete_edge, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(delete_edge)
    sizes = sorted(np.bincount(lab.ravel())[1:], reverse=True)
    sizes = [max(1, int(round(s * scale))) for s in sizes]
    free = cond_edge.copy()
    out = np.zeros_like(delete_edge)
    h, w = cond_edge.shape
    coords = np.argwhere(cond_edge)
    rng.shuffle(coords)
    cursor = 0
    for size in sizes:
        while cursor < len(coords) and not free[coords[cursor][0], coords[cursor][1]]:
            cursor += 1
        if cursor >= len(coords):
            break
        seed = (int(coords[cursor][0]), int(coords[cursor][1]))
        grown, q = [], deque([seed])
        free[seed] = False
        while q and len(grown) < size:
            y, x = q.popleft()
            grown.append((y, x))
            for dy, dx in NEIGHBOURS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and free[ny, nx]:
                    free[ny, nx] = False
                    q.append((ny, nx))
        for y, x in grown:
            out[y, x] = True
    return out


def matched_placebo(ink, edge, delete_edge, rng, target, iters=8):
    """A relocated deletion carrying the same *erased ink* as the real one.

    Matching on deleted contour pixels is not enough: relocated fragments land
    on differently dense parts of the drawing, and it is the visible ink a judge
    would otherwise read as a cue rather than the choice of strokes.
    """
    lo, hi, best, best_err = 0.2, 2.0, None, None
    for _ in range(iters):
        mid = (lo + hi) / 2
        cand = project(ink, edge, relocate(delete_edge, edge, rng, mid))
        err = abs(int(cand.sum()) - target)
        if best is None or err < best_err:
            best, best_err = cand, err
        if int(cand.sum()) > target:
            hi = mid
        else:
            lo = mid
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-labels", default=str(ROOT / "holdout_keep_labels"))
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    tiles = [l.strip() for l in open(TILES) if l.strip()]
    oracle_dir = Path(args.oracle_labels)
    arms = ["rough", "classifier", "placebo", "oracle", "placebo_oracle"]
    dirs = {a: ROOT / "rough_projected" / a for a in arms}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for tile in tiles:
        stem = Path(tile).stem
        cond01 = load_gray01(COND / tile)
        # the conditioning is white-on-black: the pixel value IS the ink strength
        cond_gray = (cond01 * 255).astype(np.uint8)
        ink = cond_gray >= INK_THRESHOLD
        cond_edge = edge_map(cond_gray)

        keep_cls = np.asarray(Image.open(CLS / f"{stem}_out.png").convert("L")) < 128
        del_cls = cond_edge & ~keep_cls

        erased_cls = project(ink, cond_edge, del_cls)
        masks = {"rough": np.zeros_like(ink), "classifier": erased_cls}
        masks["placebo"] = matched_placebo(ink, cond_edge, del_cls, rng, int(erased_cls.sum()))

        label_path = oracle_dir / f"{stem}.png"
        if label_path.exists():
            packed = np.asarray(Image.open(label_path))
            oe = (packed & 1).astype(bool)
            del_or = oe & ~((packed >> 1) & 1).astype(bool)
            erased_or = project(ink, oe, del_or)
            masks["oracle"] = erased_or
            # the instrument check needs its own ink-matched control: the oracle
            # erases more than twice what the classifier does, so pairing it
            # against `placebo` would let paper tone decide the one comparison
            # that everything else is gated on.
            masks["placebo_oracle"] = matched_placebo(ink, oe, del_or, rng, int(erased_or.sum()))

        row = {"tile": tile, "ink_px": int(ink.sum()),
               "edge_px": int(cond_edge.sum()), "del_edge_cls": int(del_cls.sum())}
        for arm, erased in masks.items():
            out = cond_gray.copy()
            out[erased] = 0                       # erased ink becomes paper
            Image.fromarray(255 - out).save(dirs[arm] / f"{stem}_out.png")
            row[f"erased_{arm}"] = int(erased.sum())
            row[f"ink_ratio_{arm}"] = round(float((out >= INK_THRESHOLD).mean()), 5)
            row[f"near_white_{arm}"] = round(float((out < 32).mean()), 5)
        rows.append(row)

    with open(ROOT / "rough_projected_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"tiles {len(rows)}")
    for arm in arms:
        k = f"erased_{arm}"
        if k in rows[0]:
            e = np.array([r[k] for r in rows], dtype=float)
            ik = np.array([r[f"ink_ratio_{arm}"] for r in rows])
            nw = np.array([r[f"near_white_{arm}"] for r in rows])
            print(f"  {arm:11s} erased_ink mean {e.mean():8.1f}  ink_ratio {ik.mean():.4f}  near_white {nw.mean():.4f}")


if __name__ == "__main__":
    main()
