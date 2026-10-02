"""Measure problem 1 directly: did the output move from the rough toward the
GT line art, or did it stay put / just add ink?

Why this exists (user framing, 2026-10-02). Two problems were being conflated:

  1. is the rough moving toward the GT line art at all
  2. is empty paper being filled with meaningless imagery

Solving 2 is pointless if 1 never happens, and `gt_bsds_f1` cannot separate
them -- 2026-10-02 measured its floor at 0.1941, i.e. an image with no relation
to the input at all scores that much simply by being dense. So f1 rewards ink,
and these two failures look identical through it:

  1a the output is effectively the rough, unchanged
  1b the output adds information that suits the metric without being line art

This tool splits the strokes by where they are, which makes 1a and 1b distinct
and separates both from problem 2. With the rough's conditioning map and the GT
aligned on the same tile:

  A  shared   in both the conditioning map and the GT -- free to any copier
  B  GT only  in the GT, absent from the conditioning map -- MUST BE DRAWN
  C  map only in the conditioning map, absent from the GT -- MUST BE DELETED

and then, for an output:

  b_recall      fraction of B covered by output ink  -- the "add" half of 1
  c_survival    fraction of C still covered          -- the "delete" half (lower is better)
  a_recall      fraction of A covered                -- sanity; a copier scores high here
  neither_ink   fraction of output ink further than the tolerance from BOTH
                the map and the GT -- problem 2

The conditioning map itself is the trivial baseline and is scored as an arm:
by construction it has b_recall 0, c_survival 1, neither_ink 0. Anything that
does not beat that on b_recall and c_survival is doing nothing for problem 1,
whatever its f1 says.

IMPORTANT, and the reason `floor` matters here too: b_recall is a proximity
test, not one-to-one matching, so a dense enough output earns it for free. Read
b_recall only together with neither_ink, and against the cs0.00 arm, which is
this metric's floor the same way 0.1941 is f1's.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from tile_region_manifest_480 import edge_map  # noqa: E402

IMAGE_SIZE = 480
TOL = 2.0
COND_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning"
)
GT_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line"
)
SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
COLS = ["a_recall", "b_recall", "c_survival", "neither_ink", "b_px", "c_px", "out_px"]


def load_gray(path, invert=False):
    gray = np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))
    return 255 - gray if invert else gray


def dist_to(edge):
    """Euclidean distance from every pixel to the nearest True in `edge`."""
    if not edge.any():
        return np.full(edge.shape, np.inf)
    return distance_transform_edt(~edge)


def decompose(cond_gray, gt_gray):
    """Split GT and map strokes into shared / GT-only / map-only."""
    cond_edge, gt_edge = edge_map(cond_gray), edge_map(gt_gray)
    d_cond, d_gt = dist_to(cond_edge), dist_to(gt_edge)
    shared = gt_edge & (d_cond <= TOL)     # A
    gt_only = gt_edge & (d_cond > TOL)     # B -- must be drawn
    map_only = cond_edge & (d_gt > TOL)    # C -- must be deleted
    return shared, gt_only, map_only, d_cond, d_gt


def score_tile(out_path, cond_gray, gt_gray, parts, invert_pred=False):
    shared, gt_only, map_only, d_cond, d_gt = parts
    out_edge = edge_map(load_gray(out_path, invert=invert_pred))
    d_out = dist_to(out_edge)
    covered = d_out <= TOL

    def frac(mask):
        return float(covered[mask].mean()) if mask.any() else float("nan")

    if out_edge.any():
        neither = float((out_edge & (d_cond > TOL) & (d_gt > TOL)).sum() / out_edge.sum())
    else:
        neither = float("nan")
    return {
        "a_recall": frac(shared),
        "b_recall": frac(gt_only),
        "c_survival": frac(map_only),
        "neither_ink": neither,
        "b_px": int(gt_only.sum()),
        "c_px": int(map_only.sum()),
        "out_px": int(out_edge.sum()),
    }


def gt_tile_path(tile):
    direct = GT_DIR / tile
    if direct.exists():
        return direct
    for split in ("train", "test"):
        candidate = SHARED / split / "line" / tile
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no GT for {tile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True,
                        help="directories whose immediate subdirectories are arms")
    parser.add_argument("--sample-list", default=str(SHARED / "holdout_lineart_family.txt"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True, help="summary CSV")
    parser.add_argument("--per-tile", default=None)
    args = parser.parse_args()

    tiles = [l.strip() for l in open(args.sample_list) if l.strip()]
    if args.limit:
        tiles = tiles[: args.limit]

    # Decomposition depends only on the tile, so compute it once and reuse it
    # across every arm -- it is the expensive part.
    cache = {}
    for tile in tiles:
        cond = load_gray(COND_DIR / tile, invert=True)
        gt = load_gray(gt_tile_path(tile))
        cache[tile] = (cond, gt, decompose(cond, gt))

    arms = [("condition (lineart_coarse)", None)]
    for root in args.roots:
        root = Path(root)
        for d in sorted(root.iterdir()):
            if d.is_dir() and not d.name.startswith("_") and any(d.glob("*_out.png")):
                arms.append((f"{root.name}/{d.name}", d))

    rows, summary = [], []
    for name, arm_dir in arms:
        scored = []
        for tile in tiles:
            cond, gt, parts = cache[tile]
            if arm_dir is None:
                path, invert = COND_DIR / tile, True   # the trivial baseline
            else:
                path, invert = arm_dir / f"{Path(tile).stem}_out.png", False
                if not path.exists():
                    continue
            row = score_tile(path, cond, gt, parts, invert_pred=invert)
            row.update(arm=name, tile=tile)
            rows.append(row)
            scored.append(row)
        if not scored:
            continue
        means = {c: float(np.nanmean([r[c] for r in scored])) for c in COLS}
        means.update(arm=name, n=len(scored))
        summary.append(means)
        print(f"{name:46s} n={means['n']:3d}  b_recall={means['b_recall']:.4f}  "
              f"c_survival={means['c_survival']:.4f}  neither={means['neither_ink']:.4f}  "
              f"a_recall={means['a_recall']:.4f}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "n"] + COLS)
        w.writeheader()
        w.writerows(summary)
    if args.per_tile:
        with open(args.per_tile, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["arm", "tile"] + COLS)
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
