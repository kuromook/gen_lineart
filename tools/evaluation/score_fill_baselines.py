"""Track J stage 1: score every arm on the solid-fill mask, stratified.

Pre-registered in doc/work_log.md 2026-10-09 (with the binarization amendment
added the same day, before the full run). Writes one row per (tile, arm) so
the strata can be cut afterwards without rescoring.

Arms
  blank          paint nothing -- the trivial baseline that must appear beside
                 every number (initial_notice forbids reporting without it)
  all_black      paint everything -- the over-fill side, and instrument check
                 IC2, whose IoU is analytically |G|/|P|
  preproc@128    LineartAnimeDetector, project ink threshold
  preproc@32     the same, the "naive >32" cut
  preproc@otsu   the same, per-tile Otsu
  preproc@budget the same, handed the GT's own ink count and asked only where
                 to put it -- the most generous reading, and the one that
                 actually tests "an edge detector cannot fill"
  msgan          combined_koma_lucy_mild_msgan_20260729 (aux chain, 2ch)

Negative controls, one named confusion per row (Known Tool Traps 2026-10-09):
  nc1_polarity   the preprocessor read without inverting -- the documented
                 trap that put 99.8% of one track's tiles above ink 0.95
  nc2_offbyone   this tile's prediction against the NEXT tile's GT
  nc3_transpose  the prediction transposed (tiles are square; it passes
                 silently)
  nc4_shift50    the prediction rolled 50px -- sensitivity, no pass bar

Usage:
  score_fill_baselines.py --list housei_test --split test
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import (FILL_HALF_WIDTH_PX, IMAGE_SIZE, fill_mask, fill_ratio_cv2,  # noqa: E402
                       ink_of, load_gray, score_masks)

SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
R = Path("results/baseline_fill_20261009")
NEAR_BLANK_INK = 0.002   # inventory_pair_pools' own blank-tile cutoff
HAS_FILL_AREA = 0.001    # ~230 px; below this a tile carries no fill to find
FULL_FILL_AREA = 0.90    # tile lies inside a fill -- no boundary in the window
SENS_RADII = (3.0, 6.0)
SENS_ARMS = ("blank", "all_black", "preproc@32", "msgan")


def otsu_ink(gray):
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray > t  # gray here is the raw white-on-black map: ink is bright


def budget_ink(gray, n_px):
    """The n_px brightest pixels of the conditioning map -- the detector picks
    the places, the GT supplies the amount."""
    if n_px <= 0:
        return np.zeros(gray.shape, dtype=bool)
    n_px = min(int(n_px), gray.size)
    cut = np.partition(gray.ravel(), gray.size - n_px)[gray.size - n_px]
    mask = gray >= cut
    if mask.sum() > n_px * 1.5:  # a flat histogram would hand back the page
        return np.zeros(gray.shape, dtype=bool)
    return mask


def arm_masks(tile, split, gt_ink, radius=FILL_HALF_WIDTH_PX):
    """Every arm's ink mask for one tile. Returns {arm: ink_mask}."""
    stem = tile[:-4] if tile.endswith(".jpg") else tile
    pp_raw = load_gray(R / "preproc" / split / f"{stem}.jpg")  # white-on-black
    ms = load_gray(R / "msgan" / split / f"{stem}_out.png")    # ink-on-white
    full = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    return {
        "blank": np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool),
        "all_black": full,
        "preproc@128": ink_of(255 - pp_raw),
        "preproc@32": pp_raw > 32,
        "preproc@otsu": otsu_ink(pp_raw),
        "preproc@budget": budget_ink(pp_raw, int(gt_ink.sum())),
        "msgan": ink_of(ms),
        "nc1_polarity": ink_of(pp_raw),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", required=True, help="basename under results/.../lists")
    ap.add_argument("--split", required=True, choices=("train", "test"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tiles = [l.strip() for l in open(R / "lists" / f"{args.list}.txt") if l.strip()]
    if args.limit:
        tiles = tiles[:args.limit]
    out_path = Path(args.out or R / f"per_tile_{args.list}.csv")

    rows = []
    prev_gt_fill = None   # for nc2: this tile's prediction vs the NEXT tile's GT
    pending = None        # (tile, {arm: pred_fill}) waiting for the next GT
    first_gt_fill = None

    for i, tile in enumerate(tiles):
        stem = tile[:-4] if tile.endswith(".jpg") else tile
        gt_gray = load_gray(SHARED / args.split / "line" / f"{stem}.jpg")
        gt_ink = ink_of(gt_gray)
        gt_fill = fill_mask(gt_ink)
        gt_meta = {
            "tile": stem,
            "pool": args.list,
            "gt_ink_ratio": round(float(gt_ink.mean()), 6),
            "gt_fill_ratio": round(fill_ratio_cv2(gt_ink), 6),
            "gt_fill_area_frac": round(float(gt_fill.mean()), 6),
            "near_blank": int(gt_ink.mean() < NEAR_BLANK_INK),
            "has_gt_fill": int(gt_fill.mean() > HAS_FILL_AREA),
            "full_fill": int(gt_fill.mean() > FULL_FILL_AREA),
        }
        if first_gt_fill is None:
            first_gt_fill = gt_fill

        inks = arm_masks(tile, args.split, gt_ink)
        preds = {arm: fill_mask(ink) for arm, ink in inks.items()}

        for arm, pred in preds.items():
            s = score_masks(pred, gt_fill)
            rows.append({**gt_meta, "arm": arm, "radius": FILL_HALF_WIDTH_PX,
                         "pred_fill_ratio": round(fill_ratio_cv2(inks[arm]), 6),
                         "pred_ink_ratio": round(float(inks[arm].mean()), 6), **s})

        # nc3 / nc4 on the two arms that are not degenerate by construction
        for arm in ("preproc@32", "msgan"):
            for nc, mutated in (("nc3_transpose", preds[arm].T),
                                ("nc4_shift50", np.roll(preds[arm], (50, 50), (0, 1)))):
                s = score_masks(mutated, gt_fill)
                rows.append({**gt_meta, "arm": f"{nc}:{arm}", "radius": FILL_HALF_WIDTH_PX,
                             "pred_fill_ratio": "", "pred_ink_ratio": "", **s})

        # nc2: the PREVIOUS tile's prediction against this tile's GT
        if pending is not None:
            prev_tile, prev_preds, prev_meta = pending
            for arm in ("preproc@32", "msgan", "all_black"):
                s = score_masks(prev_preds[arm], gt_fill)
                rows.append({**gt_meta, "arm": f"nc2_offbyone:{arm}",
                             "radius": FILL_HALF_WIDTH_PX, "pred_fill_ratio": "",
                             "pred_ink_ratio": "", **s})
        pending = (stem, {a: preds[a] for a in ("preproc@32", "msgan", "all_black")}, gt_meta)

        # radius sensitivity -- the conclusion must not live on radius 4 alone
        for radius in SENS_RADII:
            gt_r = fill_mask(gt_ink, radius=radius)
            for arm in SENS_ARMS:
                s = score_masks(fill_mask(inks[arm], radius=radius), gt_r)
                rows.append({**gt_meta, "arm": arm, "radius": radius,
                             "gt_fill_area_frac": round(float(gt_r.mean()), 6),
                             "pred_fill_ratio": "", "pred_ink_ratio": "", **s})

        if (i + 1) % 100 == 0:
            print(f"[{args.list}] {i + 1}/{len(tiles)}", flush=True)

    # close the nc2 ring so the last tile is not silently dropped
    if pending is not None and first_gt_fill is not None:
        _, prev_preds, prev_meta = pending
        for arm in ("preproc@32", "msgan", "all_black"):
            s = score_masks(prev_preds[arm], first_gt_fill)
            rows.append({**prev_meta, "arm": f"nc2_offbyone:{arm}",
                         "radius": FILL_HALF_WIDTH_PX, "pred_fill_ratio": "",
                         "pred_ink_ratio": "", **s})

    fields = ["tile", "pool", "arm", "radius", "gt_ink_ratio", "gt_fill_ratio",
              "gt_fill_area_frac", "near_blank", "has_gt_fill", "full_fill",
              "pred_fill_ratio", "pred_ink_ratio", "pred_fill_area_frac",
              "inter_px", "pred_px", "gt_px", "union_px", "iou", "f1"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved {len(rows)} rows for {len(tiles)} tiles: {out_path}")


if __name__ == "__main__":
    main()
