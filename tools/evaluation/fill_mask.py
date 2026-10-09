"""The solid-fill mask: what Track J is predicting, and how it is scored.

Registered 2026-10-09 in doc/work_log.md before any number was produced.

Why this file exists rather than a call into measure_lineart_profile: that
tool's `fill_ratio` is the share of *ink pixels* sitting more than 4px from a
background pixel -- an ink-normalised scalar, not an area. Track J needs a
*mask* to score IoU against, and it needs the area share as well, because the
two are easy to confuse: ako5's GT `fill_ratio` of 25.6% is about 2.4% of the
page once `ink_ratio` 0.0920 is applied. Both are reported, never one alone.

The mask:

    ink  = gray < 128                       # project-wide ink threshold
    core = ink & (dist_to_background > 4)   # exactly what fill_ratio counts
    fill = (dist_to_core <= 4) & ink        # the solid area, fringe included

`core` is the fill minus its own 4px rim, which is the right set to *count*
and the wrong set to *score*, since a 20px-wide stroke's rim is most of a
small fill. Dilating back within the ink is a morphological opening by a disk
of radius 4: the region that can hold a disk of that radius, which is what a
person means by "blacked in".

Both radii are done with a Euclidean distance transform rather than a square
kernel so that `radius` means the same thing on both legs, and so the
sensitivity rows at radius 3 and 6 are comparable to the radius-4 row.

`fill_ratio_cv2` and `fill_ratio_exact` are deliberately two implementations
of one definition: cv2's maskSize-3 approximation (what the published
inventory used) and scipy's exact EDT. They are expected to agree to ~0.02,
and the gap between them is the measure of how much any fill number depends
on that approximation.
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

IMAGE_SIZE = 480
INK_THRESHOLD = 128  # the project-wide ink/background split
FILL_HALF_WIDTH_PX = 4.0  # ink further than this from paper is solid fill, not stroke
MIN_INK_PX = 20  # below this, measure_lineart_profile reports 0.0; matched here


def load_gray(path, invert=False):
    """Grayscale 480x480. `invert` is for white-line-on-black conditioning maps
    -- forgetting it is negative control NC1, not a hypothetical."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    if gray.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    if invert:
        gray = 255 - gray
    return gray


def ink_of(gray):
    return gray < INK_THRESHOLD


def fill_ratio_cv2(ink, radius=FILL_HALF_WIDTH_PX):
    """measure_lineart_profile.fill_ratio, reimplemented on the same primitive.
    Share of ink in strokes thicker than 2*radius."""
    if ink.sum() < MIN_INK_PX:
        return 0.0
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    return float((dist[ink] > radius).mean())


def fill_ratio_exact(ink, radius=FILL_HALF_WIDTH_PX):
    """The same definition on an exact Euclidean transform -- the independent
    leg of the instrument check."""
    if ink.sum() < MIN_INK_PX:
        return 0.0
    dist = distance_transform_edt(ink)
    return float((dist[ink] > radius).mean())


def fill_mask(ink, radius=FILL_HALF_WIDTH_PX):
    """The solid-fill mask itself: ink opened by a disk of `radius`."""
    if ink.sum() < MIN_INK_PX:
        return np.zeros_like(ink, dtype=bool)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    core = ink & (dist > radius)
    if not core.any():
        return np.zeros_like(ink, dtype=bool)
    # dilate the core back by the same radius, staying inside the ink
    back = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3)
    return (back <= radius) & ink


def fill_mask_of_path(path, invert=False, radius=FILL_HALF_WIDTH_PX):
    return fill_mask(ink_of(load_gray(path, invert=invert)), radius=radius)


def score_masks(pred, gt):
    """IoU and F1 for one tile, plus the pixel counts the micro-average pools.

    `iou` is None when both masks are empty -- that tile carries no evidence
    either way and averaging a 1.0 into it would reward an arm for the tiles
    where there was nothing to find. The counts are still returned, so the
    micro-average sees the tile.
    """
    inter = int(np.logical_and(pred, gt).sum())
    p, g = int(pred.sum()), int(gt.sum())
    union = p + g - inter
    return {
        "inter_px": inter,
        "pred_px": p,
        "gt_px": g,
        "union_px": union,
        "iou": (inter / union) if union > 0 else None,
        "f1": (2.0 * inter / (p + g)) if (p + g) > 0 else None,
        "pred_fill_area_frac": p / float(IMAGE_SIZE * IMAGE_SIZE),
        "gt_fill_area_frac": g / float(IMAGE_SIZE * IMAGE_SIZE),
    }


def micro(rows, key_inter="inter_px", key_union="union_px"):
    """Pixel-pooled IoU over a set of tiles -- the pre-registered primary."""
    inter = sum(r[key_inter] for r in rows)
    union = sum(r[key_union] for r in rows)
    return (inter / union) if union > 0 else None


def micro_f1(rows):
    inter = sum(r["inter_px"] for r in rows)
    denom = sum(r["pred_px"] + r["gt_px"] for r in rows)
    return (2.0 * inter / denom) if denom > 0 else None
