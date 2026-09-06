"""Characterize "line-art-ness" as a multi-axis structural profile, computed
per-image with no GT/pairing required -- for comparing real line-art tiles
against unconditional diffusion samples (or any other single-image source).

Motivation (doc/work_log.md 2026-08-05, diffusion domain-LoRA review): every
past failure mode in this project came from optimizing one axis (F1@2px,
chamfer, edge_f1) while leaving another unchecked, and the output quietly
stopped being "line art" along the unchecked axis -- soft/marbled texture
(never actually binarized), wobbly/fragmented strokes (binarized but not
continuous), or ControlNet hallucination (confident and continuous but not
tied to the input). This tool deliberately reports several independent axes
instead of one scalar, reusing metrics already validated elsewhere in this
project:

- binarization/confidence: intensity histogram split into ink (`< THRESHOLD`,
  the same 128 cutoff used project-wide), faint/ambiguous
  (`THRESHOLD..BACKGROUND_THRESHOLD`), and background (`> BACKGROUND_THRESHOLD`).
  `deep_black_ratio` (`< DEEP_BLACK_THRESHOLD`) is a separate, stricter
  diagnostic of how much ink is near-pure-black vs. merely ink-dark; it is
  NOT used as the locality anchor below (see the note in the first revision
  of this tool for why that was a bug).
- faint locality (added 2026-08-05, revised same day after the first
  koma_ref-vs-domain-LoRA run): the first version anchored "confident"
  pixels at `< 30` and called everything from 30-220 "midtone", which
  produced a misleading number -- a check of real ink pixel values showed
  the *median* ink pixel (already `< 128`, unambiguously ink by this
  project's own convention) sits at gray 73, so most real strokes were
  being counted as "midtone" purely from an uncalibrated threshold, not
  because they're actually ambiguous. Fixed by anchoring locality on the
  project's real ink threshold (128) instead: `faint_near_ink_ratio` /
  `faint_mean_dist_to_ink` measure, for every faint (128-220) pixel, its
  distance to the nearest actual ink pixel via a distance transform (same
  technique as `chamfer()` in tile_region_manifest_480.py, just
  self-referential instead of rough-vs-line). A thin anti-aliasing halo
  hugging real strokes reads as genuine line art even though it is almost
  entirely "faint" by area; faint pixels sitting far from any ink is the
  actual soft/marbled signature this axis is meant to catch.
- stroke continuity: skeletonize + connected-component length distribution
  (long_component_ratio, components_per_1k_ink_px), from
  evaluate_stroke_stability.py.
- line width consistency: distance-transform stroke width p50/p95, from
  tile_region_manifest_480.py::line_width_stats; width_consistency
  (p95/p50) is new here -- large values mean width varies a lot across the
  image (blobby), small values mean uniform clean strokes.
- long straight-segment ratio and edge-orientation entropy, also reused from
  tile_region_manifest_480.py.
- structural (macro) collapse (added 2026-08-06, diffusion domain-LoRA
  rough-isolation review): every axis above describes *stroke*-level
  quality (width, faintness, continuity) and says nothing about whether the
  image has real composition -- a page that is one repeating parallel-hatch
  texture wall-to-wall can score reasonably on stroke axes while being
  content-free. This was found directly: the roughclean_20260805 LoRA
  variant scored closer to the real rough_ref pool on 7/8 stroke metrics
  than the roughfull_e10 variant, yet a human visual read judged
  roughfull_e10 as *more* likely to be attempting recognizable shapes --
  the stroke axes above cannot see that distinction. `grid_ink_cv` /
  `blank_cell_fraction` divide the image into an 8x8 grid and measure how
  unevenly ink is distributed across cells: real sketches have both
  near-blank paper cells and dense subject cells (high variance), while a
  uniform repeating texture fills the canvas evenly (low variance). This
  does not replace visual review -- it is a first cheap probe for the
  specific "did this collapse into flat/uniform texture" failure, not a
  general content-quality or shape-recognition metric.

Typical use: run once against a reference pool of real line-art tiles to get
a baseline distribution per axis, then run against generated/candidate
samples and compare where each axis falls relative to that reference --
looking for any axis that stands out as the one being silently sacrificed,
rather than trusting a single combined score.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pair_extraction"))

import cv2
import numpy as np

from evaluate_stroke_stability import component_stats, skeletonize
from tile_region_manifest_480 import edge_map, line_width_stats, long_line_ratio, orientation_entropy

IMAGE_SIZE = 480
THRESHOLD = 128  # standard ink/background split used across this project
DEEP_BLACK_THRESHOLD = 30  # diagnostic only -- near-pure-black vs. merely ink-dark
BACKGROUND_THRESHOLD = 220  # standard "confident white" cutoff used elsewhere (evaluate_halo_outputs.py)
FAINT_NEAR_PX = 3.0
GRID_CELLS = 8  # 8x8 grid for structural/macro heterogeneity metrics
BLANK_CELL_INK_RATIO = 0.01  # a cell counts as "near-blank paper" below this local ink_ratio
# Paper axis (added 2026-09-06, from the SDXL track's resolution sweep).
# Deliberately NOT reusing BACKGROUND_THRESHOLD (220): these three came from
# experiments/score_resolution_sweep_20260906.py::paper_metrics in the
# lineart-controlnet-sdxl-fidelity track, and keeping the exact cutoffs keeps
# the numbers comparable with that sweep's published table and its GT anchors
# (bg_mode 255, near_white_frac 94.8%, midtone_frac 1.8%).
NEAR_WHITE = 224  # "this pixel is paper"
MIDTONE_LO, MIDTONE_HI = 64, 192  # neither paper nor ink -- grey wash territory

METRIC_KEYS = [
    "ink_ratio",
    "deep_black_ratio",
    "background_ratio",
    "faint_of_drawn_ratio",
    "faint_near_ink_ratio",
    "faint_mean_dist_to_ink",
    "long_component_ratio",
    "components_per_1k_ink_px",
    "line_width_p50",
    "width_consistency",
    "long_line_ratio",
    "orientation_entropy",
    "grid_ink_cv",
    "blank_cell_fraction",
    "bg_mode",
    "near_white_frac",
    "midtone_frac",
]


def load_gray(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    if gray.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return gray


def intensity_profile(gray):
    total = float(gray.size)
    ink = gray < THRESHOLD
    faint = (gray >= THRESHOLD) & (gray <= BACKGROUND_THRESHOLD)
    deep_black = float((gray < DEEP_BLACK_THRESHOLD).sum()) / total
    background = float((gray > BACKGROUND_THRESHOLD).sum()) / total
    drawn_px = int(ink.sum()) + int(faint.sum())
    faint_of_drawn = float(faint.sum()) / max(drawn_px, 1)
    return {
        "deep_black_ratio": deep_black,
        "background_ratio": background,
        "faint_of_drawn_ratio": faint_of_drawn,
    }


def paper_profile(gray):
    """Is there white paper under the ink, or is the whole page a grey wash?

    Added 2026-09-06 from the SDXL track's resolution sweep, which found that
    `gt_bsds_f1` -- and every stroke axis above -- is blind to this. Two
    measured examples, both scoring respectably on f1 while not being line art:
    an output at f1 0.2263 that was a uniform grey field (near_white 3.1%,
    midtone 84.8%), and one at f1 0.2121 that was a nearly blank page (white
    87.8%, but the subject simply was not drawn).

    That second example is why `near_white_frac` must never be read alone: it
    cannot tell "clean white paper" from "nothing was drawn". Always read it
    together with `ink_ratio` and `line_width_p50`, and confirm visually.
    """
    return {
        "bg_mode": float(np.bincount(gray.astype(np.uint8).ravel(), minlength=256).argmax()),
        "near_white_frac": float((gray >= NEAR_WHITE).mean()),
        "midtone_frac": float(((gray > MIDTONE_LO) & (gray < MIDTONE_HI)).mean()),
    }


def faint_locality(gray):
    """Where faint (ambiguous, non-ink) pixels sit relative to the nearest
    actual ink pixel, not just how many of them exist. A thin
    anti-aliasing halo hugging real strokes (small
    `faint_mean_dist_to_ink`, high `faint_near_ink_ratio`) reads as genuine
    line art even though it is almost entirely faint by area; faint pixels
    sitting far from any ink is the actual soft/marbled signature.
    """
    ink = gray < THRESHOLD
    faint = (gray >= THRESHOLD) & (gray <= BACKGROUND_THRESHOLD)
    ink_px = int(ink.sum())
    faint_px = int(faint.sum())

    if faint_px < 20:
        # no meaningful faint band to judge -- vacuously "not diffuse"
        return {"faint_near_ink_ratio": 1.0, "faint_mean_dist_to_ink": 0.0}
    if ink_px < 5:
        # faint pixels exist with no ink to anchor to -- maximally diffuse
        # by construction
        return {"faint_near_ink_ratio": 0.0, "faint_mean_dist_to_ink": float(IMAGE_SIZE)}

    dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3)
    faint_dist = dist[faint]
    return {
        "faint_near_ink_ratio": float((faint_dist <= FAINT_NEAR_PX).mean()),
        "faint_mean_dist_to_ink": float(faint_dist.mean()),
    }


def grid_heterogeneity(ink, cells=GRID_CELLS):
    """How unevenly ink is spread across an 8x8 grid of the page -- the
    cheap first probe for "did this collapse into one flat repeating
    texture" (see module docstring). `grid_ink_cv` is the coefficient of
    variation (std/mean) of per-cell ink_ratio across the grid: real
    sketches mix near-blank paper cells with dense subject cells (high
    CV); a texture that fills the canvas evenly has low CV regardless of
    how "line-like" that texture is stroke-by-stroke. `blank_cell_fraction`
    is the fraction of cells that are near-empty paper.
    """
    h, w = ink.shape
    cell_h, cell_w = h // cells, w // cells
    cell_ratios = []
    for r in range(cells):
        for c in range(cells):
            block = ink[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            cell_ratios.append(float(block.sum()) / float(block.size))
    cell_ratios = np.array(cell_ratios)
    mean_ratio = cell_ratios.mean()
    cv = float(cell_ratios.std() / mean_ratio) if mean_ratio > 1e-9 else 0.0
    blank_fraction = float((cell_ratios < BLANK_CELL_INK_RATIO).mean())
    return {"grid_ink_cv": cv, "blank_cell_fraction": blank_fraction}


def profile_metrics(path, min_length_fraction=0.08):
    gray = load_gray(path)
    ink = gray < THRESHOLD
    ink_px = int(ink.sum())

    result = {"ink_ratio": ink_px / float(IMAGE_SIZE * IMAGE_SIZE)}
    result.update(intensity_profile(gray))
    result.update(paper_profile(gray))
    result.update(faint_locality(gray))
    result.update(grid_heterogeneity(ink))

    if ink_px < 20:
        result.update(
            {
                "long_component_ratio": 0.0,
                "components_per_1k_ink_px": 0.0,
                "component_count": 0,
                "line_width_p50": 0.0,
                "line_width_p95": 0.0,
                "width_consistency": 0.0,
                "long_line_ratio": 0.0,
                "orientation_entropy": 0.0,
            }
        )
        return result

    edges = edge_map(gray)
    skel = skeletonize(ink)
    comp = component_stats(skel)
    width_p50, width_p95 = line_width_stats(ink)
    result.update(
        {
            "long_component_ratio": comp["long_component_ratio"],
            "components_per_1k_ink_px": (comp["component_count"] / max(ink_px, 1)) * 1000.0,
            "component_count": comp["component_count"],
            "line_width_p50": width_p50,
            "line_width_p95": width_p95,
            "width_consistency": width_p95 / max(width_p50, 1e-6),
            "long_line_ratio": long_line_ratio(ink, IMAGE_SIZE, min_length_fraction),
            "orientation_entropy": orientation_entropy(edges),
        }
    )
    return result


def collect_paths(spec):
    """`spec` is `label=path`, where path is a directory (globs *.png/*.jpg/*.jpeg) or a single file."""
    label, _, raw_path = spec.partition("=")
    if not _:
        raise SystemExit(f"--source must be label=path, got: {spec!r}")
    path = Path(raw_path)
    if path.is_dir():
        paths = sorted(
            p for p in path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
    elif path.is_file():
        paths = [path]
    else:
        raise SystemExit(f"not found: {path}")
    return label, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        dest="sources",
        help="label=path (path is a directory of images or a single image); repeatable",
    )
    parser.add_argument("--sample-size", type=int, default=0, help="random-subsample each source to N images (0 = use all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-length-fraction", type=float, default=0.08)
    parser.add_argument("--output-csv", default="results/lineart_profile_metrics.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for spec in args.sources:
        label, paths = collect_paths(spec)
        if args.sample_size and len(paths) > args.sample_size:
            idx = rng.choice(len(paths), size=args.sample_size, replace=False)
            paths = [paths[i] for i in sorted(idx)]
        print(f"[{label}] {len(paths)} images", file=sys.stderr)
        for i, path in enumerate(paths):
            metrics = profile_metrics(path, args.min_length_fraction)
            rows.append({"label": label, "sample": path.name, **metrics})
            if (i + 1) % 200 == 0:
                print(f"[{label}] {i + 1}/{len(paths)}", file=sys.stderr)

    fields = ["label", "sample"] + METRIC_KEYS + ["component_count", "line_width_p95"]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    header = "label".ljust(16) + "n".rjust(6) + "".join(k[:16].rjust(18) for k in METRIC_KEYS)
    print(header)
    for label, label_rows in by_label.items():
        means = {k: np.mean([r[k] for r in label_rows]) for k in METRIC_KEYS}
        line = label.ljust(16) + str(len(label_rows)).rjust(6) + "".join(f"{means[k]:18.4f}" for k in METRIC_KEYS)
        print(line)

    print("\n(median)")
    for label, label_rows in by_label.items():
        medians = {k: np.median([r[k] for r in label_rows]) for k in METRIC_KEYS}
        line = label.ljust(16) + str(len(label_rows)).rjust(6) + "".join(f"{medians[k]:18.4f}" for k in METRIC_KEYS)
        print(line)

    print(f"\nsaved: {args.output_csv}")


if __name__ == "__main__":
    main()
