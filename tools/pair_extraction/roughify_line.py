"""Deterministic clean-line-art -> synthetic-rough degradation, for
generating pseudo-pairs to pretrain/bootstrap ControlNet conditioning
before real paired data is available.

Motivation (doc/work_log.md, `diffusion` branch, 2026-08-08): SDEdit-based
(diffusion model) rough<->line conversion showed unreliable structure
preservation -- content sometimes drifted even at low strength, so using
it to generate pseudo-pairs would not guarantee correspondence. This
module instead applies deterministic image-processing degradation
directly to real clean line art, so correspondence to the source is
guaranteed by construction (same underlying source, not a generative
guess) -- the same bootstrapping idea used to train public
canny/scribble-type ControlNets from millions of real photos via
deterministic edge extraction, applied in reverse (clean line -> rough)
for our case.

Four degradation stages (revised 2026-08-08 after comparing v1's output
against real rough_ref via tools/evaluation/measure_lineart_profile.py):
1. Multi-pass jittered duplication, each pass with its OWN independent,
   small-sigma (local, not whole-image-smooth) elastic distortion applied
   BEFORE accumulation -- v1 applied one shared, large-sigma elastic warp
   AFTER accumulating all passes via whole-image affine transforms, which
   produced near-parallel duplicate copies of the same long curve
   (measured: long_line_ratio 0.74 vs real rough's 0.46, long_component_
   ratio 0.57 vs real's 0.44 -- too continuous/confident). Per-pass local
   distortion makes each copy diverge independently, breaking the
   "parallel duplicate" look into more genuinely varied exploratory
   strokes.
2. Stroke fragmentation: randomly punch small gaps into the accumulated
   ink to break long continuous strokes into shorter segments, directly
   targeting the same long_line_ratio/long_component_ratio gap.
3. Stray mark pass: one additional very-low-opacity, large-jitter copy of
   the ink, to create faint marks that sit further from the final line
   than the main jittered passes do (measured gap: real rough's faint
   pixels sit far from ink on average, faint_mean_dist_to_ink=10.7px, v1's
   sat close at 3.6px -- anti-aliasing-like rather than genuine stray
   construction lines).
4. Graphite tone + noise: remap ink from confident black to mid-gray with
   per-pixel noise, matching real rough scans' faint graphite character.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def elastic_distort(img_u8, alpha, sigma, rng):
    shape = img_u8.shape
    dx = gaussian_filter((rng.random(shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rng.random(shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img_u8, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def fragment_strokes(mask_u8, rng, n_gaps_per_1k_px=6.0, gap_radius_range=(2, 5)):
    """Punch small random gaps into an ink mask to break long continuous
    strokes into shorter segments (real rough sketches are dominated by
    short exploratory marks, not long confident curves)."""
    ink_px = int((mask_u8 > 0).sum())
    if ink_px < 50:
        return mask_u8
    n_gaps = int(ink_px / 1000.0 * n_gaps_per_1k_px)
    out = mask_u8.copy()
    h, w = mask_u8.shape
    ys, xs = np.nonzero(mask_u8 > 0)
    if len(ys) == 0:
        return mask_u8
    idx = rng.integers(0, len(ys), size=n_gaps)
    for i in idx:
        r = int(rng.integers(*gap_radius_range))
        cv2.circle(out, (int(xs[i]), int(ys[i])), r, 0, -1)
    return out


def roughify(line_gray, seed=0, n_passes_range=(3, 6), jitter_px=3, jitter_deg=2.0,
             per_pass_elastic_alpha=5.0, per_pass_elastic_sigma=3.0,
             stray_jitter_px=14, stray_jitter_deg=6.0, stray_opacity_range=(0.15, 0.28),
             fragment_gaps_per_1k_px=2.5, gray_level_range=(60, 110), noise_std=8.0):
    """line_gray: uint8 grayscale clean line-art image (ink dark, background light).
    Returns a uint8 grayscale synthetic-rough image, same shape."""
    rng = np.random.default_rng(seed)
    h, w = line_gray.shape
    ink = (line_gray < 128).astype(np.float32)

    accum = np.zeros((h, w), dtype=np.float32)
    n_passes = rng.integers(*n_passes_range)
    for _ in range(n_passes):
        dx = rng.integers(-jitter_px, jitter_px + 1)
        dy = rng.integers(-jitter_px, jitter_px + 1)
        angle = rng.uniform(-jitter_deg, jitter_deg)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        shifted = cv2.warpAffine(ink, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        # independent local distortion per pass, so copies diverge from
        # each other instead of staying near-parallel
        shifted_u8 = (shifted * 255).astype(np.uint8)
        shifted_u8 = elastic_distort(shifted_u8, per_pass_elastic_alpha, per_pass_elastic_sigma, rng)
        shifted = shifted_u8.astype(np.float32) / 255.0
        accum += shifted * rng.uniform(0.25, 0.5)

    # one additional low-opacity, large-jitter stray pass -- faint marks
    # that sit further from the final converged line than the main passes
    dx = rng.integers(-stray_jitter_px, stray_jitter_px + 1)
    dy = rng.integers(-stray_jitter_px, stray_jitter_px + 1)
    angle = rng.uniform(-stray_jitter_deg, stray_jitter_deg)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    stray = cv2.warpAffine(ink, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    accum += stray * rng.uniform(*stray_opacity_range)
    accum = np.clip(accum, 0.0, 1.0)

    accum_u8 = (accum * 255).astype(np.uint8)
    accum_u8 = fragment_strokes(accum_u8, rng, n_gaps_per_1k_px=fragment_gaps_per_1k_px)
    accum = accum_u8.astype(np.float32) / 255.0

    gray_level = rng.uniform(*gray_level_range)
    result = 255.0 - accum * (255.0 - gray_level)
    noise = rng.normal(0, noise_std, (h, w))
    result = np.clip(result + noise, 0, 255)
    return result.astype(np.uint8)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(args.images):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        rough = roughify(gray, seed=args.seed + i)
        out_path = out_dir / (Path(path).stem + "_roughified.png")
        cv2.imwrite(str(out_path), rough)
        print(f"{path} -> {out_path}")
