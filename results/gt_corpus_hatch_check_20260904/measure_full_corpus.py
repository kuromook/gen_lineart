"""User's question (2026-09-04): montage output shows ~10-30% gray fills but
never denser blacks, and cross-hatch/screentone is a legitimate manga
technique -- how much of that is *already present in the GT training data*
(data/line, 8467 tiles) rather than being a pure model hallucination? The
2026-09-02 hypothesis-1 check compared two data *pools* against each other
(koma_ref vs clip_pairs) on relative structural deviation; it never asked
what the absolute distribution of ink coverage looks like across the full
GT corpus. This computes that directly.

Two views on ink density:
- per-tile ink_ratio (measure_lineart_profile.py's whole-480x480-tile
  fraction of pixels < 128): does any real GT tile go beyond thin-line
  levels (roughly a few percent) into heavy fill/hatch territory?
- per-cell ink density on the same 8x8 grid grid_ink_cv already uses (60x60
  px cells): a whole-tile average would dilute a locally dense hatched
  region against a mostly-blank rest-of-tile. This is a closer analogue to
  the "local gray-level patch" the user is looking at in the montage.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/evaluation")
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from measure_lineart_profile import load_gray, THRESHOLD, DEEP_BLACK_THRESHOLD, GRID_CELLS

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
LINE_DIR = TRACK / "data/line"
OUT_DIR = TRACK / "results/gt_corpus_hatch_check_20260904"


def cell_ink_ratios(ink, cells=GRID_CELLS):
    h, w = ink.shape
    cell_h, cell_w = h // cells, w // cells
    ratios = []
    for r in range(cells):
        for c in range(cells):
            block = ink[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            ratios.append(float(block.sum()) / float(block.size))
    return ratios


def main():
    paths = sorted(p for p in LINE_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    print(f"{len(paths)} tiles", file=sys.stderr)

    tile_ink_ratios = []
    tile_deep_black_ratios = []
    all_cell_ratios = []
    per_tile_rows = []

    for i, p in enumerate(paths):
        gray = load_gray(p)
        ink = gray < THRESHOLD
        ink_ratio = float(ink.sum()) / gray.size
        deep_black_ratio = float((gray < DEEP_BLACK_THRESHOLD).sum()) / gray.size
        cells = cell_ink_ratios(ink)
        tile_ink_ratios.append(ink_ratio)
        tile_deep_black_ratios.append(deep_black_ratio)
        all_cell_ratios.extend(cells)
        per_tile_rows.append((p.name, ink_ratio, deep_black_ratio, max(cells)))
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(paths)}", file=sys.stderr)

    tile_ink_ratios = np.array(tile_ink_ratios)
    tile_deep_black_ratios = np.array(tile_deep_black_ratios)
    all_cell_ratios = np.array(all_cell_ratios)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "per_tile.csv", "w") as f:
        f.write("name,ink_ratio,deep_black_ratio,max_cell_ink_ratio\n")
        for row in per_tile_rows:
            f.write(f"{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f}\n")

    def pct(arr, qs):
        return {q: float(np.percentile(arr, q)) for q in qs}

    qs = [50, 75, 90, 95, 99, 99.9, 100]

    print("\n=== per-tile ink_ratio (whole 480x480 tile, fraction of pixels < 128) ===")
    print(f"mean={tile_ink_ratios.mean():.4f} median={np.median(tile_ink_ratios):.4f}")
    for q, v in pct(tile_ink_ratios, qs).items():
        print(f"  p{q}: {v:.4f}")
    print(f"  count with ink_ratio > 0.30: {(tile_ink_ratios > 0.30).sum()} / {len(tile_ink_ratios)} ({100*(tile_ink_ratios > 0.30).mean():.2f}%)")
    print(f"  count with ink_ratio > 0.50: {(tile_ink_ratios > 0.50).sum()} / {len(tile_ink_ratios)} ({100*(tile_ink_ratios > 0.50).mean():.2f}%)")

    print("\n=== per-tile deep_black_ratio (fraction of pixels < 30) ===")
    print(f"mean={tile_deep_black_ratios.mean():.4f} median={np.median(tile_deep_black_ratios):.4f}")
    for q, v in pct(tile_deep_black_ratios, qs).items():
        print(f"  p{q}: {v:.4f}")

    print(f"\n=== local 60x60 cell ink density (8x8 grid, {len(all_cell_ratios)} cells total from {len(paths)} tiles) ===")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(all_cell_ratios, bins=bins)
    for lo, hi, c in zip(bins[:-1], bins[1:], hist):
        print(f"  [{lo:.1f},{hi:.1f}): {c:8d}  ({100*c/len(all_cell_ratios):.3f}%)")
    print(f"  cells > 0.5: {(all_cell_ratios > 0.5).sum()} ({100*(all_cell_ratios > 0.5).mean():.4f}%)")
    print(f"  cells > 0.7: {(all_cell_ratios > 0.7).sum()} ({100*(all_cell_ratios > 0.7).mean():.4f}%)")
    print(f"  max cell ink ratio in entire corpus: {all_cell_ratios.max():.4f}")

    # tiles with the single darkest cell, for a visual spot check
    per_tile_rows.sort(key=lambda r: -r[3])
    print("\n=== top 15 tiles by max_cell_ink_ratio (candidates for visual spot check) ===")
    for row in per_tile_rows[:15]:
        print(f"  {row[0]}  max_cell_ink_ratio={row[3]:.4f}  tile_ink_ratio={row[1]:.4f}")

    print(f"\nsaved per-tile CSV: {OUT_DIR / 'per_tile.csv'}")


if __name__ == "__main__":
    main()
