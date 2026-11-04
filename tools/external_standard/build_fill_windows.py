"""Cut the author's fill windows (A1F) and line windows (A1L), as registered:
doc/prereg_fill_windows_20261104.md.

Material: the five A1 works' koma sub-regions at native scale (the same
sources the A1 tiler read), each with its valid mask.

Windows: a non-overlapping 480 grid anchored at (0, 0); a window is a
candidate when its mask support is >= 0.85 (the tiler default) and its raw
ink_ratio is >= 0.011992 (the registered gate, whole window).

Fill share of a window: ink (gray < 128) whose L2 distance (mask 5) to
non-ink is >= 8 px, over all ink -- the IC5 split, unchanged.

Sources are disjoint between the arms:
  F-source = has at least one candidate with fill share >= 0.10;
             contributes those windows (A1F).
  L-source = every other source; contributes its candidates with fill
             share <= 0.002 (A1L).
At most 4 windows per source (the A1 clumping); when there are more, a
seeded draw picks 4. Every candidate not written is counted by reason.

Tiles are written through normalize.save_jpeg (quality 95), then
prepare_normalized.py applies the white point -- the same path as the
external side.
"""

import argparse
import collections
import csv
import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import normalize

ROOT = Path("/home/sh1/deepl/lineart")
WORKS = ("ako5ver2", "fitness", "gakuen", "hamlabi", "housei")
TILE = 480
MIN_SUPPORT = 0.85
FILL_D = 8.0
F_MIN, L_MAX = 0.10, 0.002
CAP = 4
SEED = 20261104


def fill_share(gray):
    ink = (gray < 128).astype(np.uint8)
    n = int(ink.sum())
    if n == 0:
        return 0.0
    d = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    return float(((ink > 0) & (d >= FILL_D)).sum() / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/author_fill_windows_20261104")
    args = ap.parse_args()
    out = Path(args.out)
    for arm in ("f", "l"):
        (out / f"tiles_{arm}").mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    dropped = collections.Counter()
    rows_out = []
    per_work = {}
    for work in WORKS:
        d = glob.glob(str(ROOT / f"dataset/regions_{work}_koma_subregions_*_masked"))[0]
        n_f = n_l = s_f = s_l = 0
        for r in csv.DictReader(open(Path(d) / "manifest.csv")):
            gray = np.asarray(Image.open(r["masked_line_path"]).convert("L"))
            mask = np.asarray(Image.open(ROOT / r["valid_mask_path"]).convert("L")) > 127
            if mask.shape != gray.shape:
                dropped["source_mask_shape_mismatch"] += 1
                continue
            h, w = gray.shape
            if min(h, w) < TILE:
                dropped["source_smaller_than_tile"] += 1
                continue
            cands = []
            for y in range(0, h - TILE + 1, TILE):
                for x in range(0, w - TILE + 1, TILE):
                    if mask[y:y + TILE, x:x + TILE].mean() < MIN_SUPPORT:
                        dropped["window_support_below_0.85"] += 1
                        continue
                    win = gray[y:y + TILE, x:x + TILE]
                    if normalize.ink_ratio(win) < normalize.INK_GATE:
                        dropped["window_below_ink_gate"] += 1
                        continue
                    cands.append((y, x, fill_share(win)))
            if not cands:
                continue
            if any(s >= F_MIN for _, _, s in cands):
                arm = "f"
                keep = [c for c in cands if c[2] >= F_MIN]
                dropped["f_source_window_below_0.10"] += len(cands) - len(keep)
            else:
                arm = "l"
                keep = [c for c in cands if c[2] <= L_MAX]
                dropped["l_source_window_between_0.002_and_0.10"] += len(cands) - len(keep)
            if not keep:
                continue
            if len(keep) > CAP:
                dropped["over_cap_4"] += len(keep) - CAP
                keep = rng.sample(keep, CAP)
            stem = Path(r["masked_line_path"]).stem
            for y, x, s in sorted(keep):
                name = f"{work}_{stem}_{y:04d}_{x:04d}.jpg"
                normalize.save_jpeg(gray[y:y + TILE, x:x + TILE], out / f"tiles_{arm}" / name)
                rows_out.append({"arm": arm, "work": work, "source": stem, "y": y, "x": x,
                                 "fill_share": s, "tile": name})
            if arm == "f":
                n_f += len(keep)
                s_f += 1
            else:
                n_l += len(keep)
                s_l += 1
        per_work[work] = {"A1F": [n_f, s_f], "A1L": [n_l, s_l]}
        print(f"{work}: A1F {n_f} windows / {s_f} sources | A1L {n_l} windows / {s_l} sources", flush=True)
    with open(out / "fill_windows_manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        wr.writeheader()
        wr.writerows(rows_out)
    print("dropped by reason:", dict(dropped))
    json.dump({"per_work": per_work, "dropped": dict(dropped)},
              open(out / "fill_windows_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
