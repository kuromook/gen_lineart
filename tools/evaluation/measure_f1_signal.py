"""Measure an output's f1 *signal*: how much of its score is real correspondence
rather than the score any dense image gets for free.

Ported into the shared tools on 2026-10-03 from Track I
(`lineart-image-prompt/experiments/floor_by_density_20261003.py`), unchanged apart
from this header. Background in `doc/CURRENT.md` lesson 9 and
`inbox/note_floor_is_per_output_not_a_constant_20261003.md`.

    signal = f1(output, its own GT) - mean_j f1(output_j of the same arm, that GT)

where j runs over tiles from other source images. GT is held fixed and only the
prediction is swapped, which is the direction that works; swapping GT instead
buries how hittable that particular GT is and gave a degenerate output a signal of
-0.0645 where it must be 0.

Why a per-output null rather than one constant: the floor is a property of the
output, not of the dataset. The same degenerate construction scores 0.1941 on
holdout_lineart_family, 0.1285 on diag_valid5 and 0.1027 on housei; and density
alone does not predict it either -- two arms at ink_ratio 0.425 and 0.443 floor at
0.148 and 0.204, because ink concentrated into strokes hits a random GT less often
than ink scattered as hatching.

Protocol when measuring a new arm: run the degenerate output (ControlNet residuals
times zero) through the same procedure first, and read nothing else until its
signal comes out near zero.
"""
import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

TRACK = Path(__file__).resolve().parents[1]
FOUNDATION = Path("/home/sh1/deepl/lineart")
COND_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning"
)
GT_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line"
)
SHARED = FOUNDATION / "dataset/pairs_480"
WORKER = TRACK / "evaluation/score_f1_signal_one.py"
PYTHON = FOUNDATION / "venv/bin/python"


def gt_tile_path(tile):
    direct = GT_DIR / tile
    if direct.exists():
        return direct
    for split in ("train", "test"):
        candidate = SHARED / split / "line" / tile
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no GT for {tile}")


def source_of(tile):
    parts = tile.split("_")
    return parts[1] if len(parts) > 1 else tile


def mismatch_map(tiles, stride=97):
    """tile -> a tile from a different source image, as a PERMUTATION.

    A first attempt sent every tile to the first tile of the next source, which
    is what the probe's `gt_otherfam` arm does. That is wrong here: it draws all
    192 floor estimates from only 8 reference images, and those eight happen to
    be slightly blanker than the pool (near_white 0.9305 against 0.9237), which
    biases the floor downward. Stepping by a stride coprime with the list length
    visits every tile exactly once instead, so the reference pool is the whole
    group and its density matches by construction. The walk continues until the
    source image differs, so no content is ever shared.
    """
    out = {}
    n = len(tiles)
    for i, t in enumerate(tiles):
        j = (i + stride) % n
        while source_of(tiles[j]) == source_of(t) and tiles[j] != t:
            j = (j + 1) % n
        out[t] = tiles[j]
    return out


def score_one(img_path, gt_path, cond_path, timeout_s, invert_pred=False):
    cmd = [str(PYTHON), str(WORKER), str(img_path), str(gt_path), str(cond_path)]
    if invert_pred:
        cmd.append("--invert-pred")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True,
                        help="label=path pairs; path is a directory of <stem>_out.png, "
                             "or the literal 'CONDITION' or 'GT'")
    parser.add_argument("--sample-list", default=str(SHARED / "holdout_lineart_family.txt"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-tile", default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    tiles = [l.strip() for l in open(args.sample_list) if l.strip()]
    mism = mismatch_map(tiles)
    scratch = Path(args.out).parent / "_inverted_condition"

    rows, summary = [], []
    for spec in args.arms:
        label, _, path = spec.partition("=")
        jobs = []
        for tile in tiles:
            if path == "CONDITION":
                # profile_metrics reads from disk at GT polarity, so the map is
                # written out inverted once, exactly as the main scorer does.
                scratch.mkdir(parents=True, exist_ok=True)
                dst = scratch / f"{Path(tile).stem}_out.png"
                if not dst.exists():
                    g = np.asarray(Image.open(COND_DIR / tile).convert("L").resize((480, 480)))
                    Image.fromarray(255 - g).save(dst)
                src = dst
            elif path == "GT":
                src = gt_tile_path(tile)
            else:
                src = Path(path) / f"{Path(tile).stem}_out.png"
            if Path(src).exists():
                jobs.append((src, tile))

        if not jobs:
            print(f"[{label}] no outputs, skipped", file=sys.stderr)
            continue

        scored = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {}
            for src, tile in jobs:
                # true: against its own GT. floor: against a different source's GT.
                futs[pool.submit(score_one, src, gt_tile_path(tile), COND_DIR / tile, args.timeout)] = (tile, "true")
                futs[pool.submit(score_one, src, gt_tile_path(mism[tile]), COND_DIR / tile, args.timeout)] = (tile, "floor")
            got = {}
            for fut, (tile, kind) in futs.items():
                r = fut.result()
                if r is not None:
                    got.setdefault(tile, {})[kind] = r
        for tile, d in got.items():
            if "true" not in d or "floor" not in d:
                continue
            row = {"arm": label, "tile": tile,
                   "f1_true": d["true"]["gt_bsds_f1"], "f1_floor": d["floor"]["gt_bsds_f1"],
                   "ink_ratio": d["true"]["ink_ratio"], "near_white_frac": d["true"]["near_white_frac"]}
            row["signal"] = row["f1_true"] - row["f1_floor"]
            rows.append(row)
            scored.append(row)

        m = {k: float(np.mean([r[k] for r in scored])) for k in
             ("f1_true", "f1_floor", "signal", "ink_ratio", "near_white_frac")}
        m.update(arm=label, n=len(scored))
        summary.append(m)
        print(f"{label:34s} n={m['n']:3d} ink={m['ink_ratio']:.4f}  "
              f"f1_true={m['f1_true']:.4f}  floor={m['f1_floor']:.4f}  "
              f"signal={m['signal']:+.4f}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "n", "ink_ratio", "near_white_frac",
                                          "f1_true", "f1_floor", "signal"])
        w.writeheader()
        w.writerows(summary)
    if args.per_tile:
        with open(args.per_tile, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["arm", "tile", "ink_ratio", "near_white_frac",
                                              "f1_true", "f1_floor", "signal"])
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
