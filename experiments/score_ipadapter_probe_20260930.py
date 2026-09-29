"""Score every arm of the IP-Adapter probe, with the conditioning map's own
score as the baseline row.

Lesson 6: a `gt_bsds_f1` without the conditioning map's own score beside it is
unreadable -- it cannot tell you whether the model added anything. So the
conditioning map is scored here as a pseudo-arm, and it is the number every
other row has to be read against.

Per-tile dispatch to a subprocess with a hard timeout, because
`bipartite_match_f1` has an unbounded worst case (Known Tool Traps).
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

TRACK = Path(__file__).resolve().parents[1]      # this track's worktree
FOUNDATION = Path("/home/sh1/deepl/lineart")     # shared dataset, tools, venv
COND_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning"
)
GT_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line"
)
SHARED = FOUNDATION / "dataset/pairs_480"
WORKER = TRACK / "experiments/score_ipadapter_probe_one_20260930.py"
PYTHON = FOUNDATION / "venv/bin/python"

COLS = ["gt_bsds_f1", "precision", "recall", "vs_condition_f1", "ink_ratio",
        "fill_ratio", "line_width_p50", "near_white_frac", "midtone_frac", "bg_mode"]


def read_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def gt_tile_path(tile):
    direct = GT_DIR / tile
    if direct.exists():
        return direct
    for split in ("train", "test"):
        candidate = SHARED / split / "line" / tile
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no GT for {tile}")


def score_one(img_path, tile, timeout_s, invert_pred=False):
    cmd = [str(PYTHON), str(WORKER), str(img_path), str(gt_tile_path(tile)), str(COND_DIR / tile)]
    if invert_pred:
        cmd.append("--invert-pred")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        print(f"  FAILED {tile}: {proc.stderr.strip()[:200]}", file=sys.stderr)
        return None
    return json.loads(proc.stdout.strip().splitlines()[-1])


def prepare_condition_pseudo_arm(tiles, out_dir):
    """profile_metrics reads a file and assumes GT polarity, so write the
    conditioning maps out inverted once and score those."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for tile in tiles:
        dst = out_dir / f"{Path(tile).stem}_out.png"
        if dst.exists():
            continue
        gray = np.asarray(Image.open(COND_DIR / tile).convert("L").resize((480, 480)))
        Image.fromarray(255 - gray).save(dst)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", default="results/ipadapter_probe_20260930")
    parser.add_argument("--sample-list", default=str(SHARED / "holdout_lineart_family.txt"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    tiles = read_list(args.sample_list)
    if args.limit:
        tiles = tiles[: args.limit]
    probe_root = Path(args.probe_root)

    cond_arm = prepare_condition_pseudo_arm(tiles, probe_root / "_condition_only")
    arms = [("condition (lineart_coarse)", cond_arm)]
    for d in sorted(probe_root.iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            arms.append((d.name, d))

    rows = []
    summary = []
    for name, arm_dir in arms:
        jobs = []
        for tile in tiles:
            p = arm_dir / f"{Path(tile).stem}_out.png"
            if p.exists():
                jobs.append((p, tile))
        if not jobs:
            print(f"[{name}] no outputs, skipped", file=sys.stderr)
            continue
        scored = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(score_one, p, t, args.timeout): t for p, t in jobs}
            for fut in futures:
                pass
            for fut, tile in futures.items():
                row = fut.result()
                if row is None:
                    continue
                row["arm"] = name
                row["tile"] = tile
                rows.append(row)
                scored.append(row)
        n_timeout = len(jobs) - len(scored)
        means = {c: float(np.mean([r[c] for r in scored])) for c in COLS}
        means.update(arm=name, n=len(scored), n_timeout=n_timeout)
        summary.append(means)
        print(f"[{name}] n={len(scored)} timeouts={n_timeout} "
              f"f1={means['gt_bsds_f1']:.4f} vs_cond={means['vs_condition_f1']:.4f} "
              f"near_white={means['near_white_frac']:.3f}", flush=True)

    per_tile = probe_root / "scores_per_tile.csv"
    with open(per_tile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "tile"] + COLS)
        w.writeheader()
        w.writerows(rows)
    summary_csv = probe_root / "scores_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "n", "n_timeout"] + COLS)
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {per_tile}\nwrote {summary_csv}")


if __name__ == "__main__":
    main()
