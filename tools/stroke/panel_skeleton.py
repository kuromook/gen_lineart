#!/usr/bin/env python
"""Skeletonize every GT line panel at native resolution, once, and cache it.

Track F restart, step 1. The context unit is the panel (コマ), not the 480px
training tile. Skeletonization is ~80% of tokenization cost (~1.1 s/MP on this
CPU), so it is computed once and stored as a 1-bit PNG; splitting and stroke
linking are then iterated on the cache.

Ink is gray < 128 with enclosed holes <= 30px filled (strokes.ink_mask), then
skimage skeletonize.
Every row keeps its source set and content fingerprint (when the manifest has
one) so that duplicates and holdout panels can be excluded at split time.
"""
import argparse, csv, glob, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from strokes import ink_mask  # noqa: E402

ROOT = Path("/home/sh1/deepl/lineart")
SETS = ["regions_clip_pairs_v3_koma_panels_20260826", "regions_ako5ver2_koma_panels_20260729",
        "regions_fitness_koma_panels_20260729", "regions_gakuen_koma_panels_20260729",
        "regions_hamlabi_koma_panels_20260729", "regions_housei_koma_panels_20260729",
        "regions_4th_koma_panels_20260801"]
FIELDS = ["source", "name", "fingerprint", "w", "h", "mp", "ink_frac", "skel_px", "sec", "status"]


def work(t):
    source, name, fp, line_path, out_path = t
    row = {"source": source, "name": name, "fingerprint": fp}
    s = time.time()
    if os.path.exists(out_path):
        g = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)
        sk = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        status = "cached"
    else:
        g = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            return {**row, "status": "missing"}
        sk = skeletonize(ink_mask(g)).astype(np.uint8) * 255
        tmp = out_path + ".tmp.png"
        cv2.imwrite(tmp, sk, [cv2.IMWRITE_PNG_BILEVEL, 1])
        os.replace(tmp, out_path)
        status = "ok"
    ink = ink_mask(g)
    return {**row, "w": g.shape[1], "h": g.shape[0], "mp": round(g.size / 1e6, 3),
            "ink_frac": round(float(ink.mean()), 4), "skel_px": int((sk > 0).sum()),
            "sec": round(time.time() - s, 2), "status": status}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/panel_skeleton_20260917")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    tasks = []
    for s in SETS:
        od = Path(a.out) / "skeleton" / s
        od.mkdir(parents=True, exist_ok=True)
        for r in csv.DictReader(open(ROOT / "dataset" / s / "manifest.csv")):
            lp = ROOT / r["native_line_path"]
            tasks.append((s, r["name"], r.get("content_fingerprint", ""), str(lp), str(od / r["name"])))
    if a.limit:
        step = max(1, len(tasks) // a.limit)
        tasks = tasks[::step][: a.limit]
    # biggest first: a memory or time problem shows up in the first minutes
    tasks.sort(key=lambda t: -os.path.getsize(t[3]) if os.path.exists(t[3]) else 0)
    out_csv = Path(a.out) / ("panels_smoke.csv" if a.limit else "panels.csv")
    t0 = time.time()
    with open(out_csv, "w", newline="") as f, Pool(a.workers) as pool:
        w = csv.DictWriter(f, FIELDS); w.writeheader()
        for i, row in enumerate(pool.imap_unordered(work, tasks, chunksize=1), 1):
            w.writerow(row); f.flush()
            if i % 100 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
