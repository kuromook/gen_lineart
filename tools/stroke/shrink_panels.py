#!/usr/bin/env python
"""Small copies of the panels for the WD14 tagger.

The tagger pads to square and resizes to 448 anyway, but reading a 4-40MP PNG
costs 5-6s per panel (measured: 6.46s/img on the native files, 9h for the
corpus). A 768px JPEG copy costs ~0.2s to make and lets the tagger run at its
real speed. The tags are panel-level content labels, so nothing is lost by
tagging the small copy -- and this is recorded because it means the tags say
nothing about line-level detail.
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2

ROOT = Path("/home/sh1/deepl/lineart")


def work(t):
    src, name, path, out, long_side = t
    if Path(out).exists():
        return 1
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return 0
    h, w = g.shape
    s = long_side / max(h, w)
    if s < 1.0:
        g = cv2.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, g, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="results/panel_tokens_20260918/index.csv")
    p.add_argument("--out", default="results/panel_small_20260918")
    p.add_argument("--long-side", type=int, default=768)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.index)))
    man = {}
    for m in sorted(ROOT.glob("dataset/regions_*_koma_panels_2026*/manifest.csv")):
        s = m.parent.name
        for r in csv.DictReader(open(m)):
            man[(s, r["name"])] = str(ROOT / r["native_line_path"])
    tasks = [(r["source"], r["name"], man[(r["source"], r["name"])],
              str(Path(a.out) / r["source"] / (Path(r["name"]).stem + ".jpg")), a.long_side)
             for r in rows]
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool:
        for i, ok in enumerate(pool.imap_unordered(work, tasks, chunksize=8), 1):
            n += ok
            if i % 500 == 0:
                print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    with open(Path(a.out) / "list.txt", "w") as f:
        for r in rows:
            f.write(f"{r['source']}/{Path(r['name']).stem}.jpg\n")
    print(f"{n} copies -> {a.out}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
