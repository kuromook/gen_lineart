#!/usr/bin/env python
"""Every cluster of 3-32 strokes as one normalised, fixed-size record.

Input to the codebook. A cluster is translated to its centroid and scaled so
its bounding box's long side is 56 (the C-3 frame), so the codebook sees
arrangement and shape, not where on the page or how large. The scale and
centroid are kept so a decoded cluster can be put back.

Output (results/cluster_set_20260919/):
  pts.npy    (N, 32, 16, 2) float16, normalised, zero-padded
  width.npy  (N, 32)        float16, ink width divided by the same scale
  mask.npy   (N, 32)        bool
  meta.csv   cluster index -> panel, work, group, split, scale, centroid, n
"""
import csv, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_infill import POINTS  # noqa: E402
from cloze import split_panels  # noqa: E402

PACK = Path("results/panel_pack_20260919")
OUT = Path("results/cluster_set_20260919")
MAXS, MINS, SIDE = 32, 3, 56.0


def main():
    arr = np.load(PACK / "strokes.npy", mmap_mode="r")
    lab = np.load(PACK / "cluster_labels_noend.npy")
    rows = list(csv.DictReader(open(PACK / "panels.csv")))
    works = {(r["source"], r["name"]): r["work"] for r in csv.DictReader(open(PACK / "panel_works.csv"))}
    test = split_panels(rows)
    recs, P, W, M = [], [], [], []
    t0 = time.time()
    for pi, r in enumerate(rows):
        s, n = int(r["start"]), int(r["n"])
        block = np.asarray(arr[s:s + n])
        pts = block[:, :POINTS * 2].reshape(n, POINTS, 2)
        wid = block[:, POINTS * 2 + 1]
        L = lab[s:s + n]
        for c in np.unique(L[L >= 0]):
            idx = np.flatnonzero(L == c)
            if not (MINS <= len(idx) <= MAXS):
                continue
            p = pts[idx]
            cen = p.reshape(-1, 2).mean(0)
            lo, hi = p.reshape(-1, 2).min(0), p.reshape(-1, 2).max(0)
            scale = SIDE / max(float((hi - lo).max()), 1e-3)
            q = np.zeros((MAXS, POINTS, 2), np.float16); w = np.zeros(MAXS, np.float16); m = np.zeros(MAXS, bool)
            q[:len(idx)] = ((p - cen) * scale).astype(np.float16)
            w[:len(idx)] = (wid[idx] * scale).astype(np.float16)
            m[:len(idx)] = True
            P.append(q); W.append(w); M.append(m)
            recs.append([pi, works[(r["source"], r["name"])], r["group"], "test" if test[pi] else "train",
                         round(scale, 5), round(float(cen[0]), 1), round(float(cen[1]), 1), len(idx)])
        if (pi + 1) % 1000 == 0:
            print(f"{pi+1}/{len(rows)} clusters {len(recs)} {time.time()-t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "pts.npy", np.stack(P)); np.save(OUT / "width.npy", np.stack(W)); np.save(OUT / "mask.npy", np.stack(M))
    with open(OUT / "meta.csv", "w", newline="") as f:
        w_ = csv.writer(f); w_.writerow(["panel", "work", "group", "split", "scale", "cy", "cx", "n"]); w_.writerows(recs)
    tr = sum(1 for x in recs if x[3] == "train")
    print(f"clusters {len(recs)} (train {tr} / test {len(recs)-tr}) -> {OUT}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
