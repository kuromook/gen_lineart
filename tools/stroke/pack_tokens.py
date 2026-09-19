#!/usr/bin/env python
"""Flatten the per-panel npz corpus into one memory-mapped array.

Measured 2026-09-19: training the masked-prediction model straight off the npz
files ran at 636s/epoch with the GPU at 0% -- every sample re-opened a
compressed archive and resampled its strokes to 16 points in Python. The model
is small, so the loader is the whole cost. Packing once turns each sample into
a slice.

Layout: `strokes.npy` is (total_strokes, 16*2 + 5) float32 -- the 16 resampled
points, then [skeleton px, median width, fill share, arc length, n_spans] --
with `panels.csv` giving each panel's row range, source, name and group.
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_infill import POINTS, to16  # noqa: E402


def load(t):
    tokens, source, name = t
    d = np.load(Path(tokens) / source / (name + ".npz"))
    poly, kpts, meta = d["poly"], d["kpts"], d["meta"]
    pts = np.stack([to16(poly[j], int(kpts[j])) for j in range(len(poly))]).astype(np.float32)
    return source, name, pts.reshape(len(pts), -1), meta.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="results/panel_tokens_v2_20260918")
    p.add_argument("--out", default="results/panel_pack_20260919")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    rows = list(csv.DictReader(open(Path(a.tokens) / "index.csv")))
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    total = sum(int(r["strokes"]) for r in rows)
    dim = POINTS * 2 + 5
    arr = np.lib.format.open_memmap(out / "strokes.npy", mode="w+", dtype=np.float32,
                                    shape=(total, dim))
    t0, off, index = time.time(), 0, []
    with Pool(a.workers) as pool, open(out / "panels.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "name", "group", "start", "n"])
        gmap = {(r["source"], r["name"]): r["group"] for r in rows}
        for i, (source, name, pts, meta) in enumerate(
                pool.imap(load, [(a.tokens, r["source"], r["name"]) for r in rows], chunksize=8), 1):
            n = len(pts)
            if off + n > total:
                break
            arr[off:off + n, :POINTS * 2] = pts
            arr[off:off + n, POINTS * 2:] = meta[:, :5]
            w.writerow([source, name, gmap[(source, name)], off, n])
            off += n
            if i % 500 == 0:
                print(f"{i}/{len(rows)}  strokes {off}  {time.time()-t0:.0f}s", flush=True)
    arr.flush()
    print(f"panels {len(rows)}  strokes {off}/{total}  -> {out}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
