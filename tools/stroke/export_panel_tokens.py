#!/usr/bin/env python
"""Export every panel's strokes as polylines, for the set-Transformer discriminator.

Gate 1 has failed three times with hand-made features; this is the other half of
the 2026-09-18 agreement -- change the discriminator, not the features. The model
gets the raw geometry of every stroke in a panel and has to say which one does
not belong, so nothing about "relations" is hand-specified.

Per stroke: the ordered skeleton path resampled to POINTS points (absolute
pixel coordinates), its pixel length, median ink width and fill share.
Negatives are made from these polylines at training time -- a shift or a
rotation of a polyline is exact, so none of the rasterization artifacts that
leaked in the tile-scale generator (make_negatives, rules 1-3) can appear.

housei is excluded (tone and wash, not line art).
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate1_panel_features import EXCLUDE_SOURCES, ROOT, panel_strokes  # noqa: E402
from make_negatives import order_path  # noqa: E402

POINTS = 16


def resample(path, k=POINTS):
    """k points evenly spaced along the path by arc length."""
    if len(path) < 2:
        return np.repeat(path[:1], k, 0).astype(np.float32)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if d[-1] <= 0:
        return np.repeat(path[:1], k, 0).astype(np.float32)
    t = np.linspace(0, d[-1], k)
    return np.stack([np.interp(t, d, path[:, 0]), np.interp(t, d, path[:, 1])], 1).astype(np.float32)


def work(t):
    src, name, line_path, skel_path, out_path, min_px = t
    if Path(out_path).exists():
        return src, name, -1, 0.0
    t0 = time.time()
    got = panel_strokes(line_path, skel_path, min_px, 1.01)  # keep fills, tag them
    if got is None or len(got[3]) < 8:
        return src, name, 0, time.time() - t0
    g, sk, dist, cs = got
    polys, meta = [], []
    for c in cs:
        p = order_path(c).astype(np.float32)
        polys.append(resample(p))
        w = dist[c[:, 0], c[:, 1]] * 2.0
        meta.append([len(c), float(np.median(w)), float((w > 8).mean())])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, poly=np.stack(polys), meta=np.array(meta, np.float32),
                        shape=np.array(sk.shape, np.int32))
    return src, name, len(polys), time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="results/panel_skeleton_20260917")
    p.add_argument("--out", default="results/panel_tokens_20260918")
    p.add_argument("--min-token-px", type=int, default=20)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    meta = [r for r in csv.DictReader(open(Path(a.cache) / "panel_strokes.csv"))
            if r["source"] not in EXCLUDE_SOURCES and int(r["strokes"]) >= 40]
    paths = {}
    for src in {r["source"] for r in meta}:
        for m in csv.DictReader(open(ROOT / "dataset" / src / "manifest.csv")):
            paths[(src, m["name"])] = (str(ROOT / m["native_line_path"]), m.get("content_fingerprint", ""))
    if a.limit:
        meta = meta[:: max(1, len(meta) // a.limit)][: a.limit]
    tasks = [(r["source"], r["name"], paths[(r["source"], r["name"])][0],
              str(Path(a.cache) / "skeleton" / r["source"] / r["name"]),
              str(Path(a.out) / r["source"] / (r["name"] + ".npz")), a.min_token_px) for r in meta]
    index = Path(a.out) / "index.csv"
    index.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(index, "w", newline="") as f, Pool(a.workers) as pool:
        w = csv.writer(f); w.writerow(["source", "name", "group", "strokes"])
        for i, (src, name, n, sec) in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
            if n:
                fp = paths[(src, name)][1] or f"{src}/{name}"
                w.writerow([src, name, fp, n]); f.flush()
            if i % 200 == 0:
                print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
