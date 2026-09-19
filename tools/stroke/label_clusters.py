#!/usr/bin/env python
"""Cluster label for every stroke of the packed corpus, with the adopted rule.

The rule was chosen on 2026-09-19 against a bar fixed in advance: k-NN
proximity (k=4) at Louvain resolution 2.0 -- the only candidate giving a cluster
size median inside 3-40 (6.0) at ARI 0.840. The cloze test (B') and the
vocabulary test (C-3) both need these labels, so they are computed once.

Output is aligned row-for-row with results/panel_pack_20260919/strokes.npy:
labels.npy (int32, cluster id local to its panel, -1 if none) and a work id
per panel for the stratified negatives -- clip_pairs holds many works, so
"same corpus" is too coarse to detect a style leak.
"""
import argparse, csv, re, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stroke_graph import build_edges, cap_degree, communities, load_panel  # noqa: E402


def work_id(source, name):
    if "clip_pairs" in source:
        m = re.match(r"koma_\d+_(.+?)__", name)
        return m.group(1) if m else source
    return source


def job(t):
    pack, start, n = t
    arr = np.load(Path(pack) / "strokes.npy", mmap_mode="r")
    pts, meta = load_panel(arr, start, n)
    e = cap_degree(build_edges(pts, meta, rule="knn"), n)
    return start, communities(e, n, 2.0, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="results/panel_pack_20260919")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    rows = list(csv.DictReader(open(Path(a.pack) / "panels.csv")))
    total = sum(int(r["n"]) for r in rows)
    labels = np.full(total, -1, np.int32)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (start, lab) in enumerate(pool.imap_unordered(
                job, [(a.pack, int(r["start"]), int(r["n"])) for r in rows], chunksize=4), 1):
            labels[start:start + len(lab)] = lab
            if i % 500 == 0:
                print(f"{i}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    np.save(Path(a.pack) / "cluster_labels.npy", labels)
    with open(Path(a.pack) / "panel_works.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "name", "work"])
        for r in rows:
            w.writerow([r["source"], r["name"], work_id(r["source"], r["name"])])
    works = {work_id(r["source"], r["name"]) for r in rows}
    print(f"labels for {total} strokes, {len(works)} works  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
