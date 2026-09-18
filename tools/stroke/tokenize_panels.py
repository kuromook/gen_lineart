#!/usr/bin/env python
"""Tokenize every cached panel skeleton into linked strokes (strokes.py).

Writes one row per kept stroke and one row per panel, next to the skeleton
cache. The same stats as tokenize_corpus (tile unit) so the two can be compared,
plus how many junction spans each stroke absorbed.
"""
import argparse, csv, glob, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strokes import ink_mask, strokes_from_skeleton  # noqa: E402

ROOT = Path("/home/sh1/deepl/lineart")
SFIELDS = ["source", "name", "stroke", "n_px", "n_spans", "width_p50", "fill_share", "straightness",
           "y0", "x0", "y1", "x1", "cy", "cx", "bbox_h", "bbox_w"]
PFIELDS = ["source", "name", "w", "h", "skel_px", "spans", "spans_kept", "nodes", "strokes", "links",
           "spurs_removed", "bridges", "captured_spans", "captured_strokes", "sec"]


def work(t):
    source, name, line_path, skel_path, fill_w, max_bend = t
    s0 = time.time()
    g = cv2.imread(line_path, 0)
    sk = cv2.imread(skel_path, 0)
    if g is None or sk is None:
        return None, []
    sk = sk > 0
    ink = ink_mask(g)
    wd = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5) * 2
    spans, sos, keep, info = strokes_from_skeleton(sk, wd, max_bend=max_bend)
    groups = {}
    for i, (r, c) in enumerate(spans):
        groups.setdefault(int(sos[i]), []).append(i)
    rows = []
    kept_px = 0
    for k, members in groups.items():
        if not keep[k]:
            continue
        rr = np.concatenate([spans[i][0] for i in members]); cc = np.concatenate([spans[i][1] for i in members])
        kept_px += len(rr)
        w = wd[rr, cc]
        pts = np.stack([rr, cc], 1).astype(np.float32)
        a = int(np.linalg.norm(pts - pts[0], axis=1).argmax())
        d1 = np.linalg.norm(pts - pts[a], axis=1); b = int(d1.argmax())
        rows.append({"source": source, "name": name, "stroke": len(rows), "n_px": len(rr), "n_spans": len(members),
                     "width_p50": round(float(np.median(w)), 3), "fill_share": round(float((w > fill_w).mean()), 4),
                     "straightness": round(float(d1[b]) / len(rr), 4),
                     "y0": int(pts[a][0]), "x0": int(pts[a][1]), "y1": int(pts[b][0]), "x1": int(pts[b][1]),
                     "cy": int(rr.mean()), "cx": int(cc.mean()),
                     "bbox_h": int(rr.max() - rr.min() + 1), "bbox_w": int(cc.max() - cc.min() + 1)})
    skel_px = int(sk.sum())
    span_px = sum(len(r) for r, _ in spans if len(r) >= 8)
    prow = {"source": source, "name": name, "w": g.shape[1], "h": g.shape[0], "skel_px": skel_px, **info,
            "captured_spans": round(span_px / max(skel_px, 1), 4), "captured_strokes": round(kept_px / max(skel_px, 1), 4),
            "sec": round(time.time() - s0, 2)}
    return prow, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="results/panel_skeleton_20260917")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--fill-width-px", type=float, default=8.0)
    p.add_argument("--max-bend", type=float, default=35.0)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    tasks = []
    for r in csv.DictReader(open(Path(a.cache) / "panels.csv")):
        if r["status"] not in ("ok", "cached"):
            continue
        man = ROOT / "dataset" / r["source"] / "manifest.csv"
        tasks.append((r["source"], r["name"], man, str(Path(a.cache) / "skeleton" / r["source"] / r["name"])))
    paths = {}
    for m in {t[2] for t in tasks}:
        for r in csv.DictReader(open(m)):
            paths[(m.parent.name, r["name"])] = str(ROOT / r["native_line_path"])
    tasks = [(s, n, paths[(s, n)], sp, a.fill_width_px, a.max_bend) for s, n, _, sp in tasks]
    if a.limit:
        tasks = tasks[:: max(1, len(tasks) // a.limit)][: a.limit]
    tag = "_smoke" if a.limit else ""
    t0 = time.time()
    with open(Path(a.cache) / f"strokes{tag}.csv", "w", newline="") as fs, \
         open(Path(a.cache) / f"panel_strokes{tag}.csv", "w", newline="") as fp, Pool(a.workers) as pool:
        ws = csv.DictWriter(fs, SFIELDS); ws.writeheader()
        wp = csv.DictWriter(fp, PFIELDS); wp.writeheader()
        for i, (prow, rows) in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
            if prow is None:
                continue
            wp.writerow(prow); ws.writerows(rows)
            if i % 200 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
