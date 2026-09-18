#!/usr/bin/env python
"""Export the panel corpus as stroke tokens that can be drawn back (v2).

Changes from v1 (`export_panel_tokens.py`), each measured on 2026-09-18:

  1. spans are chained across junction gaps (`stroke_path`). v1 used
     `order_path` over a gapped pixel set and kept only 0.63 of a stroke.
  2. points follow arc length (one per 8px, 8..64) instead of a fixed 16.
  3. a width at every point instead of one median per stroke.
  4. ink is hysteresis-thresholded (<128 seeded, grown to <200): a plain <128
     cut breaks faint lines into dots, and a dotted line is not one token.
  5. skeleton pixels that belong to no stroke (junction nodes, spurs, bridges,
     sub-min_len runs) are kept as JOINT tokens -- one per connected residue,
     as (row, col, radius, n_px) -- instead of being dropped.

Round trip on 60 random panels with 1-3: recall@2px 0.705 -> 0.987, over-draw
0.046 -> 0.005 (`results/roundtrip_20260918/`).

The skeleton has to be recomputed because the ink rule changed, so this script
caches it alongside the tokens rather than reading the 2026-09-17 cache.
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strokes import ink_mask, strokes_from_skeleton
from stroke_path import chain, resample, span_paths, widths_at
from render_strokes import agreement, render, render_discs

ROOT = Path("/home/sh1/deepl/lineart")
THR, GROW, PER_PX, K_MIN, K_MAX = 128, 200, 8.0, 8, 64


def work(t):
    source, name, line_path, out_npz, skel_png, check = t
    if Path(out_npz).exists():
        return {"source": source, "name": name, "status": "cached"}
    t0 = time.time()
    g = cv2.imread(line_path, 0)
    if g is None:
        return {"source": source, "name": name, "status": "missing"}
    ink = ink_mask(g, thr=THR, grow=GROW)
    if ink.sum() < 200:
        return {"source": source, "name": name, "status": "empty"}
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(ink)
    spans, sos, keep, info = strokes_from_skeleton(sk, dist * 2)
    groups = {}
    for i, (rr, cc) in enumerate(spans):
        groups.setdefault(int(sos[i]), []).append(i)
    polys, widths, meta = [], [], []
    assigned = np.zeros_like(sk)
    for k, mem in groups.items():
        if not keep[k]:
            continue
        rr = np.concatenate([spans[i][0] for i in mem]); cc = np.concatenate([spans[i][1] for i in mem])
        assigned[rr, cc] = True
        path, gaps = chain(span_paths(spans, mem))
        if len(path) < 2:
            continue
        pts = resample(path, PER_PX, K_MIN, K_MAX)
        w = widths_at(pts, dist)
        arc = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        polys.append(pts); widths.append(w)
        meta.append([len(rr), float(np.median(dist[rr, cc] * 2.0)),
                     float((dist[rr, cc] * 2.0 > 8).mean()), arc, len(mem)])
    if not polys:
        return {"source": source, "name": name, "status": "nostrokes"}
    kpts = np.array([len(p) for p in polys], np.int16)
    kmax = int(kpts.max())
    P = np.full((len(polys), kmax, 2), np.nan, np.float32)
    Wd = np.full((len(polys), kmax), np.nan, np.float32)
    for i, (p, w) in enumerate(zip(polys, widths)):
        P[i, :len(p)] = p; Wd[i, :len(w)] = w
    # joint tokens: one per connected residue of unassigned skeleton
    resid = sk & ~assigned
    nj, jlab = cv2.connectedComponents(resid.astype(np.uint8), connectivity=8)
    joints = []
    if nj > 1:
        ys, xs = np.nonzero(resid)
        lab = jlab[ys, xs]
        order = np.argsort(lab, kind="stable")
        ys, xs, lab = ys[order], xs[order], lab[order]
        cuts = np.r_[0, np.flatnonzero(np.diff(lab)) + 1, len(lab)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            yy, xx = ys[a:b], xs[a:b]
            rad = float(dist[yy, xx].max())
            joints.append([float(yy.mean()), float(xx.mean()), rad, float(b - a)])
    J = np.array(joints, np.float32) if joints else np.zeros((0, 4), np.float32)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = out_npz + ".tmp.npz"          # atomic: a kill mid-write must not
    np.savez_compressed(tmp_npz, poly=P, kpts=kpts, width=Wd,   # leave a half file
                        meta=np.array(meta, np.float32), joints=J,
                        shape=np.array(sk.shape, np.int32))
    Path(tmp_npz).replace(out_npz)
    if skel_png:
        Path(skel_png).parent.mkdir(parents=True, exist_ok=True)
        tmp = skel_png + ".tmp.png"
        cv2.imwrite(tmp, (sk.astype(np.uint8) * 255), [cv2.IMWRITE_PNG_BILEVEL, 1])
        Path(tmp).replace(skel_png)
    row = {"source": source, "name": name, "status": "ok", "strokes": len(polys),
           "joints": len(joints), "skel_px": int(sk.sum()),
           "assigned": round(float(assigned.sum()) / max(int(sk.sum()), 1), 4),
           "sec": round(time.time() - t0, 2)}
    if check:
        out = render([P[i, :kpts[i]] for i in range(len(P))],
                     [Wd[i, :kpts[i]] for i in range(len(P))], sk.shape)
        if len(joints):
            out |= render_discs(J[:, :2], J[:, 2], sk.shape)
        a = agreement(out, ink)
        row.update({"recall": round(a["recall"], 4), "extra": round(a["extra"], 4),
                    "f1": round(a["f1"], 4)})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="results/panel_tokens_20260918/index.csv")
    p.add_argument("--out", default="results/panel_tokens_v2_20260918")
    p.add_argument("--skel-cache", default="results/panel_skeleton_v2_20260918")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--check", action="store_true", help="also render back and measure fidelity")
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.index)))
    man = {}
    for m in sorted(ROOT.glob("dataset/regions_*_koma_panels_2026*/manifest.csv")):
        src = m.parent.name
        for r in csv.DictReader(open(m)):
            man[(src, r["name"])] = (str(ROOT / r["native_line_path"]), r.get("content_fingerprint", ""))
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]
    tasks = [(r["source"], r["name"], man[(r["source"], r["name"])][0],
              str(Path(a.out) / r["source"] / (r["name"] + ".npz")),
              str(Path(a.skel_cache) / r["source"] / r["name"]) if a.skel_cache else "",
              a.check) for r in rows]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); got = []
    idx_path = out / ("index_smoke.csv" if a.limit else "index.csv")
    with open(idx_path, "w", newline="") as f, Pool(a.workers) as pool:
        w = csv.writer(f); w.writerow(["source", "name", "group", "strokes", "joints"])
        for i, r in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
            got.append(r)
            if r.get("status") == "ok":
                w.writerow([r["source"], r["name"], man[(r["source"], r["name"])][1] or
                            f"{r['source']}/{r['name']}", r["strokes"], r["joints"]]); f.flush()
            if i % 100 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    ok = [r for r in got if r.get("status") == "ok"]
    print(f"\nok {len(ok)} / {len(got)} -> {out}")
    if ok:
        med = lambda k: float(np.median([r[k] for r in ok if k in r]))
        print(f"線/コマ {med('strokes'):.0f}  接合 {med('joints'):.0f}  骨格の帰属 {med('assigned'):.3f}  {med('sec'):.1f}s/コマ")
        if a.check and "recall" in ok[0]:
            print(f"再現: 回収 {med('recall'):.3f}  描きすぎ {med('extra'):.3f}  F1 {med('f1'):.3f}")


if __name__ == "__main__":
    main()
