#!/usr/bin/env python
"""Look at the clusters: would a person name any of them?

The pre-registered visual bar (plan 2026-09-18): in at least half the windows,
one cluster is something a person would name -- an eye, a lock of hair, the jaw
contour, that hatch patch. Numbers alone have misled this project repeatedly,
and a partition can be stable (ARI 0.83) while meaning nothing.

Left: the panel's strokes in grey. Right: the same strokes coloured per cluster,
singletons left grey.
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stroke_graph import build_edges, cap_degree, communities, load_panel
from train_infill import POINTS


def draw(pts, labels, shape, colours=None, grey_singletons=True):
    img = np.full(shape + (3,), 255, np.uint8)
    sizes = np.bincount(labels[labels >= 0], minlength=(labels.max() + 1 if labels.max() >= 0 else 1))
    for i, p in enumerate(pts):
        xy = np.round(p[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        if colours is None:
            col = (170, 170, 170)
        else:
            k = labels[i]
            col = (190, 190, 190) if (k < 0 or (grey_singletons and sizes[k] < 2)) else tuple(int(v) for v in colours[k % len(colours)])
        cv2.polylines(img, [xy], False, col, 2)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="results/panel_pack_20260919")
    p.add_argument("--out", default="results/stroke_clusters_20260919/montage_clusters.png")
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--win", type=int, default=520)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--tau-rule", default="width")
    p.add_argument("--seed", type=int, default=20260919)
    a = p.parse_args()
    rows = [r for r in csv.DictReader(open(Path(a.pack) / "panels.csv")) if 80 <= int(r["n"]) <= 700]
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(20260918); rng.shuffle(groups)
    test = set(groups[: int(len(groups) * 0.25)])
    dev = [r for r in rows if r["group"] not in test]
    rng2 = np.random.default_rng(a.seed)
    sel = [dev[i] for i in rng2.choice(len(dev), a.rows, replace=False)]
    arr = np.load(Path(a.pack) / "strokes.npy", mmap_mode="r")
    hsv = np.stack([rng2.integers(0, 180, 256), np.full(256, 220), np.full(256, 190)], 1).astype(np.uint8)
    cols = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]
    out_rows = []
    for r in sel:
        pts, meta = load_panel(arr, int(r["start"]), int(r["n"]))
        lab = communities(cap_degree(build_edges(pts, meta, a.tau_rule), len(pts)),
                          len(pts), a.resolution, 0)
        hi = int(np.ceil(pts[..., 0].max())) + 8, int(np.ceil(pts[..., 1].max())) + 8
        plain = draw(pts, lab, hi)
        col = draw(pts, lab, hi, cols)
        # window: densest part of the drawing
        occ = np.zeros(hi, np.uint8)
        c = np.round(pts.reshape(-1, 2)).astype(int)
        occ[np.clip(c[:, 0], 0, hi[0] - 1), np.clip(c[:, 1], 0, hi[1] - 1)] = 1
        ii = cv2.integral(occ)
        w = min(a.win, hi[0] - 1, hi[1] - 1)
        s = ii[w:, w:] - ii[:-w, w:] - ii[w:, :-w] + ii[:-w, :-w]
        s = s[::8, ::8]
        flat = s.ravel(); cand = np.flatnonzero(flat >= 0.4 * flat.max())
        pick = cand[np.argsort(flat[cand])[len(cand) // 2]] if len(cand) else 0
        y, x = np.unravel_index(pick, s.shape); y, x = int(y * 8), int(x * 8)
        sl = (slice(y, y + w), slice(x, x + w))
        sizes = np.bincount(lab[lab >= 0])
        cells = [cv2.resize(t[sl], (520, 520), interpolation=cv2.INTER_AREA) for t in (plain, col)]
        row = np.hstack([np.hstack([t, np.full((520, 10, 3), 80, np.uint8)]) for t in cells])
        bar = np.full((30, row.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"[{a.tau_rule} res{a.resolution}] {r['name'][:34]}  strokes {r['n']}  clusters {len(sizes)}  "
                         f"largest {sizes.max() if len(sizes) else 0}  singletons {(sizes==1).mean():.0%}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        out_rows.append(np.vstack([bar, row]))
        print(f"{r['name'][:40]} clusters {len(sizes)} window y{y} x{x}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.vstack(out_rows))
    print("->", a.out)


if __name__ == "__main__":
    main()
