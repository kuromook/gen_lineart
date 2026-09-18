#!/usr/bin/env python
"""Visual half of the round-trip check: does the re-render look like the drawing?

Numbers alone have misled this project repeatedly, so the montage is part of the
criterion, not decoration. Columns: GT | re-render (v3) | difference, where red
is ink that was lost and blue is ink that was invented. Windows come from
`stroke_link_montage.pick_window` in both modes -- `typical` (ordinary line
areas) and `densest` (always a fill or tone, the worst case).
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strokes import ink_mask, strokes_from_skeleton
from stroke_path import chain, resample, span_paths, widths_at
from render_strokes import render, render_discs
from stroke_link_montage import pick_window
from roundtrip_check import ROOT, panel_paths


def rebuild(line_path, skel_path):
    g = cv2.imread(str(line_path), 0)
    sk = cv2.imread(str(skel_path), 0) > 0
    ink = ink_mask(g)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    spans, sos, keep, _ = strokes_from_skeleton(sk, dist * 2)
    groups = {}
    for i, (rr, cc) in enumerate(spans):
        groups.setdefault(int(sos[i]), []).append(i)
    polys, widths = [], []
    assigned = np.zeros_like(sk)
    for k, mem in groups.items():
        if not keep[k]:
            continue
        rr = np.concatenate([spans[i][0] for i in mem]); cc = np.concatenate([spans[i][1] for i in mem])
        assigned[rr, cc] = True
        path, _ = chain(span_paths(spans, mem))
        pts = resample(path, per_px=8.0, k_min=8, k_max=64)
        polys.append(pts); widths.append(widths_at(pts, dist))
    out = render(polys, widths, sk.shape)
    ys, xs = np.nonzero(sk & ~assigned)
    out |= render_discs(np.stack([ys, xs], 1), dist[ys, xs], sk.shape)
    return g, ink, out, sk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-panel", default="results/roundtrip_20260918/per_panel.csv")
    p.add_argument("--cache", default="results/panel_skeleton_20260917")
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--win", type=int, default=420)
    p.add_argument("--window", choices=["typical", "densest"], default="typical")
    p.add_argument("--worst", action="store_true", help="show the worst panels instead of a spread")
    p.add_argument("--out", default="results/roundtrip_20260918/montage_typical.png")
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.per_panel)))
    rows.sort(key=lambda r: float(r["v3_resid_recall"]))
    sel = rows[: a.rows] if a.worst else [rows[int(i)] for i in np.linspace(0, len(rows) - 1, a.rows)]
    man = panel_paths(a.cache)
    out_rows = []
    for r in sel:
        line = man[(r["source"], r["name"])]
        g, ink, out, sk = rebuild(line, Path(a.cache) / "skeleton" / r["source"] / r["name"])
        y, x = pick_window(sk, a.win, a.window)
        sl = (slice(y, y + a.win), slice(x, x + a.win))
        gt = cv2.cvtColor(g[sl], cv2.COLOR_GRAY2BGR)
        rend = np.full((a.win, a.win, 3), 255, np.uint8); rend[out[sl]] = (30, 30, 30)
        diff = np.full((a.win, a.win, 3), 255, np.uint8)
        both = ink[sl] & out[sl]
        diff[both] = (200, 200, 200)
        diff[ink[sl] & ~out[sl]] = (0, 0, 220)      # lost
        diff[out[sl] & ~ink[sl]] = (220, 120, 0)    # invented
        cells = [cv2.resize(c, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_NEAREST) for c in (gt, rend, diff)]
        row = np.hstack([np.hstack([c, np.full((c.shape[0], 10, 3), 80, np.uint8)]) for c in cells])
        bar = np.full((32, row.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"{r['name'][:46]}  recall {r['v3_resid_recall']}  extra {r['v3_resid_extra']}"
                         f"  f1 {r['v3_resid_f1']}  |  GT | re-render | diff(red=lost, blue=invented)",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1)
        out_rows.append(np.vstack([bar, row]))
        print(f"{r['name'][:44]} recall {r['v3_resid_recall']} window {a.window} y{y} x{x}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.vstack(out_rows))
    print("->", a.out)


if __name__ == "__main__":
    main()
