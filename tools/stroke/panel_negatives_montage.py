#!/usr/bin/env python
"""Look at the negatives at panel scale before trusting any number.

Columns: true | displaced | rotated | foreign. Each cell is the candidate's
neighbourhood: grey = the rest of the drawing, green = the real stroke, red =
the fake. One row per candidate, each from a DIFFERENT panel (collecting in
iteration order once put six rows from one tile, 2026-09-17).
"""
import argparse, csv, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate1_panel_features import EXCLUDE_SOURCES, ROOT, panel_strokes, pool_task
from make_negatives import build_candidates

KINDS = ("true", "displaced", "rotated", "foreign")


def cell(shape, ctx, pts, is_true, cen, half, zoom):
    img = np.full(shape + (3,), 255, np.uint8)
    img[ctx] = (185, 185, 185)
    img[pts[:, 0], pts[:, 1]] = (60, 160, 60) if is_true else (60, 60, 220)
    y0 = int(np.clip(cen[0] - half, 0, max(0, shape[0] - 2 * half)))
    x0 = int(np.clip(cen[1] - half, 0, max(0, shape[1] - 2 * half)))
    crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    if crop.shape[0] < 2 * half or crop.shape[1] < 2 * half:
        crop = cv2.copyMakeBorder(crop, 0, 2 * half - crop.shape[0], 0, 2 * half - crop.shape[1],
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="results/panel_skeleton_20260917")
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--half", type=int, default=140)
    p.add_argument("--zoom", type=float, default=2.0)
    p.add_argument("--min-token-px", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--out", default="results/gate1_panels_20260918/negatives_montage.png")
    a = p.parse_args()
    meta = [r for r in csv.DictReader(open(Path(a.cache) / "panel_strokes.csv"))
            if r["source"] not in EXCLUDE_SOURCES and int(r["strokes"]) >= 80]
    paths = {}
    for src in {r["source"] for r in meta}:
        for m in csv.DictReader(open(ROOT / "dataset" / src / "manifest.csv")):
            paths[(src, m["name"])] = str(ROOT / m["native_line_path"])
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(meta))
    pool = []
    for i in order[: 8]:
        r = meta[i]
        pool += pool_task((r["source"], r["name"], paths[(r["source"], r["name"])],
                           str(Path(a.cache) / "skeleton" / r["source"] / r["name"]),
                           a.min_token_px, 0.2, 20, a.seed + int(i)))
    print(f"foreign pool {len(pool)}", flush=True)
    rows, used = [], 0
    for i in order[8:]:
        if len(rows) >= a.rows:
            break
        r = meta[i]
        got = panel_strokes(paths[(r["source"], r["name"])],
                            Path(a.cache) / "skeleton" / r["source"] / r["name"], a.min_token_px, 0.2)
        if got is None or len(got[3]) < 20:
            continue
        g, sk, dist, cs = got
        full = np.zeros(sk.shape, bool)
        for c in cs:
            full[c[:, 0], c[:, 1]] = True
        for ti in rng.permutation(len(cs))[:6]:
            c = cs[ti]
            if len(c) < 30:
                continue
            cands = build_candidates(c, pool, rng, sk.shape, dist)
            if any(v is None for v in cands.values()):
                continue
            ctx = full.copy(); ctx[c[:, 0], c[:, 1]] = False
            cen = c.mean(0)
            cells = [cell(sk.shape, ctx, cands[k][0], k == "true", cen, a.half, a.zoom) for k in KINDS]
            row = np.hstack([np.hstack([t, np.full((t.shape[0], 8, 3), 90, np.uint8)]) for t in cells])
            bar = np.full((28, row.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, f"{r['name'][:46]}  stroke {ti} ({len(c)}px)   " + "   ".join(
                f"{k}" for k in KINDS), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            rows.append(np.vstack([bar, row]))
            print(f"row {len(rows)}: {r['name'][:46]} stroke {ti} {len(c)}px", flush=True)
            break
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows))


if __name__ == "__main__":
    main()
