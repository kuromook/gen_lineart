#!/usr/bin/env python
"""Visual check for strokes.py: GT | junction spans (old unit) | linked strokes.

Each row is a 400px window of one panel at native resolution, drawn 2x.
Colours are one per token; black = junction node; grey = dropped (< min_len).
The window is the 400px square with the most skeleton, so the hard part shows.
"""
import argparse, csv, glob, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from strokes import strokes_from_skeleton, split, ink_mask
from skimage.morphology import skeletonize

ROOT = Path("/home/sh1/deepl/lineart")


def colours(n, seed):
    rng = np.random.default_rng(seed)
    hsv = np.stack([rng.integers(0, 180, n), rng.integers(160, 256, n), rng.integers(140, 230, n)], 1).astype(np.uint8)
    return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]


def pick_window(sk, win, mode):
    """densest: the worst case (fills, tone). typical: the median window among
    those with a normal amount of skeleton -- the densest window is always a
    fill, so a montage built only from it says nothing about plain line areas."""
    ii = cv2.integral(sk.astype(np.uint8))
    h, w = sk.shape
    if h <= win or w <= win:
        return 0, 0
    s = ii[win:, win:] - ii[:-win, win:] - ii[win:, :-win] + ii[:-win, :-win]
    s = s[::8, ::8]
    if mode == "densest":
        y, x = np.unravel_index(np.argmax(s), s.shape)
        return int(y * 8), int(x * 8)
    flat = s.ravel()
    cand = np.flatnonzero(flat >= 0.25 * flat.max())
    if not len(cand):
        return 0, 0
    pick = cand[np.argsort(flat[cand])[len(cand) // 2]]
    y, x = np.unravel_index(pick, s.shape)
    return int(y * 8), int(x * 8)


def paint(shape, groups, keep, node_mask):
    img = np.full(shape + (3,), 255, np.uint8)
    col = colours(len(keep), 7)
    for gi, pix in groups.items():
        r, c = pix
        img[r, c] = col[gi] if keep[gi] else (190, 190, 190)
    img[node_mask] = (0, 0, 0)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panels", required=True, help="source/name, comma separated")
    p.add_argument("--skel", default="results/panel_skeleton_20260917/skeleton")
    p.add_argument("--out", default="results/stroke_link_check_20260917/montage.png")
    p.add_argument("--win", type=int, default=400)
    p.add_argument("--hole-max", type=int, default=30)
    p.add_argument("--window", choices=["densest", "typical"], default="densest")
    p.add_argument("--max-bend", type=float, default=35.0)
    a = p.parse_args()
    man = {}
    for m in glob.glob(str(ROOT / "dataset/regions_*_koma_panels_2026*/manifest.csv")):
        for r in csv.DictReader(open(m)):
            man[(Path(m).parent.name, r["name"])] = r["native_line_path"]
    rows = []
    for item in a.panels.split(","):
        src, name = item.split("/", 1)
        g = cv2.imread(str(ROOT / man[(src, name)]), 0)
        cached = Path(f"{a.skel}/{src}/{name}")
        ink = ink_mask(g, a.hole_max)
        sk = cv2.imread(str(cached), 0) > 0 if cached.exists() else skeletonize(ink)
        wd = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5) * 2
        y, x = pick_window(sk, a.win, a.window)
        sl = (slice(y, y + a.win), slice(x, x + a.win))
        # tokenise the whole panel, then crop: links must see full context
        spans, sos, keep, info = strokes_from_skeleton(sk, wd, max_bend=a.max_bend)
        node_lab, _, span_lab, _ = split(sk)
        H, W = sk.shape
        # old unit: raw spans on the unpruned skeleton
        old = {}
        lens = []
        for k in range(1, int(span_lab.max()) + 1):
            pass
        flat = span_lab.ravel(); idx = np.flatnonzero(flat); lab = flat[idx]
        o = np.argsort(lab, kind="stable"); idx, lab = idx[o], lab[o]
        cuts = np.r_[0, np.flatnonzero(np.diff(lab)) + 1, len(lab)]
        for i in range(len(cuts) - 1):
            pp = idx[cuts[i]:cuts[i + 1]]
            old[i] = (pp // W, pp % W)
        old_keep = np.array([len(v[0]) >= 8 for v in old.values()])
        img_old = paint((H, W), old, old_keep, node_lab > 0)
        new = {}
        for si, (r, c) in enumerate(spans):
            k = sos[si]
            if k in new:
                new[k] = (np.r_[new[k][0], r], np.r_[new[k][1], c])
            else:
                new[k] = (r, c)
        _, _, _, _ = info, None, None, None
        pr_nodes, _, _, _ = split(sk)  # node positions for reference
        img_new = paint((H, W), new, keep, np.zeros((H, W), bool))
        gt = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        tiles = [cv2.resize(t[sl], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) for t in (gt, img_old, img_new)]
        pad = lambda t: cv2.copyMakeBorder(t, 0, 2 * a.win - t.shape[0], 0, 2 * a.win - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        row = np.hstack([np.hstack([pad(t), np.full((2 * a.win, 12, 3), 80, np.uint8)]) for t in tiles])
        label = f"{src.replace('regions_','').split('_koma')[0]}/{name[:48]}  spans>=8: {info['spans_kept']}  strokes: {info['strokes']}  links {info['links']}  spurs {info['spurs_removed']}"
        bar = np.full((34, row.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        rows.append(np.vstack([bar, row]))
        print(label, f"window y{y} x{x}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows))


if __name__ == "__main__":
    main()
