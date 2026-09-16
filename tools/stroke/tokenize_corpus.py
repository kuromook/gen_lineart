#!/usr/bin/env python
"""Tokenize GT line art into strokes and write one row per token.

Step 1 of Track F. The unit is a 1px skeleton segment split at crossing-number
junctions (`stroke_churn`), verified 2026-09-17 to return centrelines rather
than both edges of a stroke (area / (length x width) = 1.139).

Solid blacks are the known failure: their medial axis is a branching tree, so
tokens there are fragments rather than strokes. They are TAGGED, not dropped --
`fill_share` is the fraction of a token's pixels sitting on ink thicker than
`--fill-width-px` -- because deleting a fill would also delete the outline a
human drew. Filter on it downstream.
"""
import argparse, csv, sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from stroke_churn import ink_skeleton, junction_mask, load_gray, segments  # noqa: E402

FIELDS = ["tile", "token", "n_px", "width_p50", "fill_share", "straightness",
          "y0", "x0", "y1", "x1", "cy", "cx", "bbox_h", "bbox_w"]


def tokenize(args):
    name, line_dir, min_seg, fill_w = args
    try:
        g = load_gray(Path(line_dir) / name)
    except FileNotFoundError:
        return name, [], {}
    ink = g < 128
    if int(ink.sum()) < 200:
        return name, [], {"n_tokens": 0, "skeleton_px": 0, "captured": float("nan")}
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    sk = ink_skeleton(g)
    if not sk.any():
        return name, [], {"n_tokens": 0, "skeleton_px": 0, "captured": float("nan")}
    segs = segments(sk, min_seg)
    rows = []
    for i, (rr, cc) in enumerate(segs):
        w = dist[rr, cc] * 2.0
        # endpoints: the two skeleton pixels furthest apart in the segment
        pts = np.stack([rr, cc], 1).astype(np.float32)
        d0 = np.linalg.norm(pts - pts[0], axis=1)
        a = int(d0.argmax())
        d1 = np.linalg.norm(pts - pts[a], axis=1)
        b = int(d1.argmax())
        span = float(d1[b])
        rows.append({
            "tile": name, "token": i, "n_px": len(rr),
            "width_p50": round(float(np.median(w)), 3),
            "fill_share": round(float((w > fill_w).mean()), 4),
            "straightness": round(span / len(rr), 4) if len(rr) else 0.0,
            "y0": int(pts[a][0]), "x0": int(pts[a][1]),
            "y1": int(pts[b][0]), "x1": int(pts[b][1]),
            "cy": int(rr.mean()), "cx": int(cc.mean()),
            "bbox_h": int(rr.max() - rr.min() + 1), "bbox_w": int(cc.max() - cc.min() + 1),
        })
    tot = int(sk.sum())
    summary = {"n_tokens": len(segs), "skeleton_px": tot,
               "captured": sum(len(r) for r, _ in segs) / tot if tot else float("nan"),
               "junction_frac": int(junction_mask(sk).sum()) / tot if tot else float("nan"),
               "fill_skel_share": float((dist[sk] * 2 > fill_w).mean())}
    return name, rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = "/home/sh1/deepl/lineart-controlnet-sd15-refine/data"
    p.add_argument("--line-dir", default=f"{data}/line")
    p.add_argument("--list", default=f"{data}/train_list.txt")
    p.add_argument("--min-seg-px", type=int, default=8)
    p.add_argument("--fill-width-px", type=float, default=8.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output-dir", default="results/tokenize_corpus_20260917")
    a = p.parse_args()

    names = [l.strip() for l in open(a.list) if l.strip()]
    if a.limit:
        names = names[: a.limit]
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok_f = open(out / "tokens.csv", "w", newline="")
    tw = csv.DictWriter(tok_f, FIELDS)
    tw.writeheader()
    sums = []
    tasks = [(n, a.line_dir, a.min_seg_px, a.fill_width_px) for n in names]
    with Pool(a.workers) as pool:
        for i, (name, rows, s) in enumerate(pool.imap_unordered(tokenize, tasks, chunksize=16)):
            tw.writerows(rows)
            if s:
                s["tile"] = name
                sums.append(s)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(names)}", file=sys.stderr, flush=True)
    tok_f.close()
    with open(out / "per_tile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["tile", "n_tokens", "skeleton_px", "captured", "junction_frac", "fill_skel_share"])
        w.writeheader()
        w.writerows(sums)

    n = np.array([s["n_tokens"] for s in sums])
    cap = np.array([s["captured"] for s in sums], dtype=float)
    fil = np.array([s["fill_skel_share"] for s in sums], dtype=float)
    print(f"tiles {len(sums)}  tokens {int(n.sum()):,}")
    print(f"tokens/tile  median {np.median(n):.0f}  mean {n.mean():.1f}  p10 {np.percentile(n,10):.0f}  p90 {np.percentile(n,90):.0f}")
    print(f"captured     median {np.nanmedian(cap):.3f}  p10 {np.nanpercentile(cap,10):.3f}")
    print(f"fill share   median {np.nanmedian(fil):.3f}  p90 {np.nanpercentile(fil,90):.3f}  tiles>0.2 {(fil>0.2).mean():.3f}")
    print(f"written: {out/'tokens.csv'}  {out/'per_tile.csv'}")


if __name__ == "__main__":
    main()
