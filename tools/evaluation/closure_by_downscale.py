"""Does a 面 close if you step back from it?

User's proposal, 2026-10-10: the strokes do not close at full resolution, but
a person sees the region as closed, so treat "closes when shrunk" as the
definition. Shrinking merges a stroke's two sides across a gap smaller than a
cell, which is the same thing the eye is doing.

    barrier_s = INTER_AREA downscale of the strokes by s, kept where the cell's
                ink coverage >= t
    regions_s = connected components of ~barrier_s (4-conn)

Scored against what it is for: can a solid fill be cut out as its own 面?
For every GT fill component of >= MIN_FILL_PX at native scale, find the region
that covers most of it and ask two things:

  enclosed    that region is not the panel-sized blob (area < BLOB_FRAC of the
              panel) -- the fill was separated from the background at all
  tightness   fill area / region area -- 1.0 means the region is the fill and
              nothing else; 0.1 means the fill was merged into something ten
              times its size, which is "enclosed" on a technicality

Both are needed: shrinking far enough closes everything by destroying it, so
`enclosed` alone rises to 100% for the wrong reason and `tightness` is what
catches that.

Which image the barrier comes from matters, and both are run:

  --from line   the GT with its fills removed. This is the de-filled drawing,
                and it is missing boundary the artist did draw -- a hair
                silhouette often exists only as the fill's own edge, so
                removing the fill removes the outline with it.
  --from rough  the rough, where that outline is still a line. This is also
                the image the task actually has at inference time, so it is
                the one the answer should be read from.

Usage: closure_by_downscale.py --root <panel dir> --from rough --label ...
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_mask, ink_of  # noqa: E402

R = Path("results/region_closure_20261010")
SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)
COVER = (0.10, 0.30)
MIN_FILL_PX = 500
BLOB_FRAC = 0.25


def one(gray, barrier_gray=None):
    ink = ink_of(gray)
    fill = fill_mask(ink)
    line_ink = ink_of(barrier_gray) if barrier_gray is not None else (ink & ~fill)
    nf, labf, stf, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8), 8)
    keep = [i for i in range(1, nf) if stf[i, cv2.CC_STAT_AREA] >= MIN_FILL_PX]
    if not keep:
        return []
    out = []
    for s in SCALES:
        if s == 1.0:
            cov = line_ink.astype(np.float32)
            fl = labf
        else:
            cov = cv2.resize(line_ink.astype(np.float32), None, fx=s, fy=s,
                             interpolation=cv2.INTER_AREA)
            fl = cv2.resize(labf, (cov.shape[1], cov.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        panel_px = cov.size
        for t in COVER:
            barrier = cov >= t
            n, lab, st, _ = cv2.connectedComponentsWithStats((~barrier).astype(np.uint8), 4)
            for i in keep:
                m = fl == i
                fa = int(m.sum())
                if fa == 0:
                    continue
                ids, cnt = np.unique(lab[m], return_counts=True)
                ok = ids > 0
                if not ok.any():
                    out.append({"scale": s, "cover": t, "fill_px": fa,
                                "region_px": 0, "enclosed": 0})
                    continue
                j = ids[ok][np.argmax(cnt[ok])]
                ra = int(st[j, cv2.CC_STAT_AREA])
                out.append({"scale": s, "cover": t, "fill_px": fa, "region_px": ra,
                            "panel_px": int(panel_px),
                            "enclosed": int(ra < BLOB_FRAC * panel_px)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="panel dir holding line/ and rough/")
    ap.add_argument("--from", dest="src", default="rough", choices=("rough", "line"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    paths = sorted(p for p in (root / "line").iterdir() if p.suffix.lower() in (".png", ".jpg"))
    if args.limit:
        paths = paths[:args.limit]
    rows = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        bg = None
        if args.src == "rough":
            rp = root / "rough" / p.name
            bg = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
            if bg is None:
                continue
            if bg.shape != g.shape:
                bg = cv2.resize(bg, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
        for r in one(g, bg):
            r["image"] = p.name
            rows.append(r)

    R.mkdir(parents=True, exist_ok=True)
    out = R / f"downscale_{args.label}_{args.src}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "scale", "cover", "fill_px",
                                          "region_px", "panel_px", "enclosed"])
        w.writeheader()
        w.writerows(rows)

    print(f"{args.label}: {len({r['image'] for r in rows})} panels, "
          f"{len([r for r in rows if r['scale'] == 1.0 and r['cover'] == COVER[0]])} fill regions")
    print(f"{'scale':>7}{'cover':>7}{'enclosed(area)':>16}{'median tightness':>18}"
          f"{'median region/panel':>21}")
    for s in SCALES:
        for t in COVER:
            sel = [r for r in rows if r["scale"] == s and r["cover"] == t]
            if not sel:
                continue
            tot = sum(r["fill_px"] for r in sel)
            enc = sum(r["fill_px"] for r in sel if r["enclosed"])
            tight = [r["fill_px"] / r["region_px"] for r in sel if r["region_px"]]
            share = [r["region_px"] / r["panel_px"] for r in sel if r["region_px"]]
            print(f"{s:7.4f}{t:7.2f}{100.0 * enc / tot:15.1f}%"
                  f"{(np.median(tight) if tight else float('nan')):18.3f}"
                  f"{(np.median(share) if share else float('nan')):21.3f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
