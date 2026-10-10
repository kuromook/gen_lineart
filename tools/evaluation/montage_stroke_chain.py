"""IC-j4, the external anchor (lesson 12): draw every chained stroke in its
own colour over the line PNG as it sits on disk, and look at it.

A number cannot say whether the chainer joined the arms a person would have
joined. This sheet can. Each chain gets one colour, so a stroke that runs
through several junctions reads as one continuous colour; a stroke that was
cut at a junction shows a colour change exactly there. Absorbed junction
stubs are grey and junction nodes are white dots, so the two failure shapes
(a join that should have been made, and a stub eaten by the merge) are both
visible rather than inferred.

Usage:
  montage_stroke_chain.py --root <panel dir> --which line --n 3
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import ink_of  # noqa: E402
import stroke_chain as sc  # noqa: E402

R = Path("results/stroke_chain_20261010")
SEED = 20261010
ZOOM = 520


def colours(n, rng):
    out = []
    for _ in range(n):
        h = rng.random()
        c = cv2.cvtColor(np.uint8([[[int(h * 180), 210, 235]]]), cv2.COLOR_HSV2BGR)
        out.append(tuple(int(v) for v in c[0, 0]))
    return out


def draw(ink, rng):
    frags, link, absorbed, stats = sc.build_links(ink, "real", random.Random(SEED))
    strokes = sc.assemble(frags, link, skip=absorbed)
    vis = np.full(ink.shape + (3,), 255, np.uint8)
    vis[ink] = (205, 205, 205)                        # the ink itself, pale
    for fi in sorted(absorbed):                        # stubs the merge ate
        xy = np.round(frags[fi]).astype(int)
        vis[xy[:, 1], xy[:, 0]] = (120, 120, 120)
    cols = colours(len(strokes), rng)
    order = sorted(range(len(strokes)), key=lambda i: strokes[i][0].shape[0])
    for i in order:                                    # long chains drawn last
        pts = np.round(strokes[i][0]).astype(int)
        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], False, cols[i], 2)
    return vis, stats, strokes


def busiest(ink, size):
    h, w = ink.shape
    s = min(size, h, w)
    integ = cv2.integral(ink.astype(np.uint8))
    best, by, bx = -1, 0, 0
    for y in range(0, h - s + 1, 40):
        for x in range(0, w - s + 1, 40):
            v = (integ[y + s, x + s] - integ[y, x + s]
                 - integ[y + s, x] + integ[y, x])
            if v > best:
                best, by, bx = v, y, x
    return by, bx, s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--which", default="line", choices=("rough", "line"))
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(SEED)
    paths = sorted(p for p in (Path(args.root) / args.which).iterdir()
                   if p.suffix.lower() in (".png", ".jpg"))[:args.n]
    R.mkdir(parents=True, exist_ok=True)
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        ink = ink_of(g)
        vis, stats, strokes = draw(ink, rng)
        out = R / f"icj4_{args.which}_{p.stem}.png"
        cv2.imwrite(str(out), vis)
        y, x, s = busiest(ink, ZOOM)
        z = cv2.resize(vis[y:y + s, x:x + s], (s * 2, s * 2),
                       interpolation=cv2.INTER_NEAREST)
        outz = R / f"icj4_{args.which}_{p.stem}_zoom.png"
        cv2.imwrite(str(outz), z)
        print(f"{p.name}: {len(strokes)} chains from {stats['frags']} fragments, "
              f"{stats['pairs']} joins -> {out.name}, {outz.name}")


if __name__ == "__main__":
    main()
