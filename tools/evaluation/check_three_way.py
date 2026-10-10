"""IC-t1 for the contour cutter, with the expected values written down first.

Registered 2026-10-10: a disc of radius 60 with two straight lines running
into it, one from the left and one from below.

  expected: 2 cut points -> the contour becomes EXACTLY 2 open pieces
  expected: a disc with NO line touching it -> 0 cut points, 1 `closed_uncut`

AMENDED 2026-10-10, before the measurement run, with the reason recorded.
The registered expectation also said both pieces must class `arc` or `split`.
That was wrong, and it was wrong about the MODEL, not the cutter: two cuts on
a circle give a 90-degree arc and a **270-degree** one, and an arc wider than
about 120 degrees is not a function of any straight axis, so no single
parabola in a PCA frame can fit it. IC-c1 already measured this -- a 120
degree arc at R=200 misses the parabola model by 7.3x and classes
`unexplained`. The run confirmed it exactly: `['arc', 'unexplained']`.

So the amended expectation separates the cutter from the model:
  case 1  disc + 2 lines  -> 2 pieces; the SHORT one `arc`, the long 270-degree
          one `unexplained` (the model's known limit, not a cutter failure)
  case 2  disc + 4 lines  -> 4 pieces of 90 degrees each, ALL `arc`. This is
          the case that tests cutter and model together without hitting the
          limit, and it is the one that would fail if the cuts were misplaced.
  case 3  disc alone      -> 0 cut points, 1 `closed_uncut`

Consequence for the measurement, stated before it runs: **population B's
explained share is depressed by construction wherever a fill contour has few
cut points**, because the pieces are then wide arcs the registered model
cannot fit. That is a property of keeping the model fixed, and it is reported
rather than corrected for.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_mask, ink_of  # noqa: E402
from stroke_three_way import population_b  # noqa: E402

import random


def canvas(n_lines=2, size=420, r=60):
    img = np.full((size, size), 255, np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), r, 0, -1)
    ends = [(20, c), (c, size - 20), (size - 20, c), (c, 20)][:n_lines]
    for (x, y) in ends:
        d = np.array([x - c, y - c], float)
        d /= np.hypot(*d)
        a = (np.array([c, c]) + d * (r - 2)).astype(int)
        cv2.line(img, (x, y), tuple(a), 0, 2)
    return ink_of(img)


def run(n_lines, exp_pieces, exp_uncut, exp_cls, label):
    ink = canvas(n_lines)
    fill = fill_mask(ink)
    rows, st = population_b(ink, fill, "real", random.Random(0))
    cls = sorted(r["cls"] for r in rows)
    cuts = sorted({r["n_cuts"] for r in rows})
    ok = (len(rows) == exp_pieces and st["closed_uncut"] == exp_uncut
          and cls == sorted(exp_cls))
    print(f"IC-t1  {label}: pieces {len(rows)} (expected {exp_pieces})  "
          f"closed_uncut {st['closed_uncut']} (expected {exp_uncut})  "
          f"cut points {cuts}  classes {cls} (expected {sorted(exp_cls)})  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    a = run(2, 2, 0, ["arc", "unexplained"], "disc + 2 lines  ")
    b = run(4, 4, 0, ["arc"] * 4, "disc + 4 lines  ")
    c = run(0, 0, 1, [], "disc alone      ")
    print(f"\nIC-t1 -> {'PASS' if (a and b and c) else 'FAIL'}")
    sys.exit(0 if (a and b and c) else 1)
