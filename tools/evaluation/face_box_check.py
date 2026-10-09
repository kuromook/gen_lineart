"""Does a detected face box fit inside its panel? A check for box orientation.

Lesson 12's remedy -- score points against the ink in the line PNGs -- reads
`strokes`/`pts`/`meta` and nothing else, so it has **no sensitivity** to how a
face box `(x0, y0, x1, y1)` was read. This is the cheap check that does: a box
read with x and y exchanged leaves the panel, because panels are not square.

Proposed by Track G on 2026-10-09 and shared here so the tracks that use the
boxes (G and H) run the same test. On
`../lineart-face-words/results/h3_face_parts_20261004/faces.csv` (5,531 boxes,
git-ignored and disk-only): read by column name 7.3% stick out with a 90th
percentile overflow of 0.4 px -- the detector clipping at the panel edge --
and read exchanged 40.5% leave the panel with a 90th percentile of 1,018.6 px.
94.3% of the 3,821 panels holding a box are off square by more than 5%, which
is what makes the two readings separable.

Usage:
    python tools/evaluation/face_box_check.py <faces.csv>

The csv needs columns x0, y0, x1, y1, png_h, png_w; rows with an empty x0 are
panels where the detector found nothing and are skipped.
"""

import csv
import math
import sys

TOLERANCE_PX = 2.0  # the detector clips at the edge; below this is not an error
ASPECT_TOLERANCE = 0.05  # panels within this of square cannot discriminate


def _percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def overflow(row, exchanged=False):
    """How far the box leaves the panel, in pixels, 0 if it fits."""
    x0, y0, x1, y1 = (float(row[k]) for k in ("x0", "y0", "x1", "y1"))
    h, w = float(row["png_h"]), float(row["png_w"])
    if exchanged:
        x0, y0, x1, y1 = y0, x0, y1, x1
    return max(0.0, -x0, -y0, x1 - w, y1 - h)


def check(csv_path, tolerance_px=TOLERANCE_PX):
    with open(csv_path) as handle:
        rows = [r for r in csv.DictReader(handle) if r["x0"].strip()]
    if not rows:
        raise ValueError(f"no rows with a box in {csv_path}")
    report = {"n_boxes": len(rows), "tolerance_px": tolerance_px}
    for name, exchanged in (("as_named", False), ("exchanged", True)):
        over = [overflow(r, exchanged) for r in rows]
        report[name] = {
            "out_of_panel_frac": sum(1 for o in over if o > tolerance_px) / len(over),
            "overflow_px_p90": _percentile(over, 90),
        }
    panels = {r["panel"]: (float(r["png_h"]), float(r["png_w"])) for r in rows}
    off_square = sum(
        1 for h, w in panels.values() if abs(h - w) / min(h, w) > ASPECT_TOLERANCE
    )
    report["panels"] = len(panels)
    report["non_square_frac"] = off_square / len(panels)
    # The check is only informative where the two readings differ at all.
    report["passed"] = (
        report["as_named"]["out_of_panel_frac"] < 0.20
        and report["exchanged"]["out_of_panel_frac"] > 2 * report["as_named"]["out_of_panel_frac"]
        and report["non_square_frac"] > 0.5
    )
    return report


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    report = check(argv[1])
    print(f"{report['n_boxes']} boxes in {report['panels']} panels, "
          f"{report['non_square_frac']:.1%} of panels off square")
    for name in ("as_named", "exchanged"):
        r = report[name]
        print(f"  {name:9s} out of panel {r['out_of_panel_frac']:.4f} "
              f"(overflow p90 {r['overflow_px_p90']:.1f} px)")
    print(f"passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
