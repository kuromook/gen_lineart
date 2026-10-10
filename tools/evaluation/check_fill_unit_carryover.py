"""IC2b: does `fill_ratio` carry across the unit change, 480 tile -> koma panel?

Registered 2026-10-10 in doc/work_log.md **with no pass bar**: the published
value is on 480 tiles from another track's train list, stage 2's corpus is
koma panels, and a difference between them is a measurement of the unit
change rather than a failure. IC2a is the one that can fail.

Three numbers, printed together so the unit effect is measured and not
assumed:

  published      the external 2026-09-11 inventory's tile value
  panels         our fill_ratio on clip_pairs_v3 koma panels
  windows        our fill_ratio on 480 windows cut from those SAME panels,
                 which isolates the unit from the material

The window number uses the registered stage-2 grid (`koma_windows.grid`), so
it is the unit the model will actually see.

Usage: check_fill_unit_carryover.py --n 120
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_fill_instrument import INVENTORY, SEED, published  # noqa: E402
from fill_mask import fill_ratio_cv2, fill_ratio_exact, ink_of  # noqa: E402
from koma_split import ROOT, load as load_split  # noqa: E402
from koma_windows import SIZE, grid, pad_to  # noqa: E402

OUT = Path("results/fill_koma_20261010")
ANCHOR = "trainlist:clippairskoma"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=120, help="panels, matching the "
                    "published run's 120-tile sample size")
    args = ap.parse_args()

    pub = published()
    want = float(pub[ANCHOR]["fill_ratio"])
    print(f"IC2b  external anchor {ANCHOR}: fill_ratio {want:.4f} "
          f"(n_total {pub[ANCHOR]['n_total']} tiles, {pub[ANCHOR]['n_sampled']} sampled)")
    print(f"      source: {INVENTORY}")
    print(f"      definition on both sides: share of ink in strokes wider than 8px\n")

    split = load_split()
    names = sorted(n for n, _, _ in split["train"] + split["held"])
    rng = random.Random(SEED)
    picked = rng.sample(names, min(args.n, len(names)))

    panel_vals, window_vals, exact_vals, rows = [], [], [], []
    for name in picked:
        g = cv2.imread(str(Path(ROOT) / "line" / name), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        ink = ink_of(g)
        pv = fill_ratio_cv2(ink)
        ev = fill_ratio_exact(ink)
        padded, _ = pad_to(ink.astype(np.uint8), SIZE, value=0)
        wv = [fill_ratio_cv2(padded[y:y + SIZE, x:x + SIZE].astype(bool))
              for (y, x) in grid(*padded.shape)]
        panel_vals.append(pv)
        exact_vals.append(ev)
        window_vals += wv
        rows.append({"panel": name, "panel_fill_ratio": round(pv, 6),
                     "panel_fill_ratio_exact": round(ev, 6),
                     "n_windows": len(wv),
                     "window_mean": round(float(np.mean(wv)), 6)})

    def line(label, v, extra=""):
        a = np.asarray(v, dtype=float)
        se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        print(f"{label:>34}  mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}  "
              f"se {se:.4f}  n {len(a)}   delta vs published {a.mean() - want:+.4f}{extra}")

    print(f"{'published tile value':>34}  mean {want:.4f}"
          f"{'':>38}   (external, not produced here)")
    line("our koma PANELS (cv2)", panel_vals)
    line("our koma PANELS (exact EDT)", exact_vals, "   sensitivity leg")
    line("our 480 WINDOWS from those panels", window_vals, "   <- the published unit")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "ic2b_unit_carryover.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["panel", "panel_fill_ratio",
                                          "panel_fill_ratio_exact", "n_windows",
                                          "window_mean"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nIC2b carries NO pass bar by registration; IC2a is the one that "
          f"can fail.\nsaved: {p}")


if __name__ == "__main__":
    main()
