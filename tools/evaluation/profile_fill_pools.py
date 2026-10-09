"""What the target actually looks like: GT solid-fill statistics per tile.

No model, no arm, no hypothesis -- this is the description the scoring tables
are read against, and it is the first time these pools have been measured on
the fill *area* rather than the ink-normalised `fill_ratio`.

It reports both, deliberately, because they are easy to confuse: ako5's
published `fill_ratio` of 25.6% is the share of its *ink* in thick regions,
which is about 2.4% of the page once ink_ratio 0.0920 is applied.

Usage:  profile_fill_pools.py [--workers 8]
"""

import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_mask, fill_ratio_cv2, ink_of, load_gray  # noqa: E402

SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
R = Path("results/baseline_fill_20261009")
NEAR_BLANK_INK = 0.002
HAS_FILL_AREA = 0.001
FULL_FILL_AREA = 0.90


def one(task):
    split, tile = task
    stem = tile[:-4] if tile.endswith(".jpg") else tile
    gray = load_gray(SHARED / split / "line" / f"{stem}.jpg")
    ink = ink_of(gray)
    fill = fill_mask(ink)
    pool = stem.split("_")[0] + ("_test" if split == "test" else "")
    return {
        "tile": stem,
        "pool": pool,
        "split": split,
        "ink_ratio": round(float(ink.mean()), 6),
        "fill_ratio": round(fill_ratio_cv2(ink), 6),
        "fill_area_frac": round(float(fill.mean()), 6),
        "near_blank": int(ink.mean() < NEAR_BLANK_INK),
        "has_fill": int(fill.mean() > HAS_FILL_AREA),
        "full_fill": int(fill.mean() > FULL_FILL_AREA),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(R / "gt_fill_profile.csv"))
    args = ap.parse_args()

    tasks = [("train", t.strip()) for t in open(R / "lists/all_fill_pools.txt") if t.strip()]
    tasks += [("test", t.strip()) for t in open(R / "lists/housei_test.txt") if t.strip()]
    print(f"{len(tasks)} tiles", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one, tasks, chunksize=32))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[r["pool"]].append(r)
    print(f"\n{'pool':14}{'n':>7}{'ink':>8}{'fill_ratio':>12}{'fill_area':>11}"
          f"{'nearblank%':>12}{'has_fill%':>11}{'full_fill%':>12}")
    for pool in sorted(by, key=lambda p: -len(by[p])):
        rs = by[pool]
        n = len(rs)
        print(f"{pool:14}{n:7d}"
              f"{np.mean([r['ink_ratio'] for r in rs]):8.4f}"
              f"{np.mean([r['fill_ratio'] for r in rs]):12.4f}"
              f"{np.mean([r['fill_area_frac'] for r in rs]):11.4f}"
              f"{100 * np.mean([r['near_blank'] for r in rs]):11.1f}%"
              f"{100 * np.mean([r['has_fill'] for r in rs]):10.1f}%"
              f"{100 * np.mean([r['full_fill'] for r in rs]):11.1f}%")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
