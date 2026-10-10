"""Is there a sign in the rough that a region will be blacked in?

User, 2026-10-10: "ラフ段階でベタとするサインは置かない ... ベタらしいサインを
置くのはベタにするかどうか判断に迷ってるケース". Solid fills concentrate in
hair, tops and tights; those are decided in advance, so the artist does not
mark them. A dark mark in the rough indicates *indecision*, not intent.

That is a claim with a sign, so it is tested with a sign. Per koma panel, on
the aligned rough/line pair:

  fill_covered    share of GT fill area that has dark rough ink under it
                  -- the recall ceiling of any "the rough already has it" arm
  dark_precision  share of dark rough ink that lands on GT fill
  lift            fill rate under dark rough ink, divided by the panel's own
                  fill rate. **Above 1 means a dark rough mark predicts a
                  fill; below 1 means it predicts the opposite**, which is
                  what the user's account implies and what no amount of
                  training would reverse.

Two readings of "dark rough ink", because roughs are grey: the project ink
threshold (128) and a strict one (64), reported side by side rather than
one chosen.

Usage: rough_fill_signal.py --root <panel dir> --label ako5_koma [--limit N]
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
DARK = (128, 64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    paths = sorted(p for p in (root / "line").iterdir() if p.suffix.lower() in (".png", ".jpg"))
    if args.limit:
        paths = paths[:args.limit]

    rows = []
    for p in paths:
        rp = root / "rough" / p.name
        if not rp.exists():
            continue
        line = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        rough = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
        if line is None or rough is None:
            continue
        if rough.shape != line.shape:
            rough = cv2.resize(rough, (line.shape[1], line.shape[0]),
                               interpolation=cv2.INTER_AREA)
        fill = fill_mask(ink_of(line))
        if fill.sum() < 500:
            continue
        row = {"image": p.name, "px": int(fill.size), "fill_px": int(fill.sum())}
        for d in DARK:
            dark = rough < d
            row[f"dark{d}_px"] = int(dark.sum())
            row[f"inter{d}_px"] = int((dark & fill).sum())
        # "a sign that this will be blacked in" is an AREA in the rough, not a
        # stroke: the same opening the GT side uses, applied to the rough.
        area = fill_mask(ink_of(rough))
        row["darkarea_px"] = int(area.sum())
        row["interarea_px"] = int((area & fill).sum())
        rows.append(row)

    R.mkdir(parents=True, exist_ok=True)
    out = R / f"roughsignal_{args.label}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tot_px = sum(r["px"] for r in rows)
    tot_fill = sum(r["fill_px"] for r in rows)
    base = tot_fill / tot_px
    print(f"{args.label}: {len(rows)} panels with fill, base fill rate {100 * base:.2f}% of area")
    print(f"{'rough<':>8}{'dark area':>12}{'fill_covered':>14}{'dark_precision':>16}{'lift':>8}")
    for d in list(DARK) + ["area"]:
        dk = sum(r[f"dark{d}_px"] for r in rows)
        it = sum(r[f"inter{d}_px"] for r in rows)
        prec = it / dk if dk else 0.0
        print(f"{str(d):>8}{100.0 * dk / tot_px:11.2f}%{100.0 * it / tot_fill:13.1f}%"
              f"{100.0 * prec:15.1f}%{prec / base:8.2f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
