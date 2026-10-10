"""Read the H-stroke CSVs and cut them the two ways that matter.

The registered headline is length-weighted explained share on strokes >=30px
with in_fill < 0.5. That subset is dominated by chains of ONE fragment --
the chainer only joins about a fifth of arms -- so real and control differ
very little there, which is a property of the metric, not evidence either
way. The second cut restricts to chains that were actually built (>= 2
fragments), where the arms genuinely differ. It is a diagnostic, printed
beside the registered number; it does not replace it.

Usage: report_stroke_chain.py --label ako5 --which line
"""

import argparse
import csv
from pathlib import Path

R = Path("results/stroke_chain_20261010")
EXPL = ("line", "arc", "split")


def load(label, which, mode):
    with open(R / f"strokes_{label}_{which}_{mode}.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["path_px"] = float(r["path_px"])
        r["in_fill"] = float(r["in_fill"])
        r["n"] = int(r["n"])
        r["n_frags"] = int(r["n_frags"])
        r["multi_ok"] = int(r["multi_ok"])
    return rows


def share(rows):
    tot = sum(r["n"] for r in rows)
    if not tot:
        return 0, 0, 0.0, 0.0
    e = 100.0 * sum(r["n"] for r in rows if r["cls"] in EXPL) / tot
    m = 100.0 * sum(r["n"] for r in rows if r["multi_ok"]) / tot
    return len(rows), tot, e, m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="ako5")
    ap.add_argument("--which", default="line")
    args = ap.parse_args()
    cuts = [
        ("REGISTERED: >=30px, outside fill",
         lambda r: r["path_px"] >= 30 and r["in_fill"] < 0.5),
        ("  of which chained (>=2 frags)",
         lambda r: r["path_px"] >= 30 and r["in_fill"] < 0.5 and r["n_frags"] >= 2),
        ("  of which single fragment",
         lambda r: r["path_px"] >= 30 and r["in_fill"] < 0.5 and r["n_frags"] == 1),
        (">=30px, inside fill",
         lambda r: r["path_px"] >= 30 and r["in_fill"] >= 0.5),
        ("chained, >=3 frags, outside fill",
         lambda r: r["in_fill"] < 0.5 and r["n_frags"] >= 3),
    ]
    data = {m: load(args.label, args.which, m) for m in ("real", "nc_j1", "nc_j2")}
    print(f"{args.which}: length-weighted explained share (line/arc/split, RMS<=1px)\n")
    print(f"{'subset':>34}{'arm':>7}{'strokes':>9}{'px':>9}{'EXPL':>8}"
          f"{'vs real':>9}{'multi*':>8}{'medlen':>8}{'medfrg':>7}")
    for name, sel in cuts:
        base = None
        for m in ("real", "nc_j1", "nc_j2"):
            rows = [r for r in data[m] if sel(r)]
            n, tot, e, mu = share(rows)
            if base is None:
                base = e
            lens = sorted(r["path_px"] for r in rows)
            frg = sorted(r["n_frags"] for r in rows)
            d = "" if m == "real" else f"{e - base:+.1f}"
            print(f"{(name if m == 'real' else ''):>34}{m:>7}{n:9d}{tot:9d}"
                  f"{e:7.1f}%{d:>9}{mu:7.1f}%"
                  f"{(lens[len(lens) // 2] if lens else 0):8.0f}"
                  f"{(frg[len(frg) // 2] if frg else 0):7d}")
        print()
    print("* multi = EXPLORATORY length-scaled breakpoint budget; cannot move the bar")


if __name__ == "__main__":
    main()
