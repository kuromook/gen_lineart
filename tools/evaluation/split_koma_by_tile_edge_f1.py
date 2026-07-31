"""Split combined_koma_20260729 tiles into high/low correspondence groups
using the per-tile edge_f1 already computed by the koma tiling pipeline
(tools/pair_extraction/tile_region_manifest_480.py), instead of a generic
post-hoc agreement score. edge_f1 here is measured *after* the pipeline's
own coarse-to-fine alignment refinement, so it should isolate residual
content-correspondence ambiguity rather than raw coordinate misalignment
(see doc/work_log.md 2026-08-01 for why that distinction matters).
"""

import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-csvs", nargs="+", required=True)
    parser.add_argument("--score-field", default="edge_f1")
    parser.add_argument("--count", type=int, default=450)
    parser.add_argument("--high-list", required=True)
    parser.add_argument("--low-list", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    rows = []
    for path in args.tile_csvs:
        with open(path) as f:
            rows.extend(csv.DictReader(f))

    scored = [(r["name"], float(r[args.score_field])) for r in rows]
    scored.sort(key=lambda x: -x[1])

    high = scored[: args.count]
    low = scored[-args.count :]

    with open(args.high_list, "w") as f:
        f.write("\n".join(name for name, _ in high) + "\n")
    with open(args.low_list, "w") as f:
        f.write("\n".join(name for name, _ in low) + "\n")
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", args.score_field])
        writer.writerows(scored)

    print(f"rows={len(scored)} count={args.count}")
    print(f"high {args.score_field}: min={high[-1][1]:.4f} max={high[0][1]:.4f}")
    print(f"low  {args.score_field}: min={low[-1][1]:.4f} max={low[0][1]:.4f}")
    print(f"saved: {args.high_list}")
    print(f"saved: {args.low_list}")
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
