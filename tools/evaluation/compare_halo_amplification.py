"""Compare atari and final halo metrics sample-by-sample."""

import argparse
import csv
from pathlib import Path

import numpy as np


PAIR_KEYS = {
    "raw": ("raw_atari", "cleanup_msgan"),
    "dog": ("dog_atari", "dog_final"),
    "lucy_mild": ("lucy_mild_atari", "lucy_mild_final"),
    "lucy_thin": ("lucy_thin_atari", "lucy_thin_final"),
}


def load_rows(path):
    rows = {}
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            rows[(row["sample"], row["model"])] = row
    return rows


def f(row, key):
    return float(row[key])


def ratio(after, before):
    return after / max(before, 1e-6)


def pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def mean(values):
    return float(np.mean(values)) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--halo-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    rows = load_rows(args.halo_csv)
    samples = sorted({sample for sample, _model in rows})
    out_rows = []
    for sample in samples:
        for label, (atari_model, final_model) in PAIR_KEYS.items():
            atari = rows.get((sample, atari_model))
            final = rows.get((sample, final_model))
            if atari is None or final is None:
                continue
            out_rows.append(
                {
                    "sample": sample,
                    "pair": label,
                    "atari_halo_ink": f(atari, "halo_band_ink_mean"),
                    "final_halo_ink": f(final, "halo_band_ink_mean"),
                    "halo_ink_amplification": ratio(
                        f(final, "halo_band_ink_mean"),
                        f(atari, "halo_band_ink_mean"),
                    ),
                    "atari_halo_faint": f(atari, "halo_band_faint_ratio"),
                    "final_halo_faint": f(final, "halo_band_faint_ratio"),
                    "halo_faint_amplification": ratio(
                        f(final, "halo_band_faint_ratio"),
                        f(atari, "halo_band_faint_ratio"),
                    ),
                    "atari_far_ink": f(atari, "far_bg_ink_mean"),
                    "final_far_ink": f(final, "far_bg_ink_mean"),
                    "far_ink_amplification": ratio(
                        f(final, "far_bg_ink_mean"),
                        f(atari, "far_bg_ink_mean"),
                    ),
                    "atari_halo_to_core": f(atari, "halo_to_core"),
                    "final_halo_to_core": f(final, "halo_to_core"),
                }
            )

    fields = list(out_rows[0]) if out_rows else ["sample", "pair"]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    summary_rows = []
    for label in PAIR_KEYS:
        part = [row for row in out_rows if row["pair"] == label]
        summary_rows.append(
            {
                "pair": label,
                "samples": len(part),
                "atari_halo_ink": mean([row["atari_halo_ink"] for row in part]),
                "final_halo_ink": mean([row["final_halo_ink"] for row in part]),
                "halo_ink_amplification": mean([row["halo_ink_amplification"] for row in part]),
                "atari_halo_faint": mean([row["atari_halo_faint"] for row in part]),
                "final_halo_faint": mean([row["final_halo_faint"] for row in part]),
                "halo_faint_amplification": mean([row["halo_faint_amplification"] for row in part]),
                "atari_final_halo_ink_corr": pearson(
                    [row["atari_halo_ink"] for row in part],
                    [row["final_halo_ink"] for row in part],
                ),
                "atari_final_halo_faint_corr": pearson(
                    [row["atari_halo_faint"] for row in part],
                    [row["final_halo_faint"] for row in part],
                ),
            }
        )

    summary_fields = list(summary_rows[0]) if summary_rows else ["pair"]
    with open(args.summary_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("pair samples atari_halo final_halo amp corr")
    for row in summary_rows:
        print(
            f"{row['pair']} {row['samples']} "
            f"{row['atari_halo_ink']:.4f} {row['final_halo_ink']:.4f} "
            f"{row['halo_ink_amplification']:.3f} "
            f"{row['atari_final_halo_ink_corr']:.3f}"
        )
    print(f"saved: {args.output_csv}")
    print(f"saved: {args.summary_csv}")


if __name__ == "__main__":
    main()
