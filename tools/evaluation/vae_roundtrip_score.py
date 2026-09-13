"""Track D, hypothesis 1 (scoring phase): score VAE-roundtrip images (from
`vae_roundtrip_generate.py`) against their GT tiles.

Deliberately a separate process from generation -- see
`vae_roundtrip_generate.py`'s docstring for why (a reproducible stall
observed when PyTorch/CUDA calls and this script's OpenCV-based scoring
shared one process). This script never imports `torch`.

Each tile is scored in its **own subprocess** (`vae_roundtrip_score_one.py`),
dispatched with a hard per-tile timeout, run `--workers` at a time. This is
not a performance optimization -- it is a correctness requirement found
during this track (2026-09-13): `tile_region_manifest_480
.bipartite_match_f1`'s scipy `maximum_bipartite_matching` was observed to
blow up to several hundred seconds, and in one case 12+ minutes, for
specific edge-point configurations that are not visually or statistically
distinguishable from fast tiles beforehand (not a density effect -- two
~9,000-edge-point tiles reproduced it, many denser tiles did not). With 584
tiles (192+100 GT tiles x 2 models) and no way to predict which will hit
this, a serial run has an unbounded worst case; per-tile subprocess timeouts
bound it, at the cost of a `bsds_f1`/`chamfer_px` of `NaN` (flagged
`timed_out=True`) for whichever tiles exceed the cap -- the `recon_*`/`gt_*`
profile metrics (never observed to be slow) are still filled in by
re-running just `profile_metrics` inline for a timed-out tile, so only the
matching-based columns are lost.

Scores: `gt_bsds_f1` (one-to-one bipartite-matched F1) plus the paper axes
(`near_white_frac`/`midtone_frac`/`fill_ratio`/`bg_mode`) from
`measure_lineart_profile.py`. If roundtrip f1 lands near this project's
current best trained score (~0.25), the VAE alone explains every failed run
to date; if it lands near ~0.9, the ceiling is not the problem and the cause
is the loss function or the data (hypotheses 2/3). Pools are never averaged
together (project rule, `doc/CURRENT.md` lesson) -- `lineart_family`
(deletion-only residual) and `housei` (solid fills the preprocessor cannot
draw at all) are different tasks with different baselines.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pair_extraction"))
from measure_lineart_profile import profile_metrics  # noqa: E402

SCORE_ONE = str(Path(__file__).resolve().parent / "vae_roundtrip_score_one.py")
PROFILE_KEYS = ("near_white_frac", "midtone_frac", "bg_mode", "fill_ratio", "ink_ratio", "line_width_p50")
FIELDS = ["model", "pool", "sample", "bsds_f1", "bsds_precision", "bsds_recall", "chamfer_px", "timed_out"]
for _key in PROFILE_KEYS:
    FIELDS += [f"recon_{_key}", f"gt_{_key}"]


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def score_entry(entry, python_bin, timeout_s):
    t0 = time.time()
    proc = subprocess.run(
        [python_bin, SCORE_ONE, "--gt", entry["gt_path"], "--recon", entry["recon_path"]],
        capture_output=True, text=True, timeout=None if timeout_s <= 0 else timeout_s,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"{entry['sample']} failed: {proc.stderr[-2000:]}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    row = {"model": entry["model"], "pool": entry["pool"], "sample": entry["sample"], "timed_out": False, **result}
    return row, elapsed


def timeout_fallback_row(entry):
    """Matching-based columns are lost, but profile_metrics is always fast
    (never observed to hang/blow up in this track) so still fill those in."""
    row = {
        "model": entry["model"], "pool": entry["pool"], "sample": entry["sample"], "timed_out": True,
        "bsds_f1": "", "bsds_precision": "", "bsds_recall": "", "chamfer_px": "",
    }
    recon_profile = profile_metrics(entry["recon_path"])
    gt_profile = profile_metrics(entry["gt_path"])
    for key in PROFILE_KEYS:
        row[f"recon_{key}"] = recon_profile[key]
        row[f"gt_{key}"] = gt_profile[key]
    return row


def print_summary(rows):
    import numpy as np

    by_key = {}
    for row in rows:
        by_key.setdefault((row["model"], row["pool"]), []).append(row)

    cols = ["bsds_f1", "bsds_precision", "bsds_recall", "chamfer_px", "recon_near_white_frac", "recon_midtone_frac", "recon_fill_ratio", "recon_ink_ratio", "gt_ink_ratio"]
    header = "model".ljust(8) + "pool".ljust(16) + "n".rjust(5) + "timeout".rjust(9) + "".join(c[:14].rjust(16) for c in cols)
    print(header)
    for (model, pool), group in sorted(by_key.items()):
        scored = [r for r in group if not r["timed_out"]]
        n_timeout = len(group) - len(scored)
        means = {c: (np.mean([float(r[c]) for r in scored]) if (scored and scored[0][c] != "") else float("nan")) for c in cols}
        line = model.ljust(8) + pool.ljust(16) + str(len(group)).rjust(5) + str(n_timeout).rjust(9) + "".join(f"{means[c]:16.4f}" for c in cols)
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-csv", default="results/vae_roundtrip_20260913/manifest.csv")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300.0, help="per-tile hard cap in seconds (<=0 disables)")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest_csv)
    rows = [None] * len(manifest)
    n_done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_entry, entry, args.python_bin, args.timeout): (i, entry) for i, entry in enumerate(manifest)}
        for future in as_completed(futures):
            i, entry = futures[future]
            label = f"{entry['model']}/{entry['pool']}/{entry['sample']}"
            try:
                row, elapsed = future.result()
                rows[i] = row
                print(f"  [{label}] bsds_f1={row['bsds_f1']:.4f} ({elapsed:.2f}s)", file=sys.stderr, flush=True)
            except subprocess.TimeoutExpired:
                print(f"  [{label}] TIMED OUT (>{args.timeout:.0f}s) -- recording profile-only row", file=sys.stderr, flush=True)
                rows[i] = timeout_fallback_row(entry)
            except Exception as exc:
                print(f"  [{label}] ERROR: {exc}", file=sys.stderr, flush=True)
                rows[i] = timeout_fallback_row(entry)
            n_done += 1
            if n_done % 50 == 0:
                print(f"-- {n_done}/{len(manifest)} done --", file=sys.stderr, flush=True)

    output_csv = args.output_csv or str(Path(args.manifest_csv).parent / "vae_roundtrip_metrics.csv")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows)
    n_timeout = sum(1 for r in rows if r["timed_out"])
    print(f"\n{n_timeout}/{len(rows)} tiles timed out (>{args.timeout:.0f}s), excluded from means above")
    print(f"saved: {output_csv}")


if __name__ == "__main__":
    main()
