"""Track D, hypothesis 2: does training loss correlate with output quality?

Track B observed loss falling monotonically while every quality axis got
worse (`doc/work_log.md`, `doc/initial_notice.md` hypothesis 2). Whether
that is a one-off or a systematic decoupling between this project's loss
function and `gt_bsds_f1` was never measured, because no track had ever
saved intermediate model weights during a run -- every checkpoint on disk
is a run's `final` state only (verified 2026-09-14, see
`doc/work_log.md`). This script is the scoring half of a diagnostic
instrumented replication: `train_controlnet_consistency.py --eval-snapshot-
steps N` (a pre-existing but previously-unused flag) saves a LoRA snapshot
every N steps; a driver script runs `infer_controlnet.py` against each
snapshot on a fixed 192-tile holdout at a fixed controlnet_conditioning_
scale, and this script scores each snapshot's outputs against GT, joins
them with the training log's loss at that step, and reports the
correlation.

Never uses `evaluate_fixed_outputs.py`'s `--split auto` path-guessing for
the holdout GT: that heuristic (`"train" if name.startswith("housei") else
"test"`) is wrong for most of `holdout_lineart_family.txt` (found
2026-09-14 building the VAE-roundtrip tooling -- 168/192 lineart_family
names are actually under train/line, not test/line, and the heuristic's
own default sample list happens to be drawn entirely from the 24 that
*are* under test/, which is why the bug was never caught). This script
instead points directly at Track A's own self-contained holdout copies
(`data/holdout_lineart_family_gt_line/`), sidestepping the ambiguity
entirely rather than reimplementing split resolution.

Each tile is scored in its own subprocess with a hard timeout, same
reasoning and same worker script as `vae_roundtrip_score.py` (`scipy`'s
`maximum_bipartite_matching`, behind `gt_bsds_f1`, can pathologically
stall on specific edge-point configurations -- see that script's
docstring). Reuses `vae_roundtrip_score_one.py` directly since the scoring
step is identical (GT path + candidate image path in, JSON metrics out);
model-generated output is no different from a VAE roundtrip's output for
scoring purposes.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCORE_ONE = str(Path(__file__).resolve().parent / "vae_roundtrip_score_one.py")
STEP_LOG_RE = re.compile(
    r"^step (\d+)/(\d+) loss=([\d.]+) eps=([\d.]+) consistency=([\d.]+) elapsed=(\d+)s"
)


def parse_training_log(path):
    """Returns {step: {"loss":..,"eps":..,"consistency":..}} for every
    logged step of the main run. The launcher's smoke test writes its own
    `step k/6` lines into the same log; only lines whose `/max_train_steps`
    denominator is the largest seen are kept, so those never leak into the
    first snapshot's loss span."""
    parsed = []
    with open(path) as f:
        for line in f:
            m = STEP_LOG_RE.match(line)
            if m:
                parsed.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
    if not parsed:
        return {}
    main_total = max(total for _, total, _, _, _ in parsed)
    return {
        step: {"loss": loss, "eps": eps, "consistency": consistency}
        for step, total, loss, eps, consistency in parsed
        if total == main_total
    }


def loss_between(by_step, lo, hi):
    """Mean of every logged loss with lo < step <= hi. Each log line is one
    batch at a randomly sampled timestep, so a handful of lines is mostly
    timestep noise -- prefer wide spans."""
    matches = [v for step, v in by_step.items() if lo < step <= hi]
    if not matches:
        return None
    return {
        "loss": sum(v["loss"] for v in matches) / len(matches),
        "eps": sum(v["eps"] for v in matches) / len(matches),
        "consistency": sum(v["consistency"] for v in matches) / len(matches),
        "n_log_lines": len(matches),
    }


def loss_near_step(by_step, target_step, window):
    return loss_between(by_step, target_step - window, target_step)


def rank_corr(x, y):
    import numpy as np
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def discover_snapshots(results_root):
    """results_root/{snapshot_label}/{base}_out.png per snapshot, e.g.
    results_root/step_1000/, results_root/final/. Snapshot step is parsed
    from the label ("step_1000" -> 1000; "final" -> None, sorted last)."""
    snapshots = []
    for d in sorted(Path(results_root).iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"step_(\d+)$", d.name)
        step = int(m.group(1)) if m else None
        snapshots.append((d.name, step, d))
    snapshots.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))
    return snapshots


def score_entry(name, gt_dir, recon_dir, python_bin, timeout_s):
    gt_path = Path(gt_dir) / name
    recon_path = Path(recon_dir) / f"{name[:-4]}_out.png"
    t0 = time.time()
    proc = subprocess.run(
        [python_bin, SCORE_ONE, "--gt", str(gt_path), "--recon", str(recon_path)],
        capture_output=True, text=True, timeout=None if timeout_s <= 0 else timeout_s,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed: {proc.stderr[-2000:]}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    return result, elapsed


def score_snapshot(names, gt_dir, recon_dir, workers, timeout_s, python_bin):
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_entry, name, gt_dir, recon_dir, python_bin, timeout_s): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result, elapsed = future.result()
                rows.append({"sample": name, "timed_out": False, **result})
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT: {name}", file=sys.stderr, flush=True)
                rows.append({"sample": name, "timed_out": True, "bsds_f1": None})
            except Exception as exc:
                print(f"    ERROR: {name}: {exc}", file=sys.stderr, flush=True)
                rows.append({"sample": name, "timed_out": True, "bsds_f1": None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", required=True, help="dir containing one subdir per snapshot")
    parser.add_argument("--gt-dir", required=True, help="dir of GT images, same basenames as sample list")
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--loss-window", type=int, default=200, help="average loss over the N steps before each snapshot")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--per-tile-csv-dir", default=None, help="optional: dump every tile's row per snapshot here")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    with open(args.sample_list) as f:
        names = [line.strip() for line in f if line.strip()]

    by_step = parse_training_log(args.training_log)
    print(f"parsed {len(by_step)} logged training steps from {args.training_log}", file=sys.stderr)

    snapshots = discover_snapshots(args.results_root)
    print(f"found {len(snapshots)} snapshots under {args.results_root}: {[s[0] for s in snapshots]}", file=sys.stderr)

    max_step = max((s for _, s, _ in snapshots if s is not None), default=max(by_step))
    numbered = sorted(s for _, s, _ in snapshots if s is not None)
    summary_rows = []
    for label, step, snap_dir in snapshots:
        effective_step = step if step is not None else max_step
        loss_info = loss_near_step(by_step, effective_step, args.loss_window) or {}
        prev_step = max((s for s in numbered if s < effective_step), default=0)
        span_info = loss_between(by_step, prev_step, effective_step) or {}
        print(f"[{label}] scoring {len(names)} tiles against {snap_dir} ...", file=sys.stderr, flush=True)
        rows = score_snapshot(names, args.gt_dir, snap_dir, args.workers, args.timeout, args.python_bin)
        scored = [r for r in rows if not r["timed_out"]]
        n_timeout = len(rows) - len(scored)
        mean_f1 = sum(r["bsds_f1"] for r in scored) / len(scored) if scored else float("nan")
        mean_near_white = sum(r["recon_near_white_frac"] for r in scored) / len(scored) if scored else float("nan")
        mean_fill = sum(r["recon_fill_ratio"] for r in scored) / len(scored) if scored else float("nan")
        summary_rows.append({
            "label": label, "step": step if step is not None else "", "n": len(rows), "n_timeout": n_timeout,
            "gt_bsds_f1": mean_f1, "recon_near_white_frac": mean_near_white, "recon_fill_ratio": mean_fill,
            "loss": loss_info.get("loss"), "eps_loss": loss_info.get("eps"), "consistency_loss": loss_info.get("consistency"),
            "n_log_lines": loss_info.get("n_log_lines"),
            "span_from_step": prev_step, "loss_span": span_info.get("loss"), "eps_loss_span": span_info.get("eps"),
            "consistency_loss_span": span_info.get("consistency"), "n_log_lines_span": span_info.get("n_log_lines"),
        })
        print(
            f"[{label}] gt_bsds_f1={mean_f1:.4f} near_white={mean_near_white:.4f} "
            f"loss={loss_info.get('loss')} eps={loss_info.get('eps')} consistency={loss_info.get('consistency')} "
            f"(n_timeout={n_timeout})", file=sys.stderr, flush=True,
        )
        if args.per_tile_csv_dir:
            Path(args.per_tile_csv_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(args.per_tile_csv_dir) / f"{label}.csv", "w", newline="") as f:
                fields = sorted({k for r in rows for k in r})
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    output_csv = args.output_csv or str(Path(args.results_root) / "loss_quality_summary.csv")
    with open(output_csv, "w", newline="") as f:
        fields = [
            "label", "step", "n", "n_timeout", "gt_bsds_f1", "recon_near_white_frac", "recon_fill_ratio",
            "loss", "eps_loss", "consistency_loss", "n_log_lines",
            "span_from_step", "loss_span", "eps_loss_span", "consistency_loss_span", "n_log_lines_span",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else "nan"

    print(
        "\n" + "label".ljust(12) + "step".rjust(8) + "gt_bsds_f1".rjust(12) + "near_white".rjust(12)
        + f"loss@{args.loss_window}".rjust(11) + "loss_span".rjust(11) + "eps_span".rjust(10) + "n_span".rjust(8)
    )
    for row in summary_rows:
        print(
            f"{row['label']:<12}{str(row['step']):>8}{fmt(row['gt_bsds_f1']):>12}{fmt(row['recon_near_white_frac']):>12}"
            f"{fmt(row['loss']):>11}{fmt(row['loss_span']):>11}{fmt(row['eps_loss_span']):>10}{str(row['n_log_lines_span']):>8}"
        )

    # train_controlnet_consistency.py saves an eval snapshot at max_train_steps
    # *and* `final` from the same weights, so `final` would enter the
    # correlation as a duplicate point; keep it in the table as a
    # run-to-run inference-noise check, but not in the fit.
    numbered_steps = {r["step"] for r in summary_rows if r["step"] != ""}
    valid = [
        r for r in summary_rows
        if r["loss"] is not None and r["gt_bsds_f1"] == r["gt_bsds_f1"]
        and not (r["label"] == "final" and max_step in numbered_steps)
    ]
    if len(valid) >= 3:
        import numpy as np
        f1 = np.array([r["gt_bsds_f1"] for r in valid])
        print(f"\nn={len(valid)} snapshots with both loss and gt_bsds_f1 (final excluded when it duplicates the last step snapshot)")
        for name, key in (
            (f"loss@{args.loss_window}", "loss"), (f"eps@{args.loss_window}", "eps_loss"),
            ("loss_span", "loss_span"), ("eps_span", "eps_loss_span"), ("consistency_span", "consistency_loss_span"),
        ):
            x = np.array([r[key] for r in valid], dtype=float)
            print(f"  {name:<18} pearson={np.corrcoef(x, f1)[0, 1]:+.3f}  spearman={rank_corr(x, f1):+.3f}")
        steps_arr = np.array([r["step"] if r["step"] != "" else max_step for r in valid], dtype=float)
        print(f"  {'training step':<18} pearson={np.corrcoef(steps_arr, f1)[0, 1]:+.3f}  spearman={rank_corr(steps_arr, f1):+.3f}")
        print("  (loss@N averages only N/50 single-batch log lines -- read the *_span rows first)")
    else:
        print(f"\nonly {len(valid)} snapshots with valid loss+f1 -- too few for a correlation")

    print(f"\nsaved: {output_csv}")


if __name__ == "__main__":
    main()
