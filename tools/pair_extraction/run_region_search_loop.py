"""Run restartable CPU-only variable-aspect region pair search rounds.

The loop orchestrates deterministic pair-search scripts and records state so it
can run unattended as a user service. It intentionally disables CUDA visibility:
this job is meant to run beside GPU training.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "region_search_loop"
DEFAULT_STATE = DEFAULT_OUT_ROOT / "state.json"


HAMLABI_PARENT_PROFILES = [
    {
        "name": "parent_balanced",
        "args": [
            "--region-close", "121",
            "--line-margin", "0.12",
            "--min-match-score", "0.80",
            "--min-f1", "0.18",
            "--max-chamfer", "22.0",
            "--search-px", "120",
            "--search-ratio", "0.18",
            "--match-size", "384",
        ],
    },
    {
        "name": "parent_line_dense",
        "args": [
            "--region-close", "81",
            "--line-margin", "0.10",
            "--min-region-area", "1600",
            "--min-region-size", "140",
            "--max-line-ink", "0.32",
            "--min-match-score", "0.66",
            "--min-f1", "0.14",
            "--max-chamfer", "26.0",
            "--search-px", "150",
            "--search-ratio", "0.22",
            "--match-size", "384",
        ],
    },
    {
        "name": "parent_large_panels",
        "args": [
            "--region-close", "181",
            "--line-margin", "0.16",
            "--min-region-area", "5500",
            "--min-region-size", "220",
            "--max-region-ratio", "0.92",
            "--max-regions-per-page", "16",
            "--min-match-score", "0.62",
            "--min-f1", "0.12",
            "--max-chamfer", "30.0",
            "--search-px", "180",
            "--search-ratio", "0.25",
            "--search-step", "56",
            "--match-size", "448",
        ],
    },
]


HAMLABI_CHILD_PROFILES = [
    {
        "name": "child_balanced",
        "args": [
            "--region-close", "61",
            "--line-margin", "0.14",
            "--min-region-area", "1200",
            "--min-region-size", "120",
            "--min-match-score", "0.45",
            "--min-f1", "0.12",
            "--max-chamfer", "26.0",
            "--search-px", "80",
            "--search-ratio", "0.12",
            "--match-size", "320",
        ],
    },
    {
        "name": "child_fine",
        "args": [
            "--region-close", "41",
            "--line-margin", "0.12",
            "--min-region-area", "700",
            "--min-region-size", "90",
            "--max-children-per-parent", "10",
            "--min-match-score", "0.36",
            "--min-f1", "0.10",
            "--max-chamfer", "30.0",
            "--search-px", "110",
            "--search-ratio", "0.16",
            "--search-step", "36",
            "--match-size", "320",
        ],
    },
]


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state(path):
    if not path.exists():
        return {"version": 1, "runs": {}}
    return json.loads(path.read_text())


def save_state(path, state):
    if state.get("_dry_run"):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def count_csv(path):
    if not path.exists():
        return {"rows": 0, "candidate": 0, "review_low_score": 0}
    counts = {"rows": 0, "candidate": 0, "review_low_score": 0}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            counts["rows"] += 1
            decision = row.get("decision", "")
            if decision:
                counts[decision] = counts.get(decision, 0) + 1
    return counts


def service_env(cpu_threads):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = str(cpu_threads)
    env["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    env["MKL_NUM_THREADS"] = str(cpu_threads)
    env["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
    return env


def run_command(command, log_path, cpu_threads, dry_run=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    if dry_run:
        print(f"[dry-run] {printable}")
        return 0
    with log_path.open("a") as log:
        log.write(f"\n[{now_iso()}] RUN {printable}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=service_env(cpu_threads),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"[{now_iso()}] EXIT {result.returncode}\n")
        return result.returncode


def parent_command(args, profile, round_dir):
    return [
        sys.executable,
        str(REPO_ROOT / "tools" / "pair_extraction" / "match_hamlabi_regions.py"),
        "--zip", args.zip_path,
        "--zip-root", args.zip_root,
        "--csv-out", str(round_dir / "parent_candidates.csv"),
        "--json-out", str(round_dir / "parent_candidates.json"),
        "--qc-out", str(round_dir / "parent_candidates_qc.png"),
        "--qc-count", str(args.qc_count),
        "--output-long-side", str(args.output_long_side),
        *profile["args"],
    ]


def child_command(args, profile, round_dir):
    return [
        sys.executable,
        str(REPO_ROOT / "tools" / "pair_extraction" / "refine_hamlabi_large_regions.py"),
        "--zip", args.zip_path,
        "--zip-root", args.zip_root,
        "--parents", str(round_dir / "parent_candidates.csv"),
        "--csv-out", str(round_dir / f'{profile["name"]}_candidates.csv'),
        "--json-out", str(round_dir / f'{profile["name"]}_candidates.json'),
        "--qc-out", str(round_dir / f'{profile["name"]}_qc.png'),
        "--qc-count", str(args.qc_count),
        "--output-long-side", str(args.output_long_side),
        *profile["args"],
    ]


def run_hamlabi_round(args, state, parent_profile, round_index):
    round_name = f"{round_index:03d}_{parent_profile['name']}"
    run_key = f"hamlabi/{round_name}"
    round_dir = Path(args.out_root) / "hamlabi" / round_name
    log_path = round_dir / "run.log"
    record = state["runs"].setdefault(
        run_key,
        {
            "dataset": "hamlabi",
            "round": round_name,
            "parent_profile": parent_profile["name"],
            "created_at": now_iso(),
            "status": "pending",
        },
    )
    if record.get("status") == "complete" and not args.force:
        print(f"skip complete {run_key}")
        return True

    record["status"] = "running_parent"
    record["updated_at"] = now_iso()
    save_state(Path(args.state), state)

    code = run_command(parent_command(args, parent_profile, round_dir), log_path, args.cpu_threads, args.dry_run)
    if code != 0:
        record["status"] = "failed_parent"
        record["returncode"] = code
        record["updated_at"] = now_iso()
        save_state(Path(args.state), state)
        return False

    parent_csv = round_dir / "parent_candidates.csv"
    record["parent_counts"] = count_csv(parent_csv)
    record["children"] = {}
    for child_profile in HAMLABI_CHILD_PROFILES:
        record["status"] = f"running_{child_profile['name']}"
        record["updated_at"] = now_iso()
        save_state(Path(args.state), state)
        code = run_command(child_command(args, child_profile, round_dir), log_path, args.cpu_threads, args.dry_run)
        child_csv = round_dir / f'{child_profile["name"]}_candidates.csv'
        record["children"][child_profile["name"]] = {
            "returncode": code,
            "counts": count_csv(child_csv),
            "csv": str(child_csv),
            "qc": str(round_dir / f'{child_profile["name"]}_qc.png'),
        }
        if code != 0:
            record["status"] = f"failed_{child_profile['name']}"
            record["updated_at"] = now_iso()
            save_state(Path(args.state), state)
            return False

    record["status"] = "complete"
    record["updated_at"] = now_iso()
    record["parent_csv"] = str(parent_csv)
    record["parent_qc"] = str(round_dir / "parent_candidates_qc.png")
    save_state(Path(args.state), state)
    print(f"complete {run_key}: parent={record['parent_counts']}")
    return True


def run_once(args):
    state = load_state(Path(args.state))
    if args.dry_run:
        state = {"version": 1, "runs": {}, "_dry_run": True}
    completed = 0
    for index, profile in enumerate(HAMLABI_PARENT_PROFILES, 1):
        if args.max_rounds and completed >= args.max_rounds:
            break
        ok = run_hamlabi_round(args, state, profile, index)
        completed += 1
        if not ok and args.stop_on_error:
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hamlabi"], default="hamlabi")
    parser.add_argument("--zip", default="dataset_hamlabi.zip", dest="zip_path")
    parser.add_argument("--zip-root", default="dataset_hamlabi")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--output-long-side", type=int, default=768)
    parser.add_argument("--qc-count", type=int, default=80)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=int, default=1800)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dataset != "hamlabi":
        raise ValueError(f"unsupported dataset: {args.dataset}")

    while True:
        code = run_once(args)
        if code != 0:
            raise SystemExit(code)
        if not args.watch:
            break
        print(f"sleep {args.sleep_seconds}s", flush=True)
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
