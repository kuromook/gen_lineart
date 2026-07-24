"""Finalize completed region-search loop outputs into review-ready pairs."""

import argparse
import ast
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = REPO_ROOT / "results" / "region_search_loop" / "state.json"
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "region_search_loop"
DEFAULT_DATASET_BASE = REPO_ROOT / "dataset" / "regions_hamlabi_loop_auto_review"


def parse_box(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v) for v in ast.literal_eval(value))


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = parse_box(a)
    bx0, by0, bx1, by1 = parse_box(b)
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(1, ax1 - ax0) * max(1, ay1 - ay0)
    area_b = max(1, bx1 - bx0) * max(1, by1 - by0)
    return inter / max(area_a + area_b - inter, 1)


def load_state(path):
    return json.loads(Path(path).read_text())


def service_active(unit):
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip() == "active"


def wait_for_service(unit, poll_seconds):
    while service_active(unit):
        print(f"waiting for {unit}", flush=True)
        time.sleep(poll_seconds)


def loop_complete(state, dataset):
    runs = {key: value for key, value in state.get("runs", {}).items() if key.startswith(f"{dataset}/")}
    if not runs:
        return False
    return all(row.get("status") == "complete" for row in runs.values())


def candidate_csvs(state, dataset):
    paths = []
    for key, run in sorted(state.get("runs", {}).items()):
        if not key.startswith(f"{dataset}/") or run.get("status") != "complete":
            continue
        if run.get("parent_csv"):
            paths.append((key, "parent", run["parent_csv"]))
        for child_name, child in sorted(run.get("children", {}).items()):
            if child.get("csv"):
                paths.append((key, child_name, child["csv"]))
    return paths


def read_candidates(state, dataset):
    rows = []
    for run_key, source_kind, path in candidate_csvs(state, dataset):
        with open(path, newline="") as file:
            for row in csv.DictReader(file):
                row["source_run"] = run_key
                row["source_kind"] = source_kind
                rows.append(row)
    return rows


def value(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except ValueError:
        return default


def candidate_score(row):
    score = value(row, "match_score")
    score += 0.25 if row.get("source_kind", "").startswith("child") else 0.0
    score += min(value(row, "edge_f1"), 1.0)
    score -= max(value(row, "chamfer") - 20.0, 0.0) * 0.02
    return score


def filter_rows(rows, args):
    kept = []
    for row in rows:
        if row.get("decision") != "candidate":
            continue
        if value(row, "match_score") < args.min_match_score:
            continue
        if value(row, "edge_f1") < args.min_f1:
            continue
        if value(row, "chamfer") > args.max_chamfer:
            continue
        if value(row, "rough_std") < args.min_rough_std:
            continue
        kept.append(row)
    kept.sort(key=candidate_score, reverse=True)
    deduped = []
    for row in kept:
        duplicate = False
        for existing in deduped:
            if row["page"] != existing["page"]:
                continue
            if (
                box_iou(row["line_box"], existing["line_box"]) >= args.duplicate_iou
                or box_iou(row["rough_box"], existing["rough_box"]) >= args.duplicate_iou
            ):
                duplicate = True
                break
        if duplicate:
            continue
        row = {
            **row,
            "codex_decision": args.auto_decision,
            "codex_region_type": row.get("feature_tags", ""),
            "codex_notes": "auto-selected by deterministic loop filter; requires human review",
            "auto_score": f"{candidate_score(row):.6f}",
        }
        deduped.append(row)
        if args.target_count and len(deduped) >= args.target_count:
            break
    for rank, row in enumerate(deduped, 1):
        row["rank"] = str(rank)
    return deduped


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_command(command, log):
    printable = " ".join(str(part) for part in command)
    print(printable, flush=True)
    with Path(log).open("a") as file:
        file.write(f"\nRUN {printable}\n")
        file.flush()
        result = subprocess.run(
            [str(part) for part in command],
            cwd=REPO_ROOT,
            stdout=file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        file.write(f"EXIT {result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {printable}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hamlabi"], default="hamlabi")
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--dataset-base", default=str(DEFAULT_DATASET_BASE))
    parser.add_argument("--zip", default=str(REPO_ROOT / "dataset_hamlabi.zip"), dest="zip_path")
    parser.add_argument("--zip-root", default="dataset_hamlabi")
    parser.add_argument("--wait-service", default="")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--min-match-score", type=float, default=0.45)
    parser.add_argument("--min-f1", type=float, default=0.12)
    parser.add_argument("--max-chamfer", type=float, default=30.0)
    parser.add_argument("--min-rough-std", type=float, default=8.0)
    parser.add_argument("--duplicate-iou", type=float, default=0.72)
    parser.add_argument("--auto-decision", default="auto_candidate")
    parser.add_argument("--postalign-shift", type=int, default=12)
    parser.add_argument("--long-side", type=int, default=768)
    parser.add_argument("--log", default=str(REPO_ROOT / "logs" / "region_search_finalize_hamlabi.log"))
    args = parser.parse_args()

    if args.wait_service:
        wait_for_service(args.wait_service, args.poll_seconds)

    state = load_state(args.state)
    if not loop_complete(state, args.dataset):
        raise SystemExit(f"loop state is not complete for {args.dataset}")

    rows = read_candidates(state, args.dataset)
    filtered = filter_rows(rows, args)
    out_root = Path(args.out_root)
    merged_csv = out_root / f"{args.dataset}_auto_candidates.csv"
    merged_json = out_root / f"{args.dataset}_auto_candidates.json"
    write_csv(filtered, merged_csv)
    merged_json.write_text(json.dumps(filtered, indent=2) + "\n")

    dataset_base = Path(args.dataset_base)
    run_command(
        [
            sys.executable,
            REPO_ROOT / "tools" / "pair_extraction" / "materialize_hamlabi_review_regions.py",
            "--zip",
            args.zip_path,
            "--zip-root",
            args.zip_root,
            "--review-csv",
            merged_csv,
            "--decision",
            args.auto_decision,
            "--out-base",
            dataset_base,
            "--manifest-out",
            dataset_base / "manifest.json",
            "--csv-out",
            dataset_base / "manifest.csv",
            "--long-side",
            args.long_side,
        ],
        args.log,
    )

    postalign_base = Path(f"{dataset_base}_postalign{args.postalign_shift}")
    run_command(
        [
            sys.executable,
            REPO_ROOT / "tools" / "pair_extraction" / "post_align_region_manifest.py",
            "--manifest",
            dataset_base / "manifest.csv",
            "--out-base",
            postalign_base,
            "--image-size",
            args.long_side,
            "--fit-mode",
            "square_pad",
            "--max-shift",
            args.postalign_shift,
            "--step",
            "3",
            "--refine-step",
            "1",
            "--min-gain",
            "0.04",
            "--qc-count",
            "120",
        ],
        args.log,
    )

    masked_base = Path(f"{postalign_base}_masked_line_conservative")
    run_command(
        [
            sys.executable,
            REPO_ROOT / "tools" / "pair_extraction" / "build_region_valid_masks.py",
            "--manifest",
            postalign_base / "manifest.csv",
            "--out-base",
            masked_base,
            "--image-size",
            args.long_side,
            "--fit-mode",
            "square_pad",
            "--support-px",
            "10",
            "--window",
            "45",
            "--edge-density",
            "0.060",
            "--expand-ignore",
            "5",
            "--close-ignore",
            "5",
            "--min-valid-ratio",
            "0.80",
            "--qc-count",
            "120",
        ],
        args.log,
    )

    summary = {
        "input_rows": len(rows),
        "auto_candidates": len(filtered),
        "merged_csv": str(merged_csv),
        "materialized_manifest": str(dataset_base / "manifest.csv"),
        "postalign_manifest": str(postalign_base / "manifest.csv"),
        "masked_manifest": str(masked_base / "manifest.csv"),
        "postalign_qc": str(postalign_base / "post_align_qc.png"),
        "mask_qc": str(masked_base / "valid_mask_qc.png"),
    }
    summary_path = out_root / f"{args.dataset}_auto_finalize_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
