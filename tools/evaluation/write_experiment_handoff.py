"""Write a concise Markdown handoff for an experiment run."""

import argparse
import csv
import re
from pathlib import Path


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return sum(values) / len(values) if values else None


def latest_epoch(log_path):
    if not log_path.exists():
        return None
    pattern = re.compile(r"Epoch\s+(\d+)/(\d+): loss=([0-9.]+)\s+lr=([0-9.e+-]+)")
    latest = None
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            latest = {
                "epoch": int(match.group(1)),
                "epochs": int(match.group(2)),
                "loss": float(match.group(3)),
                "lr": match.group(4),
            }
    return latest


def finding_counts(path):
    rows = read_csv(path)
    counts = {}
    for row in rows:
        key = row.get("finding_type", "")
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def model_means(metrics_rows):
    by_model = {}
    for row in metrics_rows:
        by_model.setdefault(row["model"], []).append(row)
    out = []
    for model, rows in sorted(by_model.items()):
        out.append({
            "model": model,
            "f1_2px": mean(rows, "f1_2px"),
            "chamfer_px": mean(rows, "chamfer_px"),
            "ink_ratio": mean(rows, "ink_ratio"),
            "precision_2px": mean(rows, "precision_2px"),
            "recall_2px": mean(rows, "recall_2px"),
        })
    return out


def fmt(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--audit-findings", required=True)
    parser.add_argument("--montage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    metrics_path = Path(args.metrics_csv)
    findings_path = Path(args.audit_findings)
    latest = latest_epoch(log_path)
    metrics_rows = read_csv(metrics_path)
    counts = finding_counts(findings_path)
    train_count = len([line for line in Path(args.train_list).read_text().splitlines() if line.strip()])

    lines = [
        f"# Experiment Handoff: {args.experiment}",
        "",
        "## Inputs",
        "",
        f"- train list: `{args.train_list}` ({train_count})",
        f"- checkpoint dir: `{args.checkpoint_dir}`",
        f"- log: `{args.log}`",
        "",
        "## Training Status",
        "",
    ]
    if latest:
        lines.append(
            f"- latest epoch in log: {latest['epoch']}/{latest['epochs']}, "
            f"loss={latest['loss']:.4f}, lr={latest['lr']}"
        )
    else:
        lines.append("- latest epoch in log: unavailable")

    lines.extend([
        "",
        "## Integrity Audit",
        "",
        f"- summary: `{args.audit_summary}`",
        f"- findings: `{args.audit_findings}`",
    ])
    if counts:
        for key, count in sorted(counts.items()):
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- findings: 0")

    lines.extend([
        "",
        "## Metrics",
        "",
        f"- csv: `{args.metrics_csv}`",
    ])
    if metrics_rows:
        lines.extend([
            "",
            "| model | F1@2px | chamfer | ink_ratio | precision | recall |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in model_means(metrics_rows):
            lines.append(
                f"| `{row['model']}` | {fmt(row['f1_2px'])} | {fmt(row['chamfer_px'], 3)} | "
                f"{fmt(row['ink_ratio'], 3)} | {fmt(row['precision_2px'])} | {fmt(row['recall_2px'])} |"
            )
    else:
        lines.append("- metrics unavailable")

    lines.extend([
        "",
        "## Review Artifacts",
        "",
        f"- montage: `{args.montage}`",
        "",
        "## Next Review Checklist",
        "",
        "1. Inspect montage for blank/collapse, line breakage, black thickening, and added non-rough lines.",
        "2. Compare clean baseline behavior against leaky shape1 only as a failure reference, not as a target score.",
        "3. If output quality is poor, choose between loss changes, list composition changes, or raw tile regeneration.",
        "4. Before any next training list is used, run `tools/evaluation/audit_pair_dataset_integrity.py`.",
        "",
    ])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
