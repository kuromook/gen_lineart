"""Audit paired-tile list integrity across train and evaluation splits."""

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path


DATASET_DIR = Path("dataset/pairs_480")
DEFAULT_OUTPUT_SUMMARY = Path("results/pair_dataset_integrity_summary.csv")
DEFAULT_OUTPUT_FINDINGS = Path("results/pair_dataset_integrity_findings.csv")
SOURCE_PREFIXES = (
    "ako5r",
    "ako5",
    "kurips960",
    "kuripr",
    "kuripm",
    "kurip",
    "housei",
    "lineart",
    "orig",
)


def read_names(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def normalize_name(name):
    return name if name.endswith(".jpg") else f"{name}.jpg"


def stem(name):
    return Path(name).stem


def source_of(name):
    base = stem(name)
    for prefix in SOURCE_PREFIXES:
        if base == prefix or base.startswith(f"{prefix}_") or base.startswith(f"{prefix}-"):
            return prefix
    return base.split("_")[0]


def canonical_tile(name):
    base = stem(name)
    match = re.fullmatch(r"orig_(\d{3})-(\d+)", base)
    if match:
        page, tile = match.groups()
        return f"lineart_{page}_{int(tile):03d}"
    return base


def canonical_page(name):
    base = stem(name)
    match = re.fullmatch(r"orig_(\d{3})-\d+", base)
    if match:
        return f"lineart_{match.group(1)}"
    parts = base.split("_")
    if len(parts) >= 2 and parts[0] in {
        "housei",
        "lineart",
        "ako5",
        "ako5r",
        "kurip",
        "kuripm",
        "kuripr",
        "kurips960",
    }:
        return "_".join(parts[:2])
    return parts[0]


def split_for_role(role):
    if role == "train":
        return "train"
    if role == "eval":
        return "test"
    raise ValueError(role)


def paired_path(split, kind, name, source, line_dir_overrides):
    if split == "train" and kind == "line" and source in line_dir_overrides:
        return line_dir_overrides[source] / normalize_name(name)
    return DATASET_DIR / split / kind / normalize_name(name)


def file_hash(path):
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_lists(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(DATASET_DIR.glob(pattern))
    return sorted({path for path in paths if path.is_file()})


def load_list(path, role, line_dir_overrides):
    split = split_for_role(role)
    names = read_names(path)
    rows = []
    for index, name in enumerate(names):
        row = {
            "list_path": str(path),
            "role": role,
            "split": split,
            "index": index,
            "name": normalize_name(name),
            "source": source_of(name),
            "canonical_tile": canonical_tile(name),
            "canonical_page": canonical_page(name),
        }
        for kind in ("rough", "line"):
            path_for_kind = paired_path(split, kind, name, row["source"], line_dir_overrides)
            row[f"{kind}_path"] = str(path_for_kind)
            row[f"{kind}_exists"] = path_for_kind.exists()
            row[f"{kind}_hash"] = file_hash(path_for_kind)
        rows.append(row)
    return rows


def parse_line_dir_overrides(values):
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--line-dir-override must be SOURCE=DIR, got: {value}")
        source, directory = value.split("=", 1)
        overrides[source] = Path(directory)
    return overrides


def index_by(rows, field):
    out = defaultdict(list)
    for row in rows:
        value = row[field]
        if value:
            out[value].append(row)
    return out


def add_finding(findings, finding_type, lhs, rhs=None, detail=""):
    row = {
        "finding_type": finding_type,
        "lhs_list": lhs["list_path"],
        "lhs_role": lhs["role"],
        "lhs_name": lhs["name"],
        "lhs_source": lhs["source"],
        "lhs_page": lhs["canonical_page"],
        "rhs_list": "",
        "rhs_role": "",
        "rhs_name": "",
        "rhs_source": "",
        "rhs_page": "",
        "detail": detail,
    }
    if rhs is not None:
        row.update({
            "rhs_list": rhs["list_path"],
            "rhs_role": rhs["role"],
            "rhs_name": rhs["name"],
            "rhs_source": rhs["source"],
            "rhs_page": rhs["canonical_page"],
        })
    findings.append(row)


def audit_missing(rows, findings):
    for row in rows:
        for kind in ("rough", "line"):
            if not row[f"{kind}_exists"]:
                add_finding(findings, f"missing_{kind}", row, detail=row[f"{kind}_path"])


def audit_within_list(rows, findings):
    for field, finding_type in (
        ("name", "duplicate_name_within_list"),
        ("canonical_tile", "duplicate_canonical_tile_within_list"),
        ("rough_hash", "duplicate_exact_rough_hash_within_list"),
        ("line_hash", "duplicate_exact_line_hash_within_list"),
    ):
        by_list = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if not row[field]:
                continue
            by_list[row["list_path"]][row[field]].append(row)
        for values in by_list.values():
            for dupes in values.values():
                if len(dupes) < 2:
                    continue
                first = dupes[0]
                for other in dupes[1:]:
                    add_finding(findings, finding_type, first, other)


def audit_cross(train_rows, eval_rows, findings):
    eval_by_tile = index_by(eval_rows, "canonical_tile")
    eval_by_page = index_by(eval_rows, "canonical_page")
    eval_by_rough_hash = index_by(eval_rows, "rough_hash")
    eval_by_line_hash = index_by(eval_rows, "line_hash")

    for train in train_rows:
        for eval_row in eval_by_tile.get(train["canonical_tile"], []):
            add_finding(findings, "train_eval_canonical_tile_overlap", train, eval_row)
        for eval_row in eval_by_page.get(train["canonical_page"], []):
            add_finding(findings, "train_eval_canonical_page_overlap", train, eval_row)
        for eval_row in eval_by_rough_hash.get(train["rough_hash"], []):
            add_finding(findings, "train_eval_exact_rough_hash_overlap", train, eval_row)
        for eval_row in eval_by_line_hash.get(train["line_hash"], []):
            add_finding(findings, "train_eval_exact_line_hash_overlap", train, eval_row)


def write_findings(findings, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "finding_type",
        "lhs_list",
        "lhs_role",
        "lhs_name",
        "lhs_source",
        "lhs_page",
        "rhs_list",
        "rhs_role",
        "rhs_name",
        "rhs_source",
        "rhs_page",
        "detail",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(findings)


def write_summary(all_rows, findings, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    by_list = defaultdict(list)
    for row in all_rows:
        by_list[row["list_path"]].append(row)

    finding_counts = Counter((row["lhs_list"], row["finding_type"]) for row in findings)
    fields = [
        "list_path",
        "role",
        "rows",
        "sources",
        "missing_rough",
        "missing_line",
        "duplicate_name_within_list",
        "duplicate_canonical_tile_within_list",
        "duplicate_exact_rough_hash_within_list",
        "duplicate_exact_line_hash_within_list",
        "train_eval_canonical_tile_overlap",
        "train_eval_canonical_page_overlap",
        "train_eval_exact_rough_hash_overlap",
        "train_eval_exact_line_hash_overlap",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for list_path, rows in sorted(by_list.items()):
            source_counts = Counter(row["source"] for row in rows)
            summary = {
                "list_path": list_path,
                "role": rows[0]["role"],
                "rows": len(rows),
                "sources": ";".join(f"{key}:{source_counts[key]}" for key in sorted(source_counts)),
                "missing_rough": sum(not row["rough_exists"] for row in rows),
                "missing_line": sum(not row["line_exists"] for row in rows),
            }
            for field in fields[6:]:
                summary[field] = finding_counts[(list_path, field)]
            writer.writerow(summary)


def print_summary(train_rows, eval_rows, findings, summary_path, findings_path):
    counts = Counter(row["finding_type"] for row in findings)
    print(f"train rows: {len(train_rows)}")
    print(f"eval rows:  {len(eval_rows)}")
    print(f"findings:   {len(findings)}")
    for finding_type, count in sorted(counts.items()):
        print(f"  {finding_type}: {count}")
    print(f"saved summary:  {summary_path}")
    print(f"saved findings: {findings_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-lists", nargs="*", default=None)
    parser.add_argument("--eval-lists", nargs="*", default=None)
    parser.add_argument(
        "--line-dir-override",
        action="append",
        default=[],
        help="Use SOURCE=DIR for train line files stored outside train/line.",
    )
    parser.add_argument("--output-summary", default=str(DEFAULT_OUTPUT_SUMMARY))
    parser.add_argument("--output-findings", default=str(DEFAULT_OUTPUT_FINDINGS))
    args = parser.parse_args()

    train_lists = [Path(path) for path in args.train_lists] if args.train_lists else default_lists(["valid_train*.txt"])
    eval_lists = [Path(path) for path in args.eval_lists] if args.eval_lists else default_lists(["valid_test.txt", "eval_fixed*.txt"])
    line_dir_overrides = parse_line_dir_overrides(args.line_dir_override)

    train_rows = [row for path in train_lists for row in load_list(path, "train", line_dir_overrides)]
    eval_rows = [row for path in eval_lists for row in load_list(path, "eval", line_dir_overrides)]
    all_rows = train_rows + eval_rows

    findings = []
    audit_missing(all_rows, findings)
    audit_within_list(all_rows, findings)
    audit_cross(train_rows, eval_rows, findings)

    summary_path = Path(args.output_summary)
    findings_path = Path(args.output_findings)
    write_summary(all_rows, findings, summary_path)
    write_findings(findings, findings_path)
    print_summary(train_rows, eval_rows, findings, summary_path, findings_path)


if __name__ == "__main__":
    main()
