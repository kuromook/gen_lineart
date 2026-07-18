"""Build leak-free train lists with exact duplicate tiles removed."""

import argparse
import hashlib
import re
from pathlib import Path


DATASET_DIR = Path("dataset/pairs_480")
DEFAULT_WARM_IN = DATASET_DIR / "valid_train_warm_regions_clean_split.txt"
DEFAULT_WARM_OUT = DATASET_DIR / "valid_train_warm_regions_clean_unique.txt"
DEFAULT_BASE_OUT = DATASET_DIR / "valid_train_base_clean_unique.txt"
DEFAULT_REMOVED = DATASET_DIR / "clean_unique_removed.txt"


def read_names(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_names(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\n" for name in names))


def canonical_tile(name):
    base = Path(name).stem
    match = re.fullmatch(r"orig_(\d{3})-(\d+)", base)
    if match:
        page, tile = match.groups()
        return f"lineart_{page}_{int(tile):03d}"
    return base


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_key(name):
    rough = DATASET_DIR / "train/rough" / name
    line = DATASET_DIR / "train/line" / name
    return canonical_tile(name), file_hash(rough), file_hash(line)


def prefer_score(name):
    base = Path(name).stem
    if base.startswith("lineart_"):
        return 0
    if base.startswith("housei_"):
        return 1
    if base.startswith("ako5r_"):
        return 2
    if base.startswith("orig_"):
        return 3
    return 4


def dedupe(names):
    grouped = {}
    for name in names:
        grouped.setdefault(pair_key(name), []).append(name)

    kept = []
    removed = []
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda name: (prefer_score(name), name))
        kept.append(ordered[0])
        for name in ordered[1:]:
            removed.append((name, ordered[0], key[0]))

    input_order = {name: index for index, name in enumerate(names)}
    kept.sort(key=lambda name: input_order[name])
    return kept, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-in", default=str(DEFAULT_WARM_IN))
    parser.add_argument("--warm-out", default=str(DEFAULT_WARM_OUT))
    parser.add_argument("--base-out", default=str(DEFAULT_BASE_OUT))
    parser.add_argument("--removed-out", default=str(DEFAULT_REMOVED))
    args = parser.parse_args()

    warm_names = read_names(Path(args.warm_in))
    warm_unique, removed = dedupe(warm_names)
    base_unique = [name for name in warm_unique if not name.startswith("ako5r_")]

    write_names(Path(args.warm_out), warm_unique)
    write_names(Path(args.base_out), base_unique)
    removed_lines = ["removed,kept,canonical_tile\n"]
    removed_lines.extend(f"{removed_name},{kept_name},{tile}\n" for removed_name, kept_name, tile in removed)
    Path(args.removed_out).write_text("".join(removed_lines))

    print(f"warm_in:   {args.warm_in} ({len(warm_names)})")
    print(f"warm_out:  {args.warm_out} ({len(warm_unique)})")
    print(f"base_out:  {args.base_out} ({len(base_unique)})")
    print(f"removed:   {len(removed)}")
    print(f"removed_out: {args.removed_out}")


if __name__ == "__main__":
    main()
