"""Audit train/test leakage for paired 480px lineart lists."""

import argparse
import hashlib
import re
from pathlib import Path


DATASET_DIR = Path("dataset/pairs_480")
DEFAULT_TRAIN_LIST = DATASET_DIR / "valid_train_warm_regions.txt"
DEFAULT_TEST_LIST = DATASET_DIR / "valid_test.txt"


def read_names(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def stem(name):
    return Path(name).stem


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
    if len(parts) >= 2 and parts[0] in {"housei", "lineart"}:
        return "_".join(parts[:2])
    if len(parts) >= 2 and parts[0] in {"ako5", "ako5r", "kurip", "kuripm", "kuripr", "kurips960"}:
        return "_".join(parts[:2])
    return parts[0]


def file_hash(path):
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_path(split, kind, name):
    return DATASET_DIR / split / kind / name


def hash_index(names, split, kind):
    out = {}
    missing = []
    for name in names:
        path = split_path(split, kind, name)
        digest = file_hash(path)
        if digest is None:
            missing.append(name)
            continue
        out.setdefault(digest, []).append(name)
    return out, missing


def print_matches(title, rows, limit):
    print(title)
    print(f"  count: {len(rows)}")
    for train_name, test_name in rows[:limit]:
        print(f"  {train_name} == {test_name}")
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", default=str(DEFAULT_TRAIN_LIST))
    parser.add_argument("--test-list", default=str(DEFAULT_TEST_LIST))
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    train_names = read_names(Path(args.train_list))
    test_names = read_names(Path(args.test_list))

    test_by_tile = {}
    for name in test_names:
        test_by_tile.setdefault(canonical_tile(name), []).append(name)
    canonical_matches = [
        (name, test_name)
        for name in train_names
        for test_name in test_by_tile.get(canonical_tile(name), [])
    ]

    train_pages = {}
    test_pages = {}
    for name in train_names:
        train_pages.setdefault(canonical_page(name), 0)
        train_pages[canonical_page(name)] += 1
    for name in test_names:
        test_pages.setdefault(canonical_page(name), 0)
        test_pages[canonical_page(name)] += 1
    page_overlaps = sorted(set(train_pages) & set(test_pages))

    print(f"train_list: {args.train_list} ({len(train_names)})")
    print(f"test_list:  {args.test_list} ({len(test_names)})")
    print_matches("canonical tile overlaps", canonical_matches, args.limit)
    print("canonical page overlaps")
    print(f"  count: {len(page_overlaps)}")
    for page in page_overlaps[: args.limit]:
        print(f"  {page}: train={train_pages[page]} test={test_pages[page]}")
    if len(page_overlaps) > args.limit:
        print(f"  ... {len(page_overlaps) - args.limit} more")

    for kind in ("rough", "line"):
        train_hashes, train_missing = hash_index(train_names, "train", kind)
        test_hashes, test_missing = hash_index(test_names, "test", kind)
        exact_matches = []
        for digest, names in train_hashes.items():
            for test_name in test_hashes.get(digest, []):
                for train_name in names:
                    exact_matches.append((train_name, test_name))
        print_matches(f"exact {kind} file hash overlaps", exact_matches, args.limit)
        if train_missing:
            print(f"missing train {kind}: {len(train_missing)}")
        if test_missing:
            print(f"missing test {kind}: {len(test_missing)}")


if __name__ == "__main__":
    main()
