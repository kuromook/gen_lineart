"""Build a leakage-clean list that reintroduces limited train duplicates."""

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path


DATASET_DIR = Path("dataset/pairs_480")


def read_names(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def write_names(path, names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\n" for name in names))


def normalize_name(name):
    return name if name.endswith(".jpg") else f"{name}.jpg"


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


def pair_hash(name):
    name = normalize_name(name)
    rough = DATASET_DIR / "train/rough" / name
    line = DATASET_DIR / "train/line" / name
    if not rough.exists() or not line.exists():
        return None
    return file_hash(rough), file_hash(line)


def source_priority(name):
    base = Path(name).stem
    if base.startswith("lineart_"):
        return 0
    if base.startswith("housei_"):
        return 1
    if base.startswith("orig_"):
        return 2
    if base.startswith("ako5r_"):
        return 3
    return 4


def build(seed_names, pool_names, target_rows, max_per_canonical, allow_exact_duplicates):
    selected = []
    selected_set = set()
    exact_hashes = set()
    canonical_counts = Counter()
    skipped = Counter()

    def try_add(name, reason):
        name = normalize_name(name)
        if name in selected_set:
            skipped["already_selected"] += 1
            return False
        canonical = canonical_tile(name)
        if canonical_counts[canonical] >= max_per_canonical:
            skipped["canonical_limit"] += 1
            return False
        hashes = pair_hash(name)
        if hashes is None:
            skipped["missing_pair"] += 1
            return False
        if hashes in exact_hashes and not allow_exact_duplicates:
            skipped["exact_duplicate"] += 1
            return False
        selected.append(name)
        selected_set.add(name)
        exact_hashes.add(hashes)
        canonical_counts[canonical] += 1
        return True

    for name in seed_names:
        try_add(name, "seed")

    pool_order = sorted(
        (normalize_name(name) for name in pool_names),
        key=lambda name: (canonical_counts[canonical_tile(name)] == 0, source_priority(name), name),
    )
    for name in pool_order:
        if len(selected) >= target_rows:
            break
        try_add(name, "pool")

    return selected, skipped, canonical_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-list", default="dataset/pairs_480/valid_train_base_clean_unique.txt")
    parser.add_argument("--pool-list", default="dataset/pairs_480/valid_train_std15_clean_split_moredupes.txt")
    parser.add_argument("--out", default="dataset/pairs_480/valid_train_milddup800_clean.txt")
    parser.add_argument("--target-rows", type=int, default=800)
    parser.add_argument("--max-per-canonical", type=int, default=2)
    parser.add_argument("--allow-exact-duplicates", action="store_true")
    args = parser.parse_args()

    seed_names = read_names(args.seed_list)
    pool_names = read_names(args.pool_list)
    selected, skipped, canonical_counts = build(
        seed_names, pool_names, args.target_rows, args.max_per_canonical,
        args.allow_exact_duplicates,
    )
    write_names(args.out, selected)
    duplicate_canonicals = sum(1 for count in canonical_counts.values() if count > 1)

    print(f"seed_list: {args.seed_list} ({len(seed_names)})")
    print(f"pool_list: {args.pool_list} ({len(pool_names)})")
    print(f"out:       {args.out} ({len(selected)})")
    print(f"target:    {args.target_rows}")
    print(f"max_per_canonical: {args.max_per_canonical}")
    print(f"allow_exact_duplicates: {args.allow_exact_duplicates}")
    print(f"canonicals: {len(canonical_counts)}")
    print(f"duplicate_canonicals: {duplicate_canonicals}")
    for key in sorted(skipped):
        print(f"skipped_{key}: {skipped[key]}")


if __name__ == "__main__":
    main()
