"""Build leakage-clean train/eval lists for shape experiments."""

import argparse
import re
from pathlib import Path


DATASET_DIR = Path("dataset/pairs_480")
DEFAULT_TRAIN_IN = DATASET_DIR / "valid_train_warm_regions.txt"
DEFAULT_TEST_IN = DATASET_DIR / "valid_test.txt"
DEFAULT_TRAIN_OUT = DATASET_DIR / "valid_train_warm_regions_clean_split.txt"
DEFAULT_EVAL_OUT = DATASET_DIR / "eval_fixed_clean.txt"

DEFAULT_EVAL_NAMES = [
    "housei_004_02_02.jpg",
    "housei_004_04_04.jpg",
    "housei_004_06_06.jpg",
    "housei_009_02_04.jpg",
    "housei_009_04_04.jpg",
    "lineart_004_002.jpg",
    "lineart_004_006.jpg",
    "lineart_004_010.jpg",
]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-in", default=str(DEFAULT_TRAIN_IN))
    parser.add_argument("--test-in", default=str(DEFAULT_TEST_IN))
    parser.add_argument("--train-out", default=str(DEFAULT_TRAIN_OUT))
    parser.add_argument("--eval-out", default=str(DEFAULT_EVAL_OUT))
    args = parser.parse_args()

    train_names = read_names(Path(args.train_in))
    test_names = read_names(Path(args.test_in))
    test_tiles = {canonical_tile(name) for name in test_names}

    kept = []
    removed = []
    for name in train_names:
        if canonical_tile(name) in test_tiles:
            removed.append(name)
        else:
            kept.append(name)

    test_set = set(test_names)
    eval_names = [name for name in DEFAULT_EVAL_NAMES if name in test_set]
    missing_eval = [name for name in DEFAULT_EVAL_NAMES if name not in test_set]
    if missing_eval:
        raise SystemExit(f"eval samples missing from test list: {missing_eval}")

    write_names(Path(args.train_out), kept)
    write_names(Path(args.eval_out), eval_names)

    print(f"train_in:  {args.train_in} ({len(train_names)})")
    print(f"test_in:   {args.test_in} ({len(test_names)})")
    print(f"train_out: {args.train_out} ({len(kept)})")
    print(f"removed:   {len(removed)}")
    for name in removed[:40]:
        print(f"  removed {name}")
    if len(removed) > 40:
        print(f"  ... {len(removed) - 40} more")
    print(f"eval_out:  {args.eval_out} ({len(eval_names)})")


if __name__ == "__main__":
    main()
