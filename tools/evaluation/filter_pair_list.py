"""Write a pair list with named exclusions removed."""

import argparse
from pathlib import Path


def read_names(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def write_names(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\n" for name in names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", required=True)
    parser.add_argument("--exclude-list", required=True)
    parser.add_argument("--output-list", required=True)
    parser.add_argument("--removed-out")
    args = parser.parse_args()

    names = read_names(args.input_list)
    exclude = set(read_names(args.exclude_list))
    kept = [name for name in names if name not in exclude]
    removed = [name for name in names if name in exclude]

    write_names(Path(args.output_list), kept)
    if args.removed_out:
        write_names(Path(args.removed_out), removed)

    print(f"input:   {args.input_list} ({len(names)})")
    print(f"exclude: {args.exclude_list} ({len(exclude)})")
    print(f"output:  {args.output_list} ({len(kept)})")
    print(f"removed: {len(removed)}")
    if args.removed_out:
        print(f"removed_out: {args.removed_out}")


if __name__ == "__main__":
    main()
