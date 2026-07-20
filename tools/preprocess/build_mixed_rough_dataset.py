"""Build a mixed raw/cleaned rough training directory with duplicated targets."""

import argparse
import shutil
from pathlib import Path


def read_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--raw-rough-dir", required=True)
    parser.add_argument("--cleaned-rough-dir", required=True)
    parser.add_argument("--line-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-list", required=True)
    args = parser.parse_args()

    raw_rough_dir = Path(args.raw_rough_dir)
    cleaned_rough_dir = Path(args.cleaned_rough_dir)
    line_dir = Path(args.line_dir)
    output_root = Path(args.output_root)
    rough_out = output_root / "rough"
    line_out = output_root / "line"
    rows = []

    for sample in read_list(args.file_list):
        base = Path(sample).stem
        for kind, source_dir in (("raw", raw_rough_dir), ("clean", cleaned_rough_dir)):
            out_name = f"{base}__{kind}.jpg"
            link_or_copy(source_dir / f"{base}.jpg", rough_out / out_name)
            link_or_copy(line_dir / f"{base}.jpg", line_out / out_name)
            rows.append(out_name)

    Path(args.output_list).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_list, "w") as file:
        for row in rows:
            file.write(f"{row}\n")

    print(f"saved mixed rows={len(rows)} root={output_root}")
    print(f"saved: {args.output_list}")


if __name__ == "__main__":
    main()
