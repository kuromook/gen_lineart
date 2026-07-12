"""Build a low-ratio kurip VLM-accept training list and mixed line directory."""

import argparse
import os
import random
from pathlib import Path


DEFAULT_BASE_LIST = "dataset/pairs_480/valid_train_warm_regions.txt"
DEFAULT_KURIP_LIST = "dataset/pairs_480/valid_train_kurip_vlm_accept_top500.txt"
DEFAULT_BASE_LINE_DIR = "dataset/pairs_480/train/line"
DEFAULT_KURIP_LINE_DIR = "dataset/pairs_480/train/line_kurip_vlm_candidates_top500_clean_t192_cc8"
DEFAULT_OUTPUT_LIST = "dataset/pairs_480/valid_train_warm_regions_kurip_vlm_accept190.txt"
DEFAULT_OUTPUT_KURIP_LIST = "dataset/pairs_480/valid_train_kurip_vlm_accept190.txt"
DEFAULT_OUTPUT_LINE_DIR = "dataset/pairs_480/train/line_kurip_vlm_accept190_mix_clean_t192_cc8"


def read_names(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(os.path.relpath(src, dst.parent), dst)


def build_mix(args):
    base_names = read_names(args.base_list)
    kurip_names = read_names(args.kurip_list)
    rng = random.Random(args.seed)
    selected_kurip = sorted(rng.sample(kurip_names, args.kurip_count))
    output_names = base_names + selected_kurip

    output_line_dir = Path(args.output_line_dir)
    output_line_dir.mkdir(parents=True, exist_ok=True)

    for name in base_names:
        src = Path(args.base_line_dir) / name
        if not src.exists():
            raise FileNotFoundError(src)
        link_or_copy(src, output_line_dir / name)

    for name in selected_kurip:
        src = Path(args.kurip_line_dir) / name
        if not src.exists():
            raise FileNotFoundError(src)
        link_or_copy(src, output_line_dir / name)

    Path(args.output_list).write_text("\n".join(output_names) + "\n")
    Path(args.output_kurip_list).write_text("\n".join(selected_kurip) + "\n")

    kurip_ratio = len(selected_kurip) / len(output_names)
    print(f"base: {len(base_names)}")
    print(f"kurip: {len(selected_kurip)}")
    print(f"total: {len(output_names)}")
    print(f"kurip_ratio: {kurip_ratio:.3f}")
    print(f"output_list: {args.output_list}")
    print(f"output_kurip_list: {args.output_kurip_list}")
    print(f"output_line_dir: {args.output_line_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-list", default=DEFAULT_BASE_LIST)
    parser.add_argument("--kurip-list", default=DEFAULT_KURIP_LIST)
    parser.add_argument("--base-line-dir", default=DEFAULT_BASE_LINE_DIR)
    parser.add_argument("--kurip-line-dir", default=DEFAULT_KURIP_LINE_DIR)
    parser.add_argument("--output-list", default=DEFAULT_OUTPUT_LIST)
    parser.add_argument("--output-kurip-list", default=DEFAULT_OUTPUT_KURIP_LIST)
    parser.add_argument("--output-line-dir", default=DEFAULT_OUTPUT_LINE_DIR)
    parser.add_argument("--kurip-count", type=int, default=190)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    build_mix(args)


if __name__ == "__main__":
    main()
