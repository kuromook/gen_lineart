"""Measure grayscale halo-like ink around GT line art."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = 480
GT_THRESHOLD = 128


def read_sample_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize_name(name):
    return name[:-4] if name.endswith(".jpg") else name


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def halo_metrics(pred_gray, gt_gray, inner_px=2.0, outer_px=9.0):
    pred_ink = 1.0 - pred_gray.astype(np.float32) / 255.0
    gt_ink = gt_gray < GT_THRESHOLD
    dist = cv2.distanceTransform((~gt_ink).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    core = gt_ink
    halo_band = (dist > inner_px) & (dist <= outer_px)
    far_bg = dist > outer_px
    faint = (pred_ink > 0.03) & (pred_ink < 0.35)
    return {
        "core_ink_mean": float(pred_ink[core].mean()) if core.any() else 0.0,
        "halo_band_ink_mean": float(pred_ink[halo_band].mean()) if halo_band.any() else 0.0,
        "halo_band_faint_ratio": float(faint[halo_band].mean()) if halo_band.any() else 0.0,
        "far_bg_ink_mean": float(pred_ink[far_bg].mean()) if far_bg.any() else 0.0,
        "far_bg_faint_ratio": float(faint[far_bg].mean()) if far_bg.any() else 0.0,
        "halo_to_core": float(pred_ink[halo_band].mean() / max(pred_ink[core].mean(), 1e-6))
        if core.any() and halo_band.any()
        else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--inner-px", type=float, default=2.0)
    parser.add_argument("--outer-px", type=float, default=9.0)
    args = parser.parse_args()

    rows = []
    for sample in read_sample_list(args.sample_list):
        base = normalize_name(sample)
        gt = load_gray(Path("dataset/pairs_480") / args.split / "line" / f"{base}.jpg")
        for model in args.models:
            pred = load_gray(Path("results") / model / f"{base}_out.png")
            rows.append(
                {
                    "sample": base,
                    "model": model,
                    **halo_metrics(pred, gt, args.inner_px, args.outer_px),
                }
            )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with open(args.output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("model halo_ink halo_faint far_ink far_faint halo_to_core")
    for model in args.models:
        part = [row for row in rows if row["model"] == model]
        mean = {
            key: np.mean([row[key] for row in part])
            for key in (
                "halo_band_ink_mean",
                "halo_band_faint_ratio",
                "far_bg_ink_mean",
                "far_bg_faint_ratio",
                "halo_to_core",
            )
        }
        print(
            f"{model} "
            f"{mean['halo_band_ink_mean']:.4f} "
            f"{mean['halo_band_faint_ratio']:.4f} "
            f"{mean['far_bg_ink_mean']:.4f} "
            f"{mean['far_bg_faint_ratio']:.4f} "
            f"{mean['halo_to_core']:.4f}"
        )
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
