"""Measure grayscale haze and line-near uncertainty around GT line art."""

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


def parse_model(value):
    if "=" in value:
        label, directory = value.split("=", 1)
        return label, Path(directory)
    return value, Path("results") / value


def halo_metrics(pred_gray, gt_gray, inner_px=2.0, outer_px=9.0):
    pred_ink = 1.0 - pred_gray.astype(np.float32) / 255.0
    gt_ink = gt_gray < GT_THRESHOLD
    dist = cv2.distanceTransform((~gt_ink).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    core = gt_ink
    halo_band = (dist > inner_px) & (dist <= outer_px)
    far_bg = dist > outer_px
    faint = (pred_ink > 0.03) & (pred_ink < 0.35)
    strong = pred_ink >= 0.35
    background_faint = far_bg & faint
    background_ink = far_bg & (pred_ink > 0.03)
    core_ink = float(pred_ink[core].mean()) if core.any() else 0.0
    line_near_ink = float(pred_ink[halo_band].mean()) if halo_band.any() else 0.0
    background_haze = float(pred_ink[far_bg].mean()) if far_bg.any() else 0.0
    background_faint_ink = float(pred_ink[background_faint].mean()) if background_faint.any() else 0.0
    return {
        "core_ink_mean": core_ink,
        "halo_band_ink_mean": line_near_ink,
        "halo_band_faint_ratio": float(faint[halo_band].mean()) if halo_band.any() else 0.0,
        "far_bg_ink_mean": float(pred_ink[far_bg].mean()) if far_bg.any() else 0.0,
        "far_bg_faint_ratio": float(faint[far_bg].mean()) if far_bg.any() else 0.0,
        "halo_to_core": float(line_near_ink / max(core_ink, 1e-6)) if core.any() and halo_band.any() else 0.0,
        "line_near_uncertainty_ink": line_near_ink,
        "line_near_uncertainty_faint_ratio": float(faint[halo_band].mean()) if halo_band.any() else 0.0,
        "line_near_strong_ratio": float(strong[halo_band].mean()) if halo_band.any() else 0.0,
        "line_near_to_core": float(line_near_ink / max(core_ink, 1e-6)) if core.any() and halo_band.any() else 0.0,
        "background_haze_ink_mean": background_haze,
        "background_haze_faint_ink_mean": background_faint_ink,
        "background_haze_faint_ratio": float(faint[far_bg].mean()) if far_bg.any() else 0.0,
        "background_haze_area_ratio": float(background_faint.mean()),
        "background_ink_area_ratio": float(background_ink.mean()),
        "background_haze_to_core": float(background_haze / max(core_ink, 1e-6)) if core.any() else 0.0,
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
    models = [parse_model(value) for value in args.models]
    for sample in read_sample_list(args.sample_list):
        base = normalize_name(sample)
        gt = load_gray(Path("dataset/pairs_480") / args.split / "line" / f"{base}.jpg")
        for model, directory in models:
            pred = load_gray(directory / f"{base}_out.png")
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

    print("model line_near_ink line_near_faint bg_haze bg_haze_area halo_to_core")
    for model, _directory in models:
        part = [row for row in rows if row["model"] == model]
        mean = {
            key: np.mean([row[key] for row in part])
            for key in (
                "line_near_uncertainty_ink",
                "line_near_uncertainty_faint_ratio",
                "background_haze_ink_mean",
                "background_haze_area_ratio",
                "halo_to_core",
            )
        }
        print(
            f"{model} "
            f"{mean['line_near_uncertainty_ink']:.4f} "
            f"{mean['line_near_uncertainty_faint_ratio']:.4f} "
            f"{mean['background_haze_ink_mean']:.4f} "
            f"{mean['background_haze_area_ratio']:.4f} "
            f"{mean['halo_to_core']:.4f}"
        )
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
