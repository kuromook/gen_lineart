"""Clean rough sketch inputs before rough-to-lineart inference."""

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = 480


def read_list(path):
    with open(path) as file:
        return [line.strip() for line in file if line.strip()]


def normalize01(value):
    value = value.astype(np.float32)
    lo = float(np.percentile(value, 1.0))
    hi = float(np.percentile(value, 99.5))
    if hi <= lo + 1e-6:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - lo) / (hi - lo), 0.0, 1.0)


def softcut_ink(ink, cutoff, gamma=1.0):
    ink = np.clip((ink - cutoff) / max(1.0 - cutoff, 1e-6), 0.0, 1.0)
    if gamma != 1.0:
        ink = ink**gamma
    return ink


def dog_ink(gray, small=0.8, large=3.4):
    small_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=small, sigmaY=small)
    large_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=large, sigmaY=large)
    return np.clip(large_blur - small_blur, 0.0, 1.0)


def background_haze_cleanup(gray, cutoff=0.045, gamma=0.92, bg_sigma=18.0):
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=bg_sigma, sigmaY=bg_sigma)
    corrected = np.clip(gray - (bg - np.percentile(bg, 92.0)), 0.0, 1.0)
    ink = 1.0 - corrected
    ink = softcut_ink(ink, cutoff=cutoff, gamma=gamma)
    return 1.0 - ink


def sketch_line_cleanup(gray, cutoff=0.060, ink_mix=0.34, ridge_mix=2.25, edge_mix=0.10, blur=0.35):
    ink = 1.0 - gray
    ridge = dog_ink(gray, small=0.7, large=3.0)
    edge = normalize01(cv2.Laplacian(gray, cv2.CV_32F, ksize=3) * -1.0)
    line = np.clip(ink * ink_mix + ridge * ridge_mix + edge * edge_mix, 0.0, 1.0)
    line = softcut_ink(line, cutoff=cutoff, gamma=0.85)
    if blur > 0.0:
        line = cv2.GaussianBlur(line, (0, 0), sigmaX=blur, sigmaY=blur)
    return 1.0 - np.clip(line, 0.0, 1.0)


def edge_preserve_cleanup(gray):
    src = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(src, d=7, sigmaColor=22, sigmaSpace=5).astype(np.float32) / 255.0
    ink = 1.0 - filtered
    ridge = dog_ink(filtered, small=0.8, large=3.2)
    line = np.clip(ink * 0.42 + ridge * 1.65, 0.0, 1.0)
    return 1.0 - softcut_ink(line, cutoff=0.045, gamma=0.95)


def process(gray, mode):
    if mode == "identity":
        return gray
    if mode == "background":
        return background_haze_cleanup(gray)
    if mode == "background_mild":
        return background_haze_cleanup(gray, cutoff=0.030, gamma=0.96, bg_sigma=22.0)
    if mode == "background_strong":
        return background_haze_cleanup(gray, cutoff=0.070, gamma=0.88, bg_sigma=14.0)
    if mode == "line":
        return sketch_line_cleanup(gray)
    if mode == "line_mild":
        return sketch_line_cleanup(gray, cutoff=0.040, ink_mix=0.42, ridge_mix=1.65, edge_mix=0.05, blur=0.25)
    if mode == "line_strong":
        return sketch_line_cleanup(gray, cutoff=0.085, ink_mix=0.25, ridge_mix=2.85, edge_mix=0.16, blur=0.20)
    if mode == "background_line":
        return sketch_line_cleanup(background_haze_cleanup(gray))
    if mode == "line_background":
        return background_haze_cleanup(sketch_line_cleanup(gray))
    if mode == "line_background_mild":
        return background_haze_cleanup(
            sketch_line_cleanup(gray, cutoff=0.040, ink_mix=0.42, ridge_mix=1.65, edge_mix=0.05, blur=0.25),
            cutoff=0.030,
            gamma=0.96,
            bg_sigma=22.0,
        )
    if mode == "line_background_strong":
        return background_haze_cleanup(
            sketch_line_cleanup(gray, cutoff=0.085, ink_mix=0.25, ridge_mix=2.85, edge_mix=0.16, blur=0.20),
            cutoff=0.070,
            gamma=0.88,
            bg_sigma=14.0,
        )
    if mode == "softcut_only":
        return 1.0 - softcut_ink(1.0 - gray, cutoff=0.055, gamma=0.92)
    if mode == "edge_preserve":
        return edge_preserve_cleanup(gray)
    raise ValueError(f"unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "identity",
            "background",
            "background_mild",
            "background_strong",
            "line",
            "line_mild",
            "line_strong",
            "background_line",
            "line_background",
            "line_background_mild",
            "line_background_strong",
            "softcut_only",
            "edge_preserve",
        ],
        required=True,
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for sample in read_list(args.file_list):
        base = Path(sample).stem
        image = cv2.imread(str(input_dir / f"{base}.jpg"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(input_dir / f"{base}.jpg")
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        gray = image.astype(np.float32) / 255.0
        cleaned = process(gray, args.mode)
        out = np.clip(cleaned * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{base}.jpg"), out)
        cv2.imwrite(str(output_dir / f"{base}_out.png"), out)
        count += 1

    print(f"saved {count} files to {output_dir} mode={args.mode}")


if __name__ == "__main__":
    main()
