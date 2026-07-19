import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.signal import convolve2d, wiener


def normalize01(value):
    value = value.astype(np.float32)
    lo = float(np.percentile(value, 1.0))
    hi = float(np.percentile(value, 99.0))
    if hi <= lo + 1e-6:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - lo) / (hi - lo), 0.0, 1.0)


def gaussian_psf(size=9, sigma=1.4):
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2
    yy, xx = np.meshgrid(axis, axis)
    psf = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    return psf / psf.sum()


def lucy_richardson(image, iterations=8, size=9, sigma=1.4):
    psf = gaussian_psf(size=size, sigma=sigma)
    estimate = np.clip(image, 1e-4, 1.0)
    psf_mirror = psf[::-1, ::-1]
    for _ in range(iterations):
        conv = convolve2d(estimate, psf, mode="same", boundary="symm")
        relative_blur = image / np.clip(conv, 1e-4, 1.0)
        estimate *= convolve2d(relative_blur, psf_mirror, mode="same", boundary="symm")
        estimate = np.clip(estimate, 0.0, 1.0)
    return estimate


def lucy_aux(ink, iterations, sigma, ink_mix, dog_mix, cutoff, dog_small=0.7, dog_large=3.8):
    deconv_ink = lucy_richardson(
        np.clip(ink, 0.0, 1.0),
        iterations=iterations,
        size=9,
        sigma=sigma,
    )
    dog_ink = dog_ink_from_gray(1.0 - deconv_ink, small=dog_small, large=dog_large)
    mixed = np.clip(deconv_ink * ink_mix + dog_ink * dog_mix, 0.0, 1.0)
    return np.clip((mixed - cutoff) / max(1.0 - cutoff, 1e-6), 0.0, 1.0)


def dog_ink_from_gray(gray, small=1.0, large=5.0):
    blur_small = ndimage.gaussian_filter(gray, sigma=small)
    blur_large = ndimage.gaussian_filter(gray, sigma=large)
    return np.clip(blur_large - blur_small, 0.0, 1.0)


def flow_coherence(gray):
    smooth = ndimage.gaussian_filter(gray, sigma=1.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    jxx = ndimage.gaussian_filter(gx * gx, sigma=2.0)
    jyy = ndimage.gaussian_filter(gy * gy, sigma=2.0)
    jxy = ndimage.gaussian_filter(gx * gy, sigma=2.0)
    denom = jxx + jyy + 1e-6
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / denom
    edge = normalize01(np.sqrt(gx * gx + gy * gy))
    return np.clip(coherence, 0.0, 1.0), edge


def process(image, mode):
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    ink = 1.0 - gray
    if mode == "cutoff":
        ink = np.clip((ink - 0.12) / 0.88, 0.0, 1.0)
        ink = ink ** 0.85
    elif mode == "dog":
        dog_ink = dog_ink_from_gray(gray, small=1.0, large=5.0)
        ink = np.clip(ink * 0.45 + dog_ink * 2.2, 0.0, 1.0)
        ink = np.clip((ink - 0.06) / 0.94, 0.0, 1.0)
    elif mode == "flowdog":
        dog_ink = dog_ink_from_gray(gray, small=0.8, large=4.5)
        coherence, edge = flow_coherence(gray)
        ink = np.clip(ink * 0.35 + dog_ink * 2.0 + edge * 0.35, 0.0, 1.0)
        ink = ink * (0.35 + 0.65 * coherence)
        ink = np.clip((ink - 0.045) / 0.955, 0.0, 1.0)
    elif mode == "flowmask":
        dog_ink = dog_ink_from_gray(gray, small=0.9, large=4.0)
        coherence, edge = flow_coherence(gray)
        ink = np.clip(dog_ink * 2.4 + edge * 0.45 + ink * 0.20, 0.0, 1.0)
        ink = ink * (0.25 + 0.75 * coherence)
        ink = np.clip((ink - 0.055) / 0.945, 0.0, 1.0)
        ink = ndimage.gaussian_filter(ink, sigma=0.45)
    elif mode == "wiener":
        denoised = np.clip(wiener(gray, mysize=(5, 5), noise=0.002), 0.0, 1.0)
        detail = np.clip(denoised - ndimage.gaussian_filter(denoised, sigma=3.5), -1.0, 1.0)
        ink = np.clip((1.0 - denoised) * 0.60 + np.maximum(-detail, 0.0) * 2.0, 0.0, 1.0)
        ink = np.clip((ink - 0.05) / 0.95, 0.0, 1.0)
    elif mode == "lucy":
        ink = lucy_aux(ink, iterations=8, sigma=1.35, ink_mix=0.55, dog_mix=1.7, cutoff=0.055)
    elif mode == "lucy_mild":
        ink = lucy_aux(ink, iterations=5, sigma=1.15, ink_mix=0.48, dog_mix=1.25, cutoff=0.070)
    elif mode == "lucy_strong":
        ink = lucy_aux(ink, iterations=12, sigma=1.55, ink_mix=0.60, dog_mix=1.95, cutoff=0.045)
    elif mode == "lucy_thin":
        ink = lucy_aux(ink, iterations=8, sigma=1.30, ink_mix=0.42, dog_mix=1.45, cutoff=0.090)
    elif mode == "bilateral":
        src = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(src, d=7, sigmaColor=28, sigmaSpace=5).astype(np.float32) / 255.0
        dog_ink = dog_ink_from_gray(filtered, small=0.9, large=4.2)
        ink = np.clip((1.0 - filtered) * 0.45 + dog_ink * 2.0, 0.0, 1.0)
        ink = np.clip((ink - 0.05) / 0.95, 0.0, 1.0)
    elif mode == "nlmeans":
        src = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        filtered = cv2.fastNlMeansDenoising(src, None, h=9, templateWindowSize=7, searchWindowSize=21)
        filtered = filtered.astype(np.float32) / 255.0
        dog_ink = dog_ink_from_gray(filtered, small=0.9, large=4.5)
        ink = np.clip((1.0 - filtered) * 0.42 + dog_ink * 2.1, 0.0, 1.0)
        ink = np.clip((ink - 0.05) / 0.95, 0.0, 1.0)
    elif mode == "edgehint":
        blur = np.asarray(
            image.convert("L").filter(ImageFilter.GaussianBlur(radius=1.4)),
            dtype=np.float32,
        ) / 255.0
        edge = np.abs(gray - blur)
        ink = np.clip(ink * 0.30 + edge * 5.0, 0.0, 1.0)
    else:
        raise ValueError(f"unknown mode: {mode}")
    out = np.clip((1.0 - ink) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "cutoff",
            "dog",
            "flowdog",
            "flowmask",
            "wiener",
            "lucy",
            "lucy_mild",
            "lucy_strong",
            "lucy_thin",
            "bilateral",
            "nlmeans",
            "edgehint",
        ],
        required=True,
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(input_dir.glob("*_out.png")):
        process(Image.open(path), args.mode).save(output_dir / path.name)
        count += 1
    print(f"saved {count} files to {output_dir} mode={args.mode}")


if __name__ == "__main__":
    main()
