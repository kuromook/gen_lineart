"""Potrace-style vector-then-raster cleanup for soft line-art outputs,
as an alternative to raw raster-threshold post-processing (which
`doc/badrough_lucy_thin_threshold_notes.md` found produces thousands of
sub-12px speckle fragments on threshold).

`libpotrace-dev`/`pypotrace` are not installable in this environment (no
sudo). This approximates potrace's pipeline with OpenCV instead: binarize
-> find contours -> drop small-area contours (despeckle, potrace's
"turdsize" analog) -> simplify contour polygons (Douglas-Peucker,
potrace's corner-smoothing analog) -> re-render as filled polygons. Holes
(e.g. an eye outline) are tracked via contour hierarchy and punched out
after the fills, not left as literal holes in the fill pass.
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def vectorize_binarize(gray, threshold=128, min_area=6.0, approx_eps=1.2):
    ink_mask = (gray < threshold).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(ink_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(ink_mask)
    if not contours:
        return canvas
    hierarchy = hierarchy[0]

    fills, holes = [], []
    for cnt, h in zip(contours, hierarchy):
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, approx_eps, True)
        is_hole = h[3] != -1
        (holes if is_hole else fills).append(approx)

    cv2.drawContours(canvas, fills, -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(canvas, holes, -1, 0, thickness=cv2.FILLED)
    return canvas


def process_image(gray, mode, threshold, min_area, approx_eps):
    threshold_match = re.fullmatch(r"threshold(\d{2,3})", mode)
    if threshold_match:
        th = int(threshold_match.group(1))
        ink = (gray < th).astype(np.uint8) * 255
        return 255 - ink
    if mode == "vector":
        ink = vectorize_binarize(gray, threshold=threshold, min_area=min_area, approx_eps=approx_eps)
        return 255 - ink
    if mode == "vector_nosimplify":
        ink = vectorize_binarize(gray, threshold=threshold, min_area=min_area, approx_eps=0.0)
        return 255 - ink
    if mode == "vector_nodespeckle":
        ink = vectorize_binarize(gray, threshold=threshold, min_area=0.0, approx_eps=approx_eps)
        return 255 - ink
    raise ValueError(f"unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        help="thresholdNNN, vector, vector_nosimplify, vector_nodespeckle",
    )
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--min-area", type=float, default=6.0)
    parser.add_argument("--approx-eps", type=float, default=1.2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in sorted(input_dir.glob("*_out.png")):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        out = process_image(gray, args.mode, args.threshold, args.min_area, args.approx_eps)
        cv2.imwrite(str(output_dir / path.name), out)
        count += 1
    print(f"saved {count} files to {output_dir} mode={args.mode}")


if __name__ == "__main__":
    main()
