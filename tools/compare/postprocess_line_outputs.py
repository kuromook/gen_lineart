import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def process_image(image, mode):
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    ink = 1.0 - gray

    if mode == "threshold45":
        out_ink = (ink >= 0.45).astype(np.float32)
    elif mode == "threshold35":
        out_ink = (ink >= 0.35).astype(np.float32)
    elif mode == "curve2":
        out_ink = np.clip(ink, 0.0, 1.0) ** 0.55
        out_ink = out_ink * out_ink * (3.0 - 2.0 * out_ink)
    elif mode == "unsharp_curve":
        blurred = np.asarray(
            image.convert("L").filter(ImageFilter.GaussianBlur(radius=1.2)),
            dtype=np.float32,
        ) / 255.0
        sharp_gray = np.clip(gray + 1.8 * (gray - blurred), 0.0, 1.0)
        out_ink = np.clip(1.0 - sharp_gray, 0.0, 1.0) ** 0.55
    else:
        raise ValueError(f"unknown mode: {mode}")

    output = np.clip((1.0 - out_ink) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="L")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["threshold45", "threshold35", "curve2", "unsharp_curve"],
        required=True,
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in sorted(input_dir.glob("*_out.png")):
        output = process_image(Image.open(path), args.mode)
        output.save(output_dir / path.name)
        count += 1
    print(f"saved {count} files to {output_dir} mode={args.mode}")


if __name__ == "__main__":
    main()
