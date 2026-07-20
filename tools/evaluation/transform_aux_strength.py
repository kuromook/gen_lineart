"""Create deterministic variants of an atari/aux directory."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def transform(gray, mode):
    ink = 1.0 - gray.astype(np.float32) / 255.0
    if mode.startswith("weak"):
        scale = float(mode.replace("weak", "")) / 100.0
        ink = ink * scale
    elif mode.startswith("hard"):
        threshold = float(mode.replace("hard", "")) / 100.0
        ink = (ink >= threshold).astype(np.float32)
    elif mode.startswith("softcut"):
        threshold = float(mode.replace("softcut", "")) / 100.0
        ink = np.clip((ink - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    elif mode == "blur":
        ink = cv2.GaussianBlur(ink, (0, 0), sigmaX=0.8, sigmaY=0.8)
    elif mode == "open":
        kernel = np.ones((3, 3), np.uint8)
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    elif mode == "identity":
        pass
    else:
        raise ValueError(f"unknown mode: {mode}")
    return np.clip((1.0 - ink) * 255.0, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(input_dir.glob("*_out.png")):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(path)
        cv2.imwrite(str(output_dir / path.name), transform(gray, args.mode))
        count += 1
    print(f"saved {count} files to {output_dir} mode={args.mode}")


if __name__ == "__main__":
    main()
