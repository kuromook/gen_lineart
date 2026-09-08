"""Batch-preprocess rough tiles with `LineartDetector(coarse=True)` instead
of `lineart_anime`, based on the 2026-08-26 finding (inbox/initial_notice.md)
that `lineart_anime` discards ~50% of a rough sketch's visible ink (it's
trained to extract lines from finished/colored anime art, not to interpret
rough/construction-line pencil sketches), while `lineart_coarse` -- derived
from the Simo-Serra et al. rough-sketch-cleanup network -- retains ~170% by
the same ink-fraction measure (i.e. picks up faint strokes the anime
detector misses) and visually tracks the raw rough much more closely.

Mirrors ../lineart/tools/pair_extraction/preprocess_lineart_anime_condition.py's
structure/CLI so it's a drop-in swap for --rough-dir in train_controlnet.py /
infer_controlnet.py, but kept track-local since the shared script is
lineart_anime-specific.
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image

IMAGE_SIZE = 480


def get_detector(device):
    from controlnet_aux import LineartDetector
    detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
    detector.to(device)
    return detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--rough-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = get_detector(device)

    with open(args.file_list) as f:
        names = [line.strip() for line in f if line.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    written, skipped = 0, 0
    for i, name in enumerate(names):
        out_path = out_dir / name
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        rough_path = os.path.join(args.rough_dir, name)
        rgb = Image.open(rough_path).convert("RGB").resize((args.image_size, args.image_size))
        with torch.no_grad():
            out = detector(rgb, coarse=True, detect_resolution=args.image_size, image_resolution=args.image_size)
        out = out.resize((args.image_size, args.image_size)).convert("RGB")
        out.save(out_path)
        written += 1
        if (i + 1) % 200 == 0 or (i + 1) == len(names):
            elapsed = time.time() - start
            rate = elapsed / max(written, 1)
            print(
                f"[preprocess_lineart_coarse_condition] {i + 1}/{len(names)} "
                f"written={written} skipped={skipped} elapsed={elapsed:.0f}s "
                f"({rate:.2f}s/img, eta={rate * (len(names) - i - 1):.0f}s)",
                flush=True,
            )

    print(f"[preprocess_lineart_coarse_condition] done: wrote {written}, skipped {skipped} to {out_dir}")


if __name__ == "__main__":
    main()
