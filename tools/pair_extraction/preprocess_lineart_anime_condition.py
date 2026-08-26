"""Batch-preprocess rough tiles with the `lineart_anime` ControlNet
conditioning preprocessor (`controlnet_aux.LineartAnimeDetector`), the
format the public `control_v11p_sd15s2_lineart_anime` checkpoint actually
expects (a clean white-line-on-black edge map) rather than a raw noisy
pencil scan -- per the 2026-08-08 decision to fine-tune that checkpoint on
real pairs with this preprocessor applied to rough tiles before
conditioning (doc/work_log.md, "ControlNet Hallucination Re-Diagnosed").

Same detector/invocation as tools/evaluation/condition_roundtrip_fidelity.py's
preprocess_condition() (kept consistent so roundtrip-fidelity eval and
training conditioning are the same transform), factored out here as a
reusable batch tool since no such script existed yet -- previously this
preprocessing was only ever run ad hoc on a handful of diagnostic tiles.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

IMAGE_SIZE = 480


def get_detector(device):
    from controlnet_aux import LineartAnimeDetector
    detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
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
            out = detector(rgb, image_resolution=args.image_size)
        out = out.resize((args.image_size, args.image_size)).convert("RGB")
        out.save(out_path)
        written += 1
        if (i + 1) % 100 == 0 or (i + 1) == len(names):
            elapsed = time.time() - start
            rate = elapsed / max(written, 1)
            print(
                f"[preprocess_lineart_anime_condition] {i + 1}/{len(names)} "
                f"written={written} skipped={skipped} elapsed={elapsed:.0f}s "
                f"({rate:.2f}s/img, eta={rate * (len(names) - i - 1):.0f}s)",
                flush=True,
            )

    print(f"[preprocess_lineart_anime_condition] done: wrote {written}, skipped {skipped} to {out_dir}")


if __name__ == "__main__":
    main()
