"""NC4 / arm A2: put BOTH pools through the preprocessor, then profile them.

doc/prereg_external_standard_20261030.md: the preprocessor (LineartDetector,
here controlnet_aux's LineartAnimeDetector on lllyasviel/Annotators, the same
one the project uses elsewhere) is this project's baseline. If the split
survives it the difference is in the drawing; if it disappears the difference
was in the raster pipeline.

The preprocessor is run on the tiles BEFORE the white-point pass, because it
outputs its own black-on-white raster -- the paper colour question does not
survive it, which is the point of the control.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SIZE = 480


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--glob", default="*.jpg")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from controlnet_aux import LineartAnimeDetector

    detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    if torch.cuda.is_available():
        detector = detector.to("cuda")

    paths = sorted(Path(args.src).glob(args.glob))
    if args.limit:
        paths = paths[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    papers = []
    for i, path in enumerate(paths):
        rgb = Image.open(path).convert("RGB").resize((SIZE, SIZE))
        with torch.no_grad():
            out = detector(rgb, image_resolution=SIZE)
        # The detector returns white strokes on black; the profile tool reads
        # ink as < 128, and the author's tiles are black on white. Invert both
        # sides identically so neither side gets a different operation.
        gray = 255 - np.asarray(out.resize((SIZE, SIZE)).convert("L"))
        papers.append(int(np.bincount(gray.ravel(), minlength=256).argmax()))
        Image.fromarray(gray).save(out_dir / f"{path.stem}.png")
        if (i + 1) % 200 == 0:
            rate = (time.time() - started) / (i + 1)
            print(f"  {i + 1}/{len(paths)}  {rate:.2f}s/img", flush=True)

    papers = np.array(papers)
    print(f"written: {len(paths)}  paper(mode): min={papers.min()} median={int(np.median(papers))} max={papers.max()}")
    print(f"elapsed: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
