"""Cut the external 1024px images into 480px tiles, exactly as registered.

doc/prereg_external_standard_20261030.md section 9-4. The point of the
allotment is that the external side gets the same clumping as the author's
side -- 457 sources spread over 1,489 tiles as 1/2/3/4 per source -- so that
"tiles from the same source look alike" cannot be the thing that separates
the pools.

Selection is a measurement: every window that is not written is counted by
reason and printed at the end, including the ones the ink gate drops.

The gate is applied to the raw pixels, because the gate value (0.011992) is
the author pool's own minimum measured on raw pixels. The white point is NOT
applied here -- both sides get it at read time, after the quality-95 JPEG,
so that neither side sees the operations in a different order.
"""

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

import normalize

SEED = 20261030
TILE = 480
# Registered window order (section 9-4). Centre first, then the four corners.
WINDOWS = [(272, 272), (0, 0), (544, 544), (0, 544), (544, 0)]
# Registered allotment: how many tiles each source contributes, 69/46/40/302.
ALLOTMENT = {1: 69, 2: 46, 3: 40, 4: 302}


def allot(n_images, seed=SEED):
    counts = []
    for per_image, how_many in sorted(ALLOTMENT.items()):
        counts.extend([per_image] * how_many)
    if len(counts) != n_images:
        raise SystemExit(f"allotment is for {len(counts)} images, got {n_images}")
    random.Random(seed).shuffle(counts)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="directory of generated 1024px images")
    parser.add_argument("--out", required=True, help="directory to write 480px JPEG tiles into")
    args = parser.parse_args()

    images = sorted(Path(args.src).glob("*.png"))
    counts = allot(len(images))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dropped = collections.Counter()
    short = []
    written = 0

    for image_path, want in zip(images, counts):
        gray = np.asarray(Image.open(image_path).convert("L"))
        if gray.shape != (1024, 1024):
            dropped["image_not_1024"] += 1
            continue
        got = 0
        for (y, x) in WINDOWS:
            if got == want:
                break
            window = gray[y : y + TILE, x : x + TILE]
            if normalize.ink_ratio(window) < normalize.INK_GATE:
                dropped["window_below_ink_gate"] += 1
                continue
            name = f"{image_path.stem}_w{y:04d}_{x:04d}.jpg"
            normalize.save_jpeg(window, out_dir / name)
            written += 1
            got += 1
        if got < want:
            short.append((image_path.name, want, got))
            dropped["source_short_of_allotment"] += want - got

    print(f"images: {len(images)}  tiles written: {written}  (target 1489)")
    print("dropped, by reason:")
    for reason in sorted(dropped):
        print(f"  {reason}: {dropped[reason]}")
    print(f"sources that could not fill their allotment: {len(short)}")
    for name, want, got in short[:10]:
        print(f"  {name}: wanted {want}, got {got}")
    if len(short) > 10:
        print(f"  ... and {len(short) - 10} more")

    report = out_dir.parent / f"tiling_report_{out_dir.name}.json"
    report.write_text(
        json.dumps(
            {
                "images": len(images),
                "written": written,
                "target": 1489,
                "dropped": dict(dropped),
                "short_sources": short,
                "seed": SEED,
                "windows": WINDOWS,
                "ink_gate": normalize.INK_GATE,
            },
            indent=1,
        )
    )
    print(f"saved: {report}")


if __name__ == "__main__":
    main()
