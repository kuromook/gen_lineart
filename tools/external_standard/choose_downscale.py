"""Pick the downscale for the external side, by the criterion registered first.

doc/prereg_external_standard_v2_20261030.md section 2: the chosen ratio is the
one whose three quantities land closest to the author's, by largest relative
error. The three are line_width_p50 (3.8200), faint_of_drawn_ratio (0.3839)
and deep_black_ratio (0.0020) -- all measured on the author's normalized pool.

No AUC is computed here. The criterion was written before the smoke ran, so
the choice does not move the bar.
"""

import argparse
import glob
import os
import pathlib
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation"))

import normalize
from measure_lineart_profile import profile_metrics

TARGETS = {
    "line_width_p50": 3.8200,
    "faint_of_drawn_ratio": 0.3839,
    "deep_black_ratio": 0.0020,
}
CANDIDATES = [2048, 1706, 1536, 1280, 1024]
TILE = 480


def measure(paths, short_side, tmp):
    out = {k: [] for k in TARGETS}
    for path in paths:
        im = Image.open(path).convert("L")
        if im.size[0] != short_side:
            im = im.resize((short_side, short_side), Image.LANCZOS)
        o = (short_side - TILE) // 2
        window = np.asarray(im.crop((o, o, o + TILE, o + TILE)))
        jpg = os.path.join(tmp, "t.jpg")
        normalize.save_jpeg(window, jpg)
        gray, _ = normalize.equalize_white_point(np.asarray(Image.open(jpg).convert("L")))
        png = os.path.join(tmp, "t.png")
        Image.fromarray(gray).save(png)
        m = profile_metrics(pathlib.Path(png), 0.08)
        for k in TARGETS:
            out[k].append(m[k])
    return {k: float(np.median(v)) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", action="append", required=True)
    args = parser.parse_args()

    paths = []
    for src in args.src:
        paths += sorted(glob.glob(os.path.join(src, "*.png")))
    print(f"smoke images: {len(paths)}")
    print(f"target (author): " + "  ".join(f"{k}={v}" for k, v in TARGETS.items()))
    print()

    tmp = tempfile.mkdtemp()
    header = f"{'短辺':>8s}" + "".join(f"{k[:18]:>20s}" for k in TARGETS) + f"{'最大相対誤差':>16s}"
    print(header)
    rows = []
    for side in CANDIDATES:
        got = measure(paths, side, tmp)
        worst = max(abs(got[k] - TARGETS[k]) / max(TARGETS[k], 1e-9) for k in TARGETS)
        rows.append((side, got, worst))
        print(f"{side:8d}" + "".join(f"{got[k]:20.4f}" for k in TARGETS) + f"{worst:16.3f}")
    print(f"{'作者側':>8s}" + "".join(f"{TARGETS[k]:20.4f}" for k in TARGETS))
    print()
    best = min(rows, key=lambda r: r[2])
    print(f"**登録した基準で選ばれる短辺: {best[0]}**  (最大相対誤差 {best[2]:.3f})")
    print("この値を doc/prereg_external_standard_v2_20261030.md に追記して commit してから本番を走らせる。")


if __name__ == "__main__":
    main()
