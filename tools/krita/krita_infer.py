"""Track O stage 2: run Krita's shipped Fast Sketch Cleanup weights on our rough tiles.

Registration: doc/prereg_krita_stage2_20261104.md. Run with the dedicated venv, isolated:

    data/krita_venv/bin/python -I tools/krita/krita_infer.py --out results/krita_stage2_20261104

Re-implements the plugin 1.0.2 pipeline (read in stage 1, not imported -- ordinance 3):
  * grey uint8 / 255 -> [0,1]; `inv` arms feed 1-x and return 1-y (the plugin's `invert`)
  * size rounded up to a multiple of 16 (480*s already is); outer margin 40 filled with 0 in
    MODEL space (so `noinv` gets a black frame, as the plugin does)
  * 256 parts, only the inner 176 px kept and stitched
  * output = truncate(clip(y*255)) to uint8 (the plugin's float->uint8 assignment)
One deviation, recorded: the plugin hands float64 to OpenVINO; here float32 (the graph's type).
Scale arms: rough LANCZOS-upscaled by s, inferred, output BOX-downscaled back to 480.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PAIRS = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
LISTS = {"lineart": PAIRS / "holdout_lineart_family.txt", "housei": PAIRS / "holdout_housei_100.txt"}
MODELS = ROOT / "data/krita_models"
PART, MARGIN, DIV = 256, 40, 16
SIZE = 480


def rough_path(tile):
    for split in ("train", "test"):
        p = PAIRS / split / "rough" / tile
        if p.exists():
            return p
    raise FileNotFoundError(tile)


def infer_plugin(compiled, gray_u8, invert):
    data = gray_u8.astype(np.float32) / 255.0
    if invert:
        data = 1.0 - data
    h, w = data.shape
    assert h % DIV == 0 and w % DIV == 0, (h, w)
    data = data[None, None]
    true = PART - 2 * MARGIN
    extra = np.ceil([h / true, w / true]) * true          # the plugin over-pads by the full span
    padded = np.pad(data, ((0, 0), (0, 0), (MARGIN, MARGIN + int(extra[0])),
                           (MARGIN, MARGIN + int(extra[1]))), constant_values=0)
    out = np.pad(data, ((0, 0), (0, 0), (0, int(extra[0])), (0, int(extra[1]))), constant_values=0)
    req = compiled.create_infer_request()
    sy = 0
    while True:
        sx = 0
        while True:
            part = np.ascontiguousarray(padded[:, :, sy:sy + PART, sx:sx + PART], dtype=np.float32)
            req.infer({0: part})
            res = req.get_output_tensor().data
            out[:, :, sy:sy + true, sx:sx + true] = res[:, :, MARGIN:PART - MARGIN, MARGIN:PART - MARGIN]
            if sx + PART >= w + 2 * MARGIN:
                break
            sx += true
        if sy + PART >= h + 2 * MARGIN:
            break
        sy += true
    y = out[0, 0, :h, :w]
    if invert:
        y = 1.0 - y
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)   # truncation, as the plugin


def run_arm(compiled, invert, scale, tile, crop_nc3=False):
    img = Image.open(rough_path(tile)).convert("L")
    assert img.size == (SIZE, SIZE), img.size
    if scale != 1:
        img = img.resize((SIZE * scale, SIZE * scale), Image.LANCZOS)
    y = infer_plugin(compiled, np.asarray(img), invert)
    if crop_nc3:
        return Image.fromarray(y[:SIZE, :SIZE])            # NC3: the forgotten downscale
    o = Image.fromarray(y)
    if scale != 1:
        o = o.resize((SIZE, SIZE), Image.BOX)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args()
    out_root = ROOT / args.out
    tiles = []
    for pool, lst in LISTS.items():
        t = [l.strip() for l in open(lst) if l.strip()]
        tiles += t[: args.limit] if args.limit else t
    core = ov.Core()
    compiled = {m: core.compile_model(core.read_model(MODELS / f"{m}Model.xml"), "CPU")
                for m in ("Sketchy", "Smooth")}

    # IC5: determinism -- the same tile twice must give identical bytes
    for m in compiled:
        a = run_arm(compiled[m], False, 1, tiles[0]).tobytes()
        b = run_arm(compiled[m], False, 1, tiles[0]).tobytes()
        print(f"IC5 {m}: identical={a == b}", flush=True)
        if a != b:
            sys.exit("IC5 failed")

    arms = [(m, inv, s) for m in ("Sketchy", "Smooth") for inv in (False, True) for s in (1, 2, 4)]
    names = {f"{m.lower()}_{'inv' if inv else 'noinv'}_s{s}": (m, inv, s, False) for m, inv, s in arms}
    names["nc3_sketchy_noinv_s4_crop"] = ("Sketchy", False, 4, True)
    for name, (m, inv, s, crop) in names.items():
        if args.arms and name not in args.arms:
            continue
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        done = 0
        for tile in tiles:
            p = d / f"{Path(tile).stem}_out.png"
            if p.exists():
                done += 1
                continue
            run_arm(compiled[m], inv, s, tile, crop).save(p)
            done += 1
        print(f"{name}: {done} tiles", flush=True)


if __name__ == "__main__":
    main()
