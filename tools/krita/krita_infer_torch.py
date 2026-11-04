"""Track O stage 2, amendment A1: the same plugin pipeline as krita_infer.py, on CUDA.

OpenVINO on CPU measured ~60 s per tile for Sketchy at x4 (Smooth is ~6x the parameters),
so the twelve arms would take days. This rebuilds the two graphs in torch from the layer
specs in the shipped .xml and the f16 constants in the .bin (offsets read from the .xml),
and runs the identical tiling. Equivalence to OpenVINO is checked (--check) on the smoke
tiles before any score is read; see the registration's amendment A1.

    /home/sh1/deepl/lineart/venv/bin/python -I tools/krita/krita_infer_torch.py --out results/krita_stage2_20261104
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PAIRS = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
LISTS = {"lineart": PAIRS / "holdout_lineart_family.txt", "housei": PAIRS / "holdout_housei_100.txt"}
MODELS = ROOT / "data/krita_models"
PART, MARGIN, DIV, SIZE = 256, 40, 16, 480


def build_from_ir(name):
    """Walk the IR: each Convolution / ConvolutionBackpropData takes its weight from
    Convert(Const), followed by Add(bias Const) and ReLU (Sigmoid at the end)."""
    root = ET.parse(MODELS / f"{name}Model.xml").getroot()
    blob = (MODELS / f"{name}Model.bin").read_bytes()
    layers = {l.get("id"): l for l in root.find("layers")}
    edges = {}
    for e in root.find("edges"):
        edges[(e.get("to-layer"), e.get("to-port"))] = e.get("from-layer")

    def const_of(layer_id):
        l = layers[layer_id]
        while l.get("type") == "Convert":
            l = layers[edges[(l.get("id"), "0")]]
        assert l.get("type") == "Const", l.get("type")
        d = l.find("data").attrib
        assert d["element_type"] == "f16"
        shape = [int(x) for x in d["shape"].split(",")] if d["shape"] else []
        arr = np.frombuffer(blob, dtype=np.float16, count=int(d["size"]) // 2, offset=int(d["offset"]))
        return torch.from_numpy(arr.astype(np.float32).reshape(shape))

    mods = []
    for lid, l in layers.items():           # IR lists layers in topological order
        t = l.get("type")
        if t in ("Convolution", "ConvolutionBackpropData"):
            d = l.find("data").attrib
            w = const_of(edges[(lid, "1")])
            s = int(d["strides"].split(",")[0])
            p = int(d["pads_begin"].split(",")[0])
            assert d["pads_begin"] == d["pads_end"], d
            if t == "Convolution":
                m = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2], s, p)
            else:
                assert d.get("output_padding", "0, 0").replace(" ", "") in ("0,0", ""), d
                m = nn.ConvTranspose2d(w.shape[0], w.shape[1], w.shape[2], s, p)
            m.weight.data.copy_(w)
            m.bias.data.zero_()
            mods.append(m)
        elif t == "Add":
            b = const_of(edges[(lid, "1")])
            mods[-1].bias.data.copy_(b.reshape(-1))
        elif t == "ReLU":
            mods.append(nn.ReLU())
        elif t == "Sigmoid":
            mods.append(nn.Sigmoid())
    return nn.Sequential(*mods).eval()


def rough_path(tile):
    for split in ("train", "test"):
        p = PAIRS / split / "rough" / tile
        if p.exists():
            return p
    raise FileNotFoundError(tile)


@torch.no_grad()
def infer_plugin(net, gray_u8, invert, dev):
    data = gray_u8.astype(np.float32) / 255.0
    if invert:
        data = 1.0 - data
    h, w = data.shape
    assert h % DIV == 0 and w % DIV == 0
    true = PART - 2 * MARGIN
    ex_h, ex_w = int(np.ceil(h / true) * true), int(np.ceil(w / true) * true)
    padded = np.pad(data, ((MARGIN, MARGIN + ex_h), (MARGIN, MARGIN + ex_w)))
    out = np.zeros((h + ex_h, w + ex_w), np.float32)
    coords = []
    sy = 0
    while True:
        sx = 0
        while True:
            coords.append((sy, sx))
            if sx + PART >= w + 2 * MARGIN:
                break
            sx += true
        if sy + PART >= h + 2 * MARGIN:
            break
        sy += true
    for i in range(0, len(coords), 8):
        chunk = coords[i:i + 8]
        x = torch.from_numpy(np.stack([padded[y:y + PART, x:x + PART] for y, x in chunk])[:, None]).to(dev)
        r = net(x).float().cpu().numpy()[:, 0]
        for (y, xx), ri in zip(chunk, r):
            out[y:y + true, xx:xx + true] = ri[MARGIN:PART - MARGIN, MARGIN:PART - MARGIN]
    y = out[:h, :w]
    if invert:
        y = 1.0 - y
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def run_arm(net, invert, scale, tile, dev, crop_nc3=False):
    img = Image.open(rough_path(tile)).convert("L")
    assert img.size == (SIZE, SIZE)
    if scale != 1:
        img = img.resize((SIZE * scale, SIZE * scale), Image.LANCZOS)
    y = infer_plugin(net, np.asarray(img), invert, dev)
    if crop_nc3:
        return Image.fromarray(y[:SIZE, :SIZE])
    o = Image.fromarray(y)
    return o.resize((SIZE, SIZE), Image.BOX) if scale != 1 else o


def arm_table():
    names = {f"{m.lower()}_{'inv' if inv else 'noinv'}_s{s}": (m, inv, s, False)
             for m in ("Sketchy", "Smooth") for inv in (False, True) for s in (1, 2, 4)}
    names["nc3_sketchy_noinv_s4_crop"] = ("Sketchy", False, 4, True)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default=None, help="OpenVINO output root to compare against")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    nets = {m: build_from_ir(m).to(dev) for m in ("Sketchy", "Smooth")}
    for m, n in nets.items():
        print(m, "params", sum(p.numel() for p in n.parameters()), flush=True)
    names = arm_table()

    if args.check:   # A1 equivalence: every OpenVINO output on disk vs torch, same tile
        ref = ROOT / args.check
        worst = 0
        for d in sorted(ref.iterdir()):
            if d.name not in names:
                continue
            m, inv, s, crop = names[d.name]
            for f in sorted(d.glob("*_out.png")):
                tile = f.name.replace("_out.png", ".jpg")
                a = np.asarray(Image.open(f)).astype(int)
                b = np.asarray(run_arm(nets[m], inv, s, tile, dev, crop)).astype(int)
                diff = np.abs(a - b)
                worst = max(worst, diff.max())
                print(f"A1 {d.name} {tile}: max|d|={diff.max()} frac>1={(diff > 1).mean():.6f}", flush=True)
        print(f"A1 worst max|d| = {worst}")
        return

    tiles = []
    for lst in LISTS.values():
        tiles += [l.strip() for l in open(lst) if l.strip()]
    out_root = ROOT / args.out
    for name, (m, inv, s, crop) in names.items():
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for tile in tiles:
            p = d / f"{Path(tile).stem}_out.png"
            if not p.exists():
                run_arm(nets[m], inv, s, tile, dev, crop).save(p)
            n += 1
        print(f"{name}: {n} tiles", flush=True)


if __name__ == "__main__":
    main()
