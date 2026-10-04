#!/usr/bin/env python
"""What did the 'ant-like' decoded dots originally look like?

Draws the ground-truth clusters that codebook_diag.py uses (test clusters 0,4,...,44,
the same cells as montage_decode.png), one large cell each, with the word id the
l4_w500 encoder assigns. Normalized frame: bbox long side = 56 units (like the decoder).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook import load
from train_codebook2 import Codebook2

CKPT = "results/invariance_20260920/l4_w500/codebook2.pt"
OUT = Path("results/decoderA_20260920/truth_clusters.png")
SIZE = 300


def draw_cell(pts, keep, label):
    img = np.full((SIZE, SIZE, 3), 255, np.uint8)
    z = SIZE / 80.0
    for s in range(pts.shape[0]):
        if not keep[s]:
            continue
        xy = np.round((pts[s] + 40) * z)[:, ::-1].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [xy], False, (40, 40, 40), 2)
    cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    return img


def main():
    dev = torch.device("cuda")
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    m = Codebook2(levels=tuple(ck["levels"]), coord_bins=ck["args"].get("coord_bins", 0)).to(dev)
    m.load_state_dict(ck["model"]); m.eval(); m.rfsq.stages = ck["args"].get("stages", 3)
    vp, vw, vm, _ = load("results/cluster_set_20260919", "test", dev)
    idx = list(range(0, 48, 4))
    with torch.no_grad():
        _q, w = m.encode(vp[idx].float(), vw[idx].float(), vm[idx], 1)
    cells = [draw_cell(vp[i].cpu().numpy(), vm[i].cpu().numpy(), f"#{i}  word {int(w[j])}")
             for j, i in enumerate(idx)]
    rows = [np.hstack(cells[j:j + 3]) for j in range(0, len(cells), 3)]
    cv2.imwrite(str(OUT), np.vstack(rows))
    print("montage ->", OUT, f"({len(cells)} truth clusters, same cells as w*/montage_decode.png)")


if __name__ == "__main__":
    main()
