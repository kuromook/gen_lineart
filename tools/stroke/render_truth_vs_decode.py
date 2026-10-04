#!/usr/bin/env python
"""Truth vs decoder-A v2 side by side, SAME order and labels as truth_clusters.png.

Each cell: [truth | v2 3-stage decode | v2 word-only decode] for one test cluster.
Cluster order: test #0,4,8,...,44 (same cells as truth_clusters.png and w*/montage_decode.png).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook import load
from train_codebook2 import Codebook2, topn_mask
from codebook_diag import draw

CKPT = "results/decoderA_20260920/v2_w500/codebook2.pt"
OUT = Path("results/decoderA_20260920/truth_vs_v2.png")
SIZE = 220


def labeled(img, text):
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)
    return img


def main():
    dev = torch.device("cuda")
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    m = Codebook2(levels=tuple(ck["levels"]), coord_bins=ck["args"].get("coord_bins", 0)).to(dev)
    m.load_state_dict(ck["model"]); m.eval(); m.rfsq.stages = ck["args"].get("stages", 3)
    vp, vw, vm, _ = load("results/cluster_set_20260919", "test", dev)
    idx = list(range(0, 48, 4))
    dec = {}
    with torch.no_grad():
        for name, ks in (("full", None), ("word", 1)):
            q, _w = m.encode(vp[idx].float(), vw[idx].float(), vm[idx], ks)
            ex, pp, pw, cl, _lg = m.decode(q)
            dec[name] = (pp.float(), topn_mask(ex.float(), cl.float()))
    cells = []
    for j, i in enumerate(idx):
        trio = np.hstack([
            labeled(draw(vp[i].cpu().numpy(), vm[i].cpu().numpy(), SIZE), f"#{i} truth"),
            labeled(draw(dec["full"][0][j].cpu().numpy(), dec["full"][1][j].cpu().numpy(), SIZE), f"#{i} v2 3-stage"),
            labeled(draw(dec["word"][0][j].cpu().numpy(), dec["word"][1][j].cpu().numpy(), SIZE), f"#{i} v2 word-only"),
        ])
        cells.append(trio)
    grid = np.vstack([np.hstack(cells[j:j + 3]) for j in range(0, len(cells), 3)])
    cv2.imwrite(str(OUT), grid)
    print("montage ->", OUT, "(each cell: truth | v2 3-stage | v2 word-only; order matches truth_clusters.png)")


if __name__ == "__main__":
    main()
