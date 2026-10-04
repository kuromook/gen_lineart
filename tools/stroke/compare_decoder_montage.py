#!/usr/bin/env python
"""Side-by-side montage: regression decoder (l4_w500) vs categorical decoder A,
same held-out clusters. Rows of 4: truth | regression 3-stage | categorical 3-stage
| categorical word-only."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook import load
from train_codebook2 import Codebook2, topn_mask
from codebook_diag import draw

REG = "results/invariance_20260920/l4_w500/codebook2.pt"
CAT = "results/decoderA_20260920/w500/codebook2.pt"
OUT = Path("results/decoderA_20260920/compare_montage.png")


def main():
    dev = torch.device("cuda")
    vp, vw, vm, _ = load("results/cluster_set_20260919", "test", dev)
    vp, vw, vm = vp[:4096].float(), vw[:4096].float(), vm[:4096]
    dec = {}
    for tag, path in (("reg", REG), ("cat", CAT)):
        ck = torch.load(path, map_location=dev, weights_only=False)
        m = Codebook2(levels=tuple(ck["levels"]), coord_bins=ck["args"].get("coord_bins", 0)).to(dev)
        m.load_state_dict(ck["model"]); m.eval(); m.rfsq.stages = ck["args"].get("stages", 3)
        with torch.no_grad():
            outs = {}
            for name, ks in (("full", None), ("word", 1)):
                q, _w = m.encode(vp, vw, vm, ks)
                ex, pp, pw, cl, _lg = m.decode(q)
                outs[name] = (pp.float(), topn_mask(ex.float(), cl.float()))
        dec[tag] = outs
    rows = []
    for i in range(0, 48, 4):
        cells = [draw(vp[i].cpu().numpy(), vm[i].cpu().numpy())]
        cells.append(draw(dec["reg"]["full"][0][i].cpu().numpy(), dec["reg"]["full"][1][i].cpu().numpy()))
        cells.append(draw(dec["cat"]["full"][0][i].cpu().numpy(), dec["cat"]["full"][1][i].cpu().numpy()))
        cells.append(draw(dec["cat"]["word"][0][i].cpu().numpy(), dec["cat"]["word"][1][i].cpu().numpy()))
        rows.append(np.hstack([np.hstack([c, np.full((200, 6, 3), 120, np.uint8)]) for c in cells]))
    grid = np.vstack([np.hstack(rows[j:j + 3]) for j in range(0, len(rows), 3)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), grid)
    print("montage ->", OUT, "(each row of 4: truth | regression 3-stage | categorical 3-stage | categorical word-only)")


if __name__ == "__main__":
    main()
