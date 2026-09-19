#!/usr/bin/env python
"""What does the codebook's decoder actually draw?

The collapse gate (decoded arc-length std >= 50% of the truth) failed twice.
Before redesigning a third time, look: compare decoded and true stroke-length
distributions quantile by quantile (are long strokes shortened, are all strokes
the same length?), and draw held-out clusters beside their decodes -- from all
three stages and from the word alone.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook import load
from train_codebook2 import Codebook2, topn_mask

OUT = Path("results/codebook2_20260919")


def arcs(p):
    return (p[:, :, 1:] - p[:, :, :-1]).norm(dim=-1).sum(-1)


def draw(pts, keep, size=200):
    img = np.full((size, size, 3), 255, np.uint8)
    z = size / 70.0
    for s in range(pts.shape[0]):
        if not keep[s]:
            continue
        xy = np.round((pts[s] + 35) * z)[:, ::-1].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [xy], False, (40, 40, 40), 1)
    return img


def main():
    dev = torch.device("cuda")
    ck = torch.load(OUT / "codebook2.pt", map_location=dev, weights_only=False)
    model = Codebook2().to(dev); model.load_state_dict(ck["model"]); model.eval()
    model.rfsq.stages = ck["args"].get("stages", 3)
    vp, vw, vm, _ = load("results/cluster_set_20260919", "test", dev)
    vp, vw, vm = vp[:4096].float(), vw[:4096].float(), vm[:4096]
    res = {}
    with torch.no_grad():
        for tag, ks in (("3段", None), ("単語のみ", 1)):
            q, _w = model.encode(vp, vw, vm, ks)
            ex, pp, pw, cl = model.decode(q)
            keep = topn_mask(ex.float(), cl.float())
            res[tag] = (pp.float(), keep)
    at = arcs(vp)[vm].cpu().numpy()
    qs = [10, 25, 50, 75, 90, 99]
    print(f"{'':<10}" + "".join(f"{'p'+str(k):>8}" for k in qs) + f"{'std':>8}")
    print(f"{'正解':<10}" + "".join(f"{np.percentile(at, k):>8.1f}" for k in qs) + f"{at.std():>8.1f}")
    for tag, (pp, keep) in res.items():
        a = arcs(pp)[keep].cpu().numpy()
        print(f"{tag:<10}" + "".join(f"{np.percentile(a, k):>8.1f}" for k in qs) + f"{a.std():>8.1f}")
    rows = []
    for i in range(0, 48, 4):
        cells = [draw(vp[i].cpu().numpy(), vm[i].cpu().numpy())]
        for tag in ("3段", "単語のみ"):
            pp, keep = res[tag]
            cells.append(draw(pp[i].cpu().numpy(), keep[i].cpu().numpy()))
        rows.append(np.hstack([np.hstack([c, np.full((200, 6, 3), 120, np.uint8)]) for c in cells]))
    grid = np.vstack([np.hstack(rows[j:j + 3]) for j in range(0, len(rows), 3)])
    cv2.imwrite(str(OUT / "montage_decode.png"), grid)
    print("montage ->", OUT / "montage_decode.png", "(each triple: truth | 3-stage decode | word-only decode)")


if __name__ == "__main__":
    main()
