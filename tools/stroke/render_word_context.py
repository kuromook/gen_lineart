#!/usr/bin/env python
"""Words that resist standalone naming, shown IN PANEL CONTEXT.

For each unnamed word: example occurrences where the whole panel's strokes are drawn
light gray and the cluster's own strokes (mapped back to native coords via its
centroid/scale) are drawn red. Naming may succeed with position/context visible.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook import load
from train_codebook2 import Codebook2

CLUSTERS = "results/cluster_set_20260919"
PACK = "results/panel_pack_20260919"
CKPT = "results/invariance_20260920/l4_w500/codebook2.pt"
OUT = Path("results/word_semantics_20260920/word_context.png")
UNNAMED = [213, 339, 318, 342, 218, 214, 209, 317, 207]
N_EX, H_CANVAS = 4, 240


def main():
    dev = torch.device("cuda")
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    m = Codebook2(levels=tuple(ck["levels"]), coord_bins=ck["args"].get("coord_bins", 0)).to(dev)
    m.load_state_dict(ck["model"]); m.eval(); m.rfsq.stages = ck["args"].get("stages", 3)
    Pt, Wt, Mt, mt = load(CLUSTERS, "train", dev)
    Pv, Wv, Mv, mv = load(CLUSTERS, "test", dev)
    P = torch.cat([Pt, Pv]); W = torch.cat([Wt, Wv]); M = torch.cat([Mt, Mv]); mrows = mt + mv
    words = []
    with torch.no_grad():
        for i in range(0, len(P), 4096):
            _q, w = m.encode(P[i:i + 4096].float(), W[i:i + 4096].float(), M[i:i + 4096], 1)
            words.append(w.cpu())
    words = torch.cat(words).numpy()

    panels_csv = list(csv.DictReader(open(f"{PACK}/panels.csv")))
    strokes = np.load(f"{PACK}/strokes.npy", mmap_mode="r")

    def render(panel_id, cl_pts, cl_mask, cy, cx, scale):
        pr = panels_csv[panel_id]
        rows = np.asarray(strokes[int(pr["start"]):int(pr["start"]) + int(pr["n"])])
        pp = rows[:, :32].reshape(-1, 16, 2)
        ink = np.abs(pp).sum(-1) > 0
        H = max(int(np.ceil(pp[..., 1][ink].max())) + 2, 2) if ink.any() else 2
        W = max(int(np.ceil(pp[..., 0][ink].max())) + 2, 2) if ink.any() else 2
        z = H_CANVAS / H
        canvas = np.full((H_CANVAS, max(int(W * z), 2), 3), 255, np.uint8)

        def xform(pts2):
            return np.round(pts2 * z)[:, ::-1].astype(np.int32).reshape(-1, 1, 2)

        for s in range(pp.shape[0]):
            if not ink[s].any():
                continue
            xy = np.clip(xform(pp[s][ink[s]]), 0, [canvas.shape[1] - 1, H_CANVAS - 1])
            cv2.polylines(canvas, [xy], False, (200, 200, 200), 1)
        nat = (cl_pts / scale) + torch.tensor([cy, cx]).to(cl_pts.device)
        for s in range(cl_pts.shape[0]):
            if not cl_mask[s]:
                continue
            xy = np.clip(xform(nat[s].cpu().numpy()), 0, [canvas.shape[1] - 1, H_CANVAS - 1])
            cv2.polylines(canvas, [xy], False, (30, 30, 230), 2)
        return canvas

    by_word = defaultdict(list)
    for j, (r, wd) in enumerate(zip(mrows, words)):
        if r["split"] == "train":
            by_word[int(wd)].append(j)

    rng = np.random.default_rng(0)
    out_rows = []
    for w in UNNAMED:
        idx = by_word.get(w, [])
        pick = rng.choice(idx, min(N_EX, len(idx)), replace=False) if idx else []
        cells = []
        for j in pick:
            r = mrows[int(j)]
            cells.append(render(int(r["panel"]), P[int(j)], M[int(j)],
                                float(r["cy"]), float(r["cx"]), float(r["scale"])))
        if not cells:
            cells = [np.full((H_CANVAS, 120, 3), 255, np.uint8)]
        hmax = max(c.shape[1] for c in cells)
        cells = [np.pad(c, ((0, 0), (0, hmax - c.shape[1]), (0, 0)), constant_values=255) for c in cells]
        tag = np.full((H_CANVAS, 110, 3), 255, np.uint8)
        cv2.putText(tag, f"word {w}", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
        out_rows.append(np.hstack([tag] + cells))
    wmax = max(r.shape[1] for r in out_rows)
    out_rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)), constant_values=255) for r in out_rows]
    cv2.imwrite(str(OUT), np.vstack(out_rows))
    print("montage ->", OUT, "(gray = all panel strokes, red = the word's cluster, in native position)")


if __name__ == "__main__":
    main()
