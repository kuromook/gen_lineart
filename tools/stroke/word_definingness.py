#!/usr/bin/env python
"""Which strokes define a cluster's word? (the user's hypothesis, 2026-09-19)

User: a short stroke removed from a cluster probably leaves its word unchanged,
so infilling it is detail-guessing; long strokes likely define the word -- with
hair as a possible exception, since a lock of hair is many long, similar strokes.

Test: for held-out clusters, drop each stroke in turn, re-encode with the
codebook, and record (a) whether the stage-1 word changes and (b) how far the
continuous pre-quantisation code moves. Group by the dropped stroke's length
(absolute px and share of the cluster's total ink) and by cluster type, where a
"parallel group" (hair, hatching) is a cluster whose strokes point the same way
(length-weighted orientation coherence >= 0.8) and the rest are contours/mixed.
"""
import csv, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook import load, stroke_feats  # noqa: E402
from train_codebook2 import Codebook2  # noqa: E402


def main():
    dev = torch.device("cuda")
    ck = torch.load("results/codebook2_20260919/codebook2.pt", map_location=dev, weights_only=False)
    m = Codebook2().to(dev); m.load_state_dict(ck["model"]); m.eval()
    vp, vw, vm, meta = load("results/cluster_set_20260919", "test", dev)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(vp), 3000, replace=False)
    rows = []

    def z_and_word(p, w, mk):
        x = m.inp(stroke_feats(p, w))
        x = torch.cat([m.cls.expand(len(x), -1, -1), x], 1)
        mm = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=dev), ~mk], 1)
        h = m.enc(x, src_key_padding_mask=mm)
        z = m.to_z(m.pre_q(h[:, 0]))
        b = torch.tanh(z + m.rfsq.shift) * m.rfsq.half_l - m.rfsq.offset       # bounded, before rounding
        _q, word = m.rfsq(z, 1)
        return b, word

    with torch.no_grad():
        for i in idx:
            p, w, mk = vp[i].float(), vw[i].float(), vm[i]
            n = int(mk.sum())
            if n < 4:
                continue
            P = p[None].repeat(n + 1, 1, 1, 1); W = w[None].repeat(n + 1, 1); M = mk[None].repeat(n + 1, 1)
            for k in range(n):
                M[k + 1, k] = False
            b, word = z_and_word(P, W, M)
            s = p[:n].cpu().numpy()
            arc = np.linalg.norm(np.diff(s, axis=1), axis=2).sum(1)
            v = s[:, -1] - s[:, 0]; ang = np.arctan2(v[:, 0], v[:, 1])
            coh = abs((arc * np.exp(2j * ang)).sum()) / max(arc.sum(), 1e-6)
            scale = float(meta[i]["scale"])
            for k in range(n):
                rows.append({"arc_px": arc[k] / scale, "ink_share": arc[k] / max(arc.sum(), 1e-6),
                             "rank": int((arc > arc[k]).sum()), "n": n,
                             "changed": int(word[k + 1] != word[0]),
                             "dz": float((b[k + 1] - b[0]).norm()), "coh": coh})
    import pandas  # noqa: F401  (not available -> fall back below)


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError:
        pass
