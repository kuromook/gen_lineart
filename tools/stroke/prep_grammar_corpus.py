#!/usr/bin/env python
"""Build the grammar corpus: panels = sequences of cluster tokens.

Each cluster becomes 4 tokens: [word, pos_y, pos_x, scale]
  word   0..499   from the l4_w500 encoder (keep_stages=1)
  pos_y  0..15    centroid y bin  (cy / panel_H)
  pos_x  0..15    centroid x bin  (cx / panel_W)
  scale  0..11    log2(scale) in [-7, 4), 12 bins   (scale = 56 / native bbox long side)
Within a panel, clusters are sorted by native bbox long side descending (coarse first,
canonical order for orderless rasters; ties break by reading order cy, cx).

Token id space (single embedding table, field known by position % 4):
  word   0..499
  pos_y  512..527
  pos_x  528..543
  scale  544..555
  BOS 556, EOS 557
Output: results/grammar_corpus_20260920/corpus.npz
  seq (N, L) int16, -1 padded; split (N,) bool (True=train); panels (N,) int64 panel idx
  dims (n_panels, 2) int64 [H, W]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook import load  # noqa: E402
from train_codebook2 import Codebook2  # noqa: E402

CKPT = "results/invariance_20260920/l4_w500/codebook2.pt"
PACK = "results/panel_pack_20260919"
CLUSTERS = "results/cluster_set_20260919"
OUT = Path("results/grammar_corpus_20260920")

OFF_Y, OFF_X, OFF_S = 512, 528, 544
BOS, EOS = 556, 557
N_BINS = 16
S_BINS, S_LO, S_HI = 12, -7.0, 4.0


def panel_dims():
    panels = list(csv.DictReader(open(f"{PACK}/panels.csv")))
    strokes = np.load(f"{PACK}/strokes.npy", mmap_mode="r")
    dims = np.zeros((len(panels), 2), np.int64)
    for i, p in enumerate(panels):
        rows = strokes[int(p["start"]):int(p["start"]) + int(p["n"])]
        pts = np.asarray(rows[:, :32]).reshape(-1, 16, 2)
        ink = np.abs(pts).sum(-1) > 0
        if ink.any():
            dims[i, 0] = int(np.ceil(pts[..., 1][ink].max())) + 2   # H (y)
            dims[i, 1] = int(np.ceil(pts[..., 0][ink].max())) + 2   # W (x)
        else:
            dims[i] = 1
    return dims


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

    by_panel = defaultdict(list)
    for r, wd in zip(mrows, words):
        by_panel[int(r["panel"])].append((wd, float(r["scale"]), float(r["cy"]), float(r["cx"]), r["split"] == "train"))
    dims = panel_dims()
    seqs, split, pids = [], [], []
    maxc = 0
    for pid, cs in sorted(by_panel.items()):
        cs.sort(key=lambda t: (-(1.0 / t[1]), t[2], t[3]))      # coarse first, reading-order ties
        H, Wd = dims[pid]
        toks = [BOS]
        for wd, sc, cy, cx, _tr in cs:
            sy = min(int(cy / max(H, 1) * N_BINS), N_BINS - 1)
            sx = min(int(cx / max(Wd, 1) * N_BINS), N_BINS - 1)
            ss = min(max(int((np.log2(sc) - S_LO) / (S_HI - S_LO) * S_BINS), 0), S_BINS - 1)
            toks += [int(wd), OFF_Y + sy, OFF_X + sx, OFF_S + ss]
        toks.append(EOS)
        seqs.append(toks); pids.append(pid); split.append(cs[0][4]); maxc = max(maxc, len(toks))
    L = min(maxc + 1, 1024)
    seq = np.full((len(seqs), L), -1, np.int16)
    for i, t in enumerate(seqs):
        seq[i, :min(len(t), L)] = t[:L]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "corpus.npz", seq=seq, split=np.array(split), panels=np.array(pids), dims=dims)
    tr = seq[np.array(split)]
    print(f"panels {len(seqs)} (train {tr.shape[0]}, test {len(seqs) - tr.shape[0]})  max len {maxc} -> L {L}")
    wpos = tr[:, 1::4]; wpos = wpos[wpos >= 0]
    print(f"word tokens {wpos.size}, used words {np.unique(wpos).size}/500")
    print("->", OUT / "corpus.npz")


if __name__ == "__main__":
    main()
