#!/usr/bin/env python
"""Word-level semantics validation (plan gate 5 + co-occurrence).

1. Nameability montage: top-20 frequent words, 8 truth examples each + the word-only
   prototype (decode of the word id). Human names the words -> plan gate 5 (>=10/20).
2. Co-occurrence: panel-level word co-occurrence (unique words per panel), PMI-ranked
   pairs among frequent words + a montage of the top pairs.

Output: results/word_semantics_20260920/  (nameability.png, cooc_pairs.png, cooc.md)
"""
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook import load
from train_codebook2 import Codebook2, topn_mask
from codebook_diag import draw

CKPT = "results/invariance_20260920/l4_w500/codebook2.pt"
CLUSTERS = "results/cluster_set_20260919"
OUT = Path("results/word_semantics_20260920")
N_TOP, N_EX, SIZE = 20, 8, 150


def main():
    dev = torch.device("cuda")
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    levels = tuple(ck["levels"])
    m = Codebook2(levels=levels, coord_bins=ck["args"].get("coord_bins", 0)).to(dev)
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
    for j, (r, wd) in enumerate(zip(mrows, words)):
        by_panel[int(r["panel"])].append((j, wd, float(r["scale"]), float(r["cy"]), float(r["cx"]), r["split"] == "train"))
    for pid in by_panel:
        by_panel[pid].sort(key=lambda t: (-(1.0 / t[2]), t[3], t[4]))
    tr_panels = {p: cs for p, cs in by_panel.items() if cs[0][5]}

    # ---- word-only prototypes for every used word ----
    used = np.unique(words)
    basis = np.cumprod((1,) + levels[:-1])
    half_w = np.floor(np.array(levels) / 2)
    q = np.zeros((len(used), len(levels)), np.float32)
    for i, w in enumerate(used):
        d = (int(w) // basis) % np.array(levels)
        q[i] = (d - half_w) / half_w
    with torch.no_grad():
        ex, pp, pw, cl, _lg = m.decode(torch.from_numpy(q).to(dev))
        keep = topn_mask(ex.float(), cl.float())
    proto = {int(w): (pp[i].float().cpu().numpy(), keep[i].cpu().numpy()) for i, w in enumerate(used)}

    # ---- 1. nameability montage ----
    tr_mask = np.array([r["split"] == "train" for r in mrows])
    freq = np.bincount(words[tr_mask], minlength=500)
    top = np.argsort(-freq)[:N_TOP]
    rows = []
    for w in top:
        exs = [t[0] for cs in tr_panels.values() for t in cs if t[1] == w][:N_EX]
        cells = [draw(P[j].cpu().numpy(), M[j].cpu().numpy(), SIZE) for j in exs]
        while len(cells) < N_EX:
            cells.append(np.full((SIZE, SIZE, 3), 255, np.uint8))
        pr, pk = proto[int(w)]
        cells.append(draw(pr, pk, SIZE))
        row = np.hstack(cells)
        tag = np.full((SIZE, 150, 3), 255, np.uint8)
        cv2.putText(tag, f"word {int(w)}", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
        cv2.putText(tag, f"freq {int(freq[w])}", (4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
        rows.append(np.hstack([tag, row]))
    grid = np.vstack(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "nameability.png"), grid)
    print("nameability ->", OUT / "nameability.png", "(rows: word, 8 truth examples + last = word-only prototype)")

    # ---- 2. co-occurrence ----
    panel_words = {p: sorted({t[1] for t in cs}) for p, cs in tr_panels.items()}
    n_w = np.zeros(500); n_p = defaultdict(int); n_panel = len(panel_words)
    for ws in panel_words.values():
        for w in ws:
            n_w[w] += 1
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                n_p[(ws[i], ws[j])] += 1
                n_p[(ws[j], ws[i])] += 1
    pmi = []
    for (a, b), c in n_p.items():
        if a < b and n_w[a] >= 30 and n_w[b] >= 30:
            pmi.append((np.log2(c * n_panel / (n_w[a] * n_w[b])), int(a), int(b), int(c)))
    pmi.sort(reverse=True)
    lines = ["# panel-level word co-occurrence (train, unique words per panel)",
             f"panels {n_panel}", "",
             "| rank | PMI | word A | freq | word B | freq | cooccur panels |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r, (v, a, b, c) in enumerate(pmi[:30]):
        lines.append(f"| {r + 1} | {v:.2f} | {a} | {int(n_w[a])} | {b} | {int(n_w[b])} | {c} |")
    # frequent words only: the semantically meaningful units live here, not in the rare tail
    pmi_f = [x for x in pmi if n_w[x[1]] >= 1000 and n_w[x[2]] >= 1000]
    lines += ["", "## frequent words (freq >= 1000)", "",
              "| rank | PMI | word A | freq | word B | freq | cooccur panels |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r, (v, a, b, c) in enumerate(pmi_f[:30]):
        lines.append(f"| {r + 1} | {v:.2f} | {a} | {int(n_w[a])} | {b} | {int(n_w[b])} | {c} |")
    (OUT / "cooc.md").write_text("\n".join(lines))
    print("cooc table ->", OUT / "cooc.md", f"(frequent pairs: {len(pmi_f)})")

    rows = []
    for v, a, b, c in pmi[:12]:
        cells = []
        for w in (a, b):
            exs = [t[0] for cs in tr_panels.values() for t in cs if t[1] == w][:4]
            pr, pk = proto[w]
            cells += [draw(P[j].cpu().numpy(), M[j].cpu().numpy(), SIZE) for j in exs]
            cells.append(draw(pr, pk, SIZE))
        while len(cells) < 10:
            cells.append(np.full((SIZE, SIZE, 3), 255, np.uint8))
        row = np.hstack(cells)
        tag = np.full((SIZE, 150, 3), 255, np.uint8)
        cv2.putText(tag, f"PMI {v:.2f}", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 2)
        cv2.putText(tag, f"{a} x {b}", (4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 2)
        rows.append(np.hstack([tag, row]))
    cv2.imwrite(str(OUT / "cooc_pairs.png"), np.vstack(rows))
    print("cooc pairs ->", OUT / "cooc_pairs.png", "(rows: PMI pair, 4+1 examples of A then 4+1 of B)")


if __name__ == "__main__":
    main()
