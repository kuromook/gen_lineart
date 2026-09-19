#!/usr/bin/env python
"""Guard against the gate-1 failure mode before believing cloze round two.

Gate 1's 0.726 (2026-09-17) turned out to be the tokenizer's artifact: a span cut
at a junction ends where another stroke is, and displacement broke that. Here the
same worry applies: strokes are cut and linked at junctions, so a true stroke's
end may touch other ink BY CONSTRUCTION, and a distractor's end, cut in another
drawing, lands on nothing. If the model's win lives only in strokes whose ends
touch other ink, it is reading the tokenizer, not the drawing.

So: split the held-out questions by how many of the TRUE stroke's two ends touch
another stroke of its panel (within 3px), and report model and baselines per
bucket. Plus a montage of questions for the eye.
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cloze as c1
from cloze import STRATA, Corpus, Scorer, base_scores
from cloze2 import N_REL, build
from train_infill import POINTS

PACK = "results/panel_pack_20260919"
OUT = Path("results/cloze2_20260919")


def ends_touching(C, t, tol=3.0):
    p = C.panel_of[t]; s, n = int(C.rows[p]["start"]), int(C.rows[p]["n"])
    tp, _ = C.stroke(t)
    others = np.asarray(C.arr[s:s + n, :POINTS * 2]).reshape(n, POINTS, 2)
    others = np.delete(others, t - s, 0).reshape(-1, 2)
    k = 0
    for e in (tp[0], tp[-1]):
        if np.linalg.norm(others - e, axis=1).min() <= tol:
            k += 1
    return k


def main():
    C = Corpus(PACK)
    C.lab = np.load(Path(PACK) / "cluster_labels_noend.npy")
    E = [tuple(e) for e in json.load(open(OUT / "evalset_arc_1500.json"))]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT / "model_arc_rel.pt", map_location=dev, weights_only=False)
    model = Scorer(POINTS * 2 + 32 + 2 + 3 + 1 + N_REL).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    rows, examples = [], []
    buf = []

    def flush():
        X, M, Y, S = c1.collate([(b[0], b[1], 0) for b in buf])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            sc = model(X.to(dev), M.to(dev), True).float().cpu().numpy()
        for b, s in zip(buf, sc):
            rows.append({**b[2], "model": int(s.argmax() == b[1])})
            if len(examples) < 400:
                examples.append((b[3], b[4], b[1], int(s.argmax()), b[2]))
        buf.clear()

    for st, t, d, mates, seed in E:
        rng = np.random.default_rng(seed)
        x, y, cands, mp = build(C, t, np.array(d), np.array(mates), rng, True)
        g, a = base_scores(cands, mp)
        info = {"stratum": st, "touch": ends_touching(C, t), "gap": int(np.argmax(g) == y),
                "angle": int(np.argmax(a) == y)}
        buf.append((x, y, info, cands, mp))
        if len(buf) == 64:
            flush()
    if buf:
        flush()
    print(f"{'層':<12}{'真の線の端が接する数':>18}{'n':>6}{'モデル':>8}{'端点規則':>10}{'角度規則':>10}")
    for st in STRATA:
        for k in (0, 1, 2):
            sel = [r for r in rows if r["stratum"] == st and r["touch"] == k]
            if not sel:
                continue
            m = np.mean([r["model"] for r in sel]); g = np.mean([r["gap"] for r in sel]); a = np.mean([r["angle"] for r in sel])
            print(f"{st:<12}{k:>18}{len(sel):>6}{m:>8.3f}{g:>10.3f}{a:>10.3f}")
    share = np.bincount([r["touch"] for r in rows], minlength=3) / len(rows)
    print(f"真の線の端の接触: 0本 {share[0]:.1%} / 1本 {share[1]:.1%} / 2本 {share[2]:.1%}")
    # montage: 12 same_panel questions, 6 right, 6 wrong
    sp = [e for e in examples if e[4]["stratum"] == "same_panel"]
    pick = [e for e in sp if e[2] == e[3]][:6] + [e for e in sp if e[2] != e[3]][:6]
    cells = []
    for cands, mp, y, pred, info in pick:
        allp = np.concatenate([mp.reshape(-1, 2)] + [c for c in cands])
        lo = allp.min(0) - 10; span = float((allp.max(0) - lo).max()) + 10
        z = 300.0 / span
        img = np.full((320, 320, 3), 255, np.uint8)
        tf = lambda p: np.round((p - lo) * z)[:, ::-1].astype(np.int32).reshape(-1, 1, 2)
        for m in mp:
            cv2.polylines(img, [tf(m)], False, (170, 170, 170), 2)
        for k, c in enumerate(cands):
            col = (40, 170, 40) if k == y else (200, 140, 60)
            cv2.polylines(img, [tf(c)], False, col, 2 if k == y else 1)
        cv2.polylines(img, [tf(cands[pred])], False, (30, 30, 220), 1)
        cv2.putText(img, f"{'OK' if pred == y else 'NG'} touch={info['touch']}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cells.append(img)
    grid = np.vstack([np.hstack(cells[i:i + 6]) for i in range(0, 12, 6)])
    cv2.imwrite(str(OUT / "montage_cloze.png"), grid)
    print("montage ->", OUT / "montage_cloze.png", " (grey=mates, green=true, orange=distractors, red=model pick)")


if __name__ == "__main__":
    main()
