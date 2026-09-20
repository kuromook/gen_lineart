#!/usr/bin/env python
"""Cloze montage: truth held-out cluster (red) vs top-4 predictions (blue prototypes).

Row per test panel: [truth | top-1 | top-2 | top-3 | top-4]. Prototypes are the
word-only decodes placed at the predicted position bin / scale bin.
build_batch returns TK (held-out cluster index) so truth is drawn exactly.
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_codebook2 import Codebook2, topn_mask
from train_cloze import ClozeNet, build_batch, unpack

CKPT_CB = "results/invariance_20260920/l4_w500/codebook2.pt"
PACK = "results/panel_pack_20260919"
H_CANVAS, N_SHOW = 220, 12
S_LO, S_HI, S_BINS = -7.0, 4.0, 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="results/grammar_corpus_20260920/corpus.npz")
    ap.add_argument("--out", default="results/cloze_20260920")
    a = ap.parse_args()
    dev = torch.device("cuda")
    z = np.load(a.data)
    seq, split, pids = z["seq"], z["split"], z["panels"]
    te, te_ids = seq[~split], pids[~split]
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cargs = ck.get("args", {})
    lab = np.load(cargs["labels"]).astype(np.int64) if cargs.get("labels") else None
    tg = np.load(cargs["tagfeats"]).astype(np.float32) if cargs.get("tagfeats") else None
    n_types = int(lab.max()) + 1 if lab is not None else 0
    n_tagf = tg.shape[1] if tg is not None else 0
    ctype_te = lab[~split] if lab is not None else None
    ctag_te = tg[~split] if tg is not None else None
    model = ClozeNet(n_types=n_types, n_tagf=n_tagf).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    cb = torch.load(CKPT_CB, map_location=dev, weights_only=False)
    levels = tuple(cb["levels"])
    cm = Codebook2(levels=levels, coord_bins=cb["args"].get("coord_bins", 0)).to(dev)
    cm.load_state_dict(cb["model"]); cm.eval(); cm.rfsq.stages = cb["args"].get("stages", 3)
    basis = np.cumprod((1,) + levels[:-1])
    half_w = np.floor(np.array(levels) / 2)
    qq = np.zeros((500, len(levels)), np.float32)
    for w in range(500):
        d = (w // basis) % np.array(levels)
        qq[w] = (d - half_w) / half_w
    with torch.no_grad():
        ex, pp, pw, cl, _lg = cm.decode(torch.from_numpy(qq).to(dev))
        keep = topn_mask(ex.float(), cl.float())
    proto = pp.float().cpu().numpy(); pkeep = keep.cpu().numpy()

    panels_csv = list(csv.DictReader(open(f"{PACK}/panels.csv")))
    strokes = np.load(f"{PACK}/strokes.npy", mmap_mode="r")

    def base_canvas(panel_id):
        pr = panels_csv[int(panel_id)]
        rows = np.asarray(strokes[int(pr["start"]):int(pr["start"]) + int(pr["n"])])
        ppn = rows[:, :32].reshape(-1, 16, 2)
        ink = np.abs(ppn).sum(-1) > 0
        H = max(int(np.ceil(ppn[..., 1][ink].max())) + 2, 2) if ink.any() else 2
        Wd = max(int(np.ceil(ppn[..., 0][ink].max())) + 2, 2) if ink.any() else 2
        zf = H_CANVAS / H
        canvas = np.full((H_CANVAS, max(int(Wd * zf), 2), 3), 255, np.uint8)
        for s in range(ppn.shape[0]):
            if not ink[s].any():
                continue
            xy = np.clip(np.round(ppn[s][ink[s]] * zf)[:, ::-1].astype(np.int32),
                         0, [canvas.shape[1] - 1, H_CANVAS - 1])
            cv2.polylines(canvas, [xy], False, (205, 205, 205), 1)
        return canvas, zf, H

    def draw_pts(canvas, zf, pts, keepmask, color, wd=2):
        for s in range(pts.shape[0]):
            if not keepmask[s]:
                continue
            xy = np.clip(np.round(pts[s] * zf)[:, ::-1].astype(np.int32),
                         0, [canvas.shape[1] - 1, H_CANVAS - 1])
            cv2.polylines(canvas, [xy], False, color, wd)
        return canvas

    def nat_place(proto_idx, py, px, sc, H, Wd):
        """prototype pts in native coords at the binned position/scale."""
        scale = 2.0 ** (S_LO + (sc + 0.5) / S_BINS * (S_HI - S_LO))
        cy = (py + 0.5) / 16 * H
        cx = (px + 0.5) / 16 * Wd
        return proto[proto_idx] / scale + np.array([cy, cx]), pkeep[proto_idx]

    shown, rows = 0, []
    with torch.no_grad():
        for i in range(0, len(te), 64):
            bt = build_batch(te[i:i + 64], np.random.default_rng(777), dev,
                             ctype_te[i:i + 64] if ctype_te is not None else None,
                             ctag_te[i:i + 64] if ctag_te is not None else None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lw, ly, lx, ls = model(*bt[:5], ctype=bt[10] if ctype_te is not None else None,
                                       ctag=bt[10 + (1 if ctype_te is not None else 0)] if ctag_te is not None else None)
            topw = lw.float().topk(4, -1).indices.cpu().numpy()
            top1y = ly.float().argmax(-1).cpu().numpy(); top1x = lx.float().argmax(-1).cpu().numpy()
            top1s = ls.float().argmax(-1).cpu().numpy()
            TW, TY, TX, TS = bt[5], bt[6], bt[7], bt[8]
            for b in range(len(TW)):
                if shown >= N_SHOW:
                    break
                row = te[i + b]
                pid = int(te_ids[i + b])
                canvas, zf, H = base_canvas(pid)
                pr = panels_csv[int(pid)]
                rows_s = np.asarray(strokes[int(pr["start"]):int(pr["start"]) + int(pr["n"])])
                pp2 = rows_s[:, :32].reshape(-1, 16, 2)
                ink2 = np.abs(pp2).sum(-1) > 0
                Wd = max(int(np.ceil(pp2[..., 0][ink2].max())) + 2, 2) if ink2.any() else 2
                # truth: prototype of the true word at the true (binned) position, red
                pts, km = nat_place(int(TW[b]), int(TY[b]), int(TX[b]), int(TS[b]), H, Wd)
                cells = [draw_pts(canvas, zf, pts, km, (30, 30, 230), 2)]
                for r in range(4):
                    c2, zf2, H2 = base_canvas(pid)
                    pts, km = nat_place(int(topw[b, r]), int(top1y[b]), int(top1x[b]), int(top1s[b]), H2, Wd)
                    cells.append(draw_pts(c2, zf2, pts, km, (230, 120, 30), 2))
                hmax = max(c.shape[1] for c in cells)
                cells = [np.pad(c, ((0, 0), (0, hmax - c.shape[1]), (0, 0)), constant_values=255) for c in cells]
                rows.append(np.hstack(cells))
                shown += 1
            if shown >= N_SHOW:
                break
    out = Path(a.out)
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(str(out / "cloze_montage.png"), np.vstack(rows))
    print("montage ->", out / "cloze_montage.png", "(columns: truth bin-box | top-1..4 predicted prototypes in blue)")


if __name__ == "__main__":
    main()
