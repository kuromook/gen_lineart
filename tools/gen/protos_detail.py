#!/usr/bin/env python
"""原型+代表残差の復号 (2026-09-23 事前登録の実験1)。
cluster_set を codebook2 で encode し、語ごとの残差(stage2+3)の平均を取る。
表示は decode(語の格子点 + その語の平均残差)。
合否は目視: rich_check.png で「蟻の群れ」から脱却しているか (ユーザーと確認)。
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trackf"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TRACKF, CODEBOOK, N_WORDS, load_protos, nat_place   # noqa: E402
from train_codebook import load                                        # noqa: E402
from train_codebook2 import Codebook2, topn_mask                       # noqa: E402

CLUSTERS = TRACKF / "results/cluster_set_20260919"
OUT = Path("results/protos_detail_20260923")
H_CANVAS = 220


def word_grid(levels):
    basis = np.cumprod((1,) + levels[:-1])
    half_w = np.floor(np.array(levels) / 2)
    qq = np.zeros((N_WORDS, len(levels)), np.float32)
    for w in range(N_WORDS):
        d = (w // basis) % np.array(levels)
        qq[w] = (d - half_w) / half_w
    return qq


def draw(canvas_shape, pts, km, zf, H):
    c = np.full(canvas_shape, 255, np.uint8)
    for s in range(pts.shape[0]):
        if not km[s]:
            continue
        xy = np.clip(np.round(pts[s] * zf)[:, ::-1].astype(np.int32), 0, H - 1)
        cv2.polylines(c, [xy], False, (40, 40, 40), 1)
    return c


def main():
    dev = torch.device("cuda")
    cb = torch.load(CODEBOOK, map_location=dev, weights_only=False)
    levels = tuple(cb["levels"])
    cm = Codebook2(levels=levels, coord_bins=cb["args"].get("coord_bins", 0)).to(dev)
    cm.load_state_dict(cb["model"]); cm.eval(); cm.rfsq.stages = cb["args"].get("stages", 3)

    res_sum = torch.zeros(N_WORDS, len(levels), dtype=torch.float64)
    cnt = np.zeros(N_WORDS, np.int64)
    arc_true, arc_word_sum = [], []
    with torch.no_grad():
        for split in ("train", "test"):
            P, W, M, rows = load(CLUSTERS, split, dev)
            for i in range(0, len(P), 4096):
                p, w, m = P[i:i + 4096].float(), W[i:i + 4096].float(), M[i:i + 4096]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    q_full, _wf = cm.encode(p, w, m, None)
                    q_word, wi = cm.encode(p, w, m, 1)
                r = (q_full.float() - q_word.float()).cpu().double()
                wi = wi.cpu()
                res_sum.index_add_(0, wi, r)
                cnt += np.bincount(wi.numpy(), minlength=N_WORDS)
                arc_true.append((p[:, :, 1:] - p[:, :, :-1]).norm(dim=-1).sum(-1)[m].cpu())
    mean_res = (res_sum / torch.from_numpy(np.maximum(cnt, 1)).double().unsqueeze(1)).numpy().astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "mean_residual.npy", mean_res)
    np.save(OUT / "count.npy", cnt)

    # --- 道具確認: 残差ノルム分布 ---
    norms = np.linalg.norm(mean_res, axis=1)
    used = cnt > 0
    print(f"語 {int(used.sum())}/500 に残差あり | 平均残差ノルム: 中央 {np.median(norms[used]):.3f} "
          f"最大 {norms[used].max():.3f} (1 語あたり stage2+3 の量子化単位)")
    stats = dict(words_with_residual=int(used.sum()),
                 residual_norm_median=float(np.median(norms[used])),
                 residual_norm_max=float(norms[used].max()))

    # --- 復号モンタージュ: word-only vs +代表残差 ---
    qq = torch.from_numpy(word_grid(levels)).to(dev)
    with torch.no_grad():
        ex1, pp1, _pw1, cl1, _ = cm.decode(qq)
        keep1 = topn_mask(ex1.float(), cl1.float())
        ex2, pp2, _pw2, cl2, _ = cm.decode(qq + torch.from_numpy(mean_res).to(dev))
        keep2 = topn_mask(ex2.float(), cl2.float())
    proto1, keep1 = pp1.float().cpu().numpy(), keep1.cpu().numpy()
    proto2, keep2 = pp2.float().cpu().numpy(), keep2.cpu().numpy()

    ids = list(np.argsort(-cnt)[:6]) + [37, 251, 409]
    cells = []
    for wi in ids:
        H = Wd = 224
        c1, _ = nat_place(proto1, keep1, wi, 7, 7, 5, H, Wd)
        c2, _ = nat_place(proto2, keep2, wi, 7, 7, 5, H, Wd)
        zf = H_CANVAS / H
        a = draw((H_CANVAS, H_CANVAS, 3), c1, keep1[wi], zf, H_CANVAS)
        b = draw((H_CANVAS, H_CANVAS, 3), c2, keep2[wi], zf, H_CANVAS)
        cv2.putText(a, f"w{wi} word-only", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
        cv2.putText(b, f"w{wi} +residual", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
        cells.append(np.vstack([a, b]))
    cv2.imwrite(str(OUT / "rich_check.png"), np.hstack(cells))
    stats["strokes_word_only"] = [int(keep1[i].sum()) for i in ids]
    stats["strokes_rich"] = [int(keep2[i].sum()) for i in ids]
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
    print("rich_check.png ->", OUT / "rich_check.png")


if __name__ == "__main__":
    main()
