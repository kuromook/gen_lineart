#!/usr/bin/env python
"""黒い塊の切り分け (2026-09-23 事前登録どおり)。

実インスタンス描画モンタージュの「太いマジックで書きなぐったような黒い塊」が
(A) 実線画の太いインクの正しい再現か (B) 描画の幅バグかを、対象パネルの
「元の線画」「A 描画(実幅つき)」「A 描画(th=1 固定)」の局所切り出しの目視で判定する。
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trackf"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS  # noqa: E402
from train_codebook import load                                        # noqa: E402
from render_real import CLUSTERS, PACK, draw_clusters                  # noqa: E402


def panel_clusters(pid):
    """cluster_set から該当パネルのクラスタを収集(encode 不要・CPU で十分)"""
    dev = torch.device("cpu")
    P2, W2, M2, scale2, cy2, cx2 = [], [], [], [], [], []
    for split in ("train", "test"):
        Pp, Ww, Mm, rows = load(CLUSTERS, split, dev)
        hit = [k for k, r in enumerate(rows) if int(r["panel"]) == pid]
        if hit:
            P2.append(Pp[hit]); W2.append(Ww[hit]); M2.append(Mm[hit])
            for k in hit:
                scale2.append(float(rows[k]["scale"]))
                cy2.append(float(rows[k]["cy"])); cx2.append(float(rows[k]["cx"]))
    P = torch.cat(P2).numpy(); W = torch.cat(W2).numpy(); M = torch.cat(M2).numpy()
    return P, W, M, np.array(scale2), np.array(cy2), np.array(cx2)


def orig_ink(pid, strokes, panels_csv):
    """元のパネル線画をネイティブサイズの白地黒インクでレンダリング"""
    pr = panels_csv[int(pid)]
    rows_s = np.asarray(strokes[int(pr["start"]):int(pr["start"]) + int(pr["n"])])
    pp = rows_s[:, :32].reshape(-1, 16, 2)
    ink = np.abs(pp).sum(-1) > 0
    H = max(int(np.ceil(pp[..., 1][ink].max())) + 2, 2) if ink.any() else 2
    Wd = max(int(np.ceil(pp[..., 0][ink].max())) + 2, 2) if ink.any() else 2
    orig = np.full((H, Wd), 255, np.uint8)
    for s in range(pp.shape[0]):
        if not ink[s].any():
            continue
        xy = np.clip(np.round(pp[s][ink[s]]).astype(np.int32), 0, [Wd - 1, H - 1])
        cv2.polylines(orig, [xy], False, 0, 1)
    return orig


def with_border(canvas, label):
    bar = np.full((22, canvas.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1, cv2.LINE_AA)
    return np.vstack([canvas, bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[21])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--R", type=int, default=150)
    ap.add_argument("--out", default="results/real_render_20260923")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    panels_csv = list(csv.DictReader(open(PACK / "panels.csv")))
    strokes = np.load(PACK / "strokes.npy", mmap_mode="r")
    z = np.load(CORPUS)
    dims = z["dims"]

    for row in a.rows:
        pid = int(z["panels"][row])
        P, W, M, scale, cy, cx = panel_clusters(pid)
        H, Wd = int(dims[pid][0]), int(dims[pid][1])
        th = np.array([float(np.asarray(W[i])[np.asarray(M[i])].mean() if M[i].any() else 1.0) / scale[i]
                       for i in range(len(P))])
        order = np.argsort(-th)
        print(f"row {row} pid {pid} ({H}x{Wd}) clusters {len(P)} | "
              f"th med {np.median(th):.1f} p90 {np.percentile(th, 90):.1f} max {th.max():.1f}", flush=True)

        orig = orig_ink(pid, strokes, panels_csv)
        real = np.full((H, Wd, 3), 255, np.uint8)
        draw_clusters(real, P, W, M, list(range(len(P))),
                      [(cy[i], cx[i], scale[i]) for i in range(len(P))], 1.0)
        thin = np.full((H, Wd, 3), 255, np.uint8)
        draw_clusters(thin, P, W, M, list(range(len(P))),
                      [(cy[i], cx[i], scale[i]) for i in range(len(P))], 1.0, th_fix=1)

        cols = []
        for j in order[:a.topk]:
            R = a.R
            yc, xc = int(cy[j]), int(cx[j])
            def cp(img):
                y0, y1 = max(0, yc - R), min(img.shape[0], yc + R)
                x0, x1 = max(0, xc - R), min(img.shape[1], xc + R)
                c = img[y0:y1, x0:x1]
                py0, px0 = max(0, R - yc), max(0, R - xc)
                cc = cv2.copyMakeBorder(c, py0, 2 * R - c.shape[0] - py0,
                                        px0, 2 * R - c.shape[1] - px0,
                                        cv2.BORDER_CONSTANT, value=255)
                return cc if cc.ndim == 3 else cv2.cvtColor(cc, cv2.COLOR_GRAY2BGR)
            w_mean = float(np.asarray(W[j])[np.asarray(M[j])].mean() if M[j].any() else 1.0)
            lab = f"th={th[j]:.1f} sc={scale[j]:.4f} w={w_mean:.2f} @({yc},{xc})"
            stack = np.vstack([with_border(cp(orig), f"orig  {lab}"),
                               with_border(cp(real), "A real-width"),
                               with_border(cp(thin), "A th=1")])
            cols.append(stack)
            print(f"  top: {lab}", flush=True)
        m = np.hstack([np.pad(c, ((0, 0), (0, 6), (0, 0)), constant_values=200) for c in cols])
        path = out / f"blob_check_{row}_{pid}.png"
        cv2.imwrite(str(path), m)
        print("->", path, flush=True)


if __name__ == "__main__":
    main()
