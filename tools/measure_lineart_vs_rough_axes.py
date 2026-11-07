#!/usr/bin/env python3
"""**作者が分けた2群**(線画として使える / 使えない = 下絵)の間で、**墨の量以外に何が違うか**を測る。

なぜ: 2026-11-07 の梯子は、GT の墨を 0〜70% 落としても判定が動かないことを示した
(全段 15/15)。**量の軸は境目を跨がない。** 作者の一言は2件とも「**下絵**」で、
**種別**を指しているように読める。**次の落とし方はこの測定から作る**(勝手に決めない)。

測る軸(どれも**量とは独立になるように正規化する**):
  ink_frac        墨の割合(量そのもの。対照として置く)
  mean_darkness   墨の画素の平均の濃さ ——「薄い」かどうか
  width_mean      線の太さ = 墨の総量 ÷ 骨格の長さ
  width_cv        **太さのばらつき**(標準偏差 ÷ 平均)——「一定の太さで引かれているか」
  comp_per_len    骨格の長さあたりの連結成分の数 —— 断片か、続いているか
  junction_den    骨格の長さあたりの分岐点 —— 重ね描き・交差の多さ
  stroke_den      骨格の長さ ÷ 画像の面積 —— **描いた線の総延長**(量とは別の量)

**同じタイルで GT とラフを対にして測る**(タイルの違いを打ち消す)。
印字は行ごと。落としたタイルは理由別に数える(規律9)。
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
GT = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line")
ROUGH = ROOT / "results/trackc_judge_20261004/stroke_projected/rough"
TILES = Path("/home/sh1/deepl/lineart/dataset/pairs_480/holdout_lineart_family.txt")
INK = 242


def axes(g):
    m = g < INK
    if m.sum() == 0:
        return None
    mass = float(((255.0 - g) / 255.0)[m].sum())
    sk = skeletonize(m)
    L = float(sk.sum())
    if L < 50:
        return None
    lab, n = ndimage.label(m, structure=np.ones((3, 3), int))
    # 太さ: 骨格の各点での距離変換 ×2
    dist = ndimage.distance_transform_edt(m)
    w = dist[sk] * 2.0
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    junc = float(((nb >= 3) & sk).sum())
    return {
        "ink_frac": float(m.mean()),
        "mean_darkness": float(((255.0 - g) / 255.0)[m].mean()),
        "width_mean": float(w.mean()),
        "width_cv": float(w.std() / w.mean()) if w.mean() else 0.0,
        "comp_per_len": n / L * 100,
        "junction_den": junc / L * 100,
        "stroke_den": L / g.size * 100,
        "_mass": mass,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/judge_damage_ladder_20261107/lineart_vs_rough_axes.csv"))
    a = ap.parse_args()
    tiles = [Path(l.strip()).stem for l in open(TILES) if l.strip()]
    drop = {"GT が無い": 0, "ラフが無い": 0, "骨格が短すぎる": 0}
    rows = []
    for t in tiles:
        pg, pr = GT / f"{t}.jpg", ROUGH / f"{t}_out.png"
        if not pg.is_file():
            drop["GT が無い"] += 1; continue
        if not pr.is_file():
            drop["ラフが無い"] += 1; continue
        ag = axes(np.asarray(Image.open(pg).convert("L").resize((480, 480)), float))
        ar = axes(np.asarray(Image.open(pr).convert("L"), float))
        if ag is None or ar is None:
            drop["骨格が短すぎる"] += 1; continue
        rows.append((t, ag, ar))
    print(f"対になったタイル: {len(rows)} / {len(tiles)}")
    print("落としたもの(理由別、規律9):")
    for k, v in drop.items():
        print(f"  {k}: {v}")
    keys = ["ink_frac", "mean_darkness", "width_mean", "width_cv",
            "comp_per_len", "junction_den", "stroke_den"]
    print(f"\n   {'軸':14s} {'GT(線画) 中央値':>16s} {'ラフ(下絵) 中央値':>18s} {'比':>8s} "
          f"{'重なり':>8s}")
    for k in keys:
        vg = np.array([r[1][k] for r in rows]); vr = np.array([r[2][k] for r in rows])
        # 重なり = 2群の値域がどれだけ被るか(1.0 = 完全に被る、0 = 全く被らない)
        lo, hi = max(vg.min(), vr.min()), min(vg.max(), vr.max())
        span = max(vg.max(), vr.max()) - min(vg.min(), vr.min())
        ov = max(0.0, hi - lo) / span if span else 1.0
        ratio = np.median(vr) / np.median(vg) if np.median(vg) else float("nan")
        mark = "  **被らない**" if ov < 0.05 else ("  被りが小さい" if ov < 0.25 else "")
        print(f"   {k:14s} {np.median(vg):>16.4f} {np.median(vr):>18.4f} {ratio:>8.2f} {ov:>8.3f}{mark}")
    print("\n   **比 = ラフ ÷ GT の中央値。** 1 から離れるほど、その軸で2群が違う")
    print("   **重なり = 2群の値域の被り**(0 = どのタイルでも分かれる)。**量の軸とそうでない軸を一緒に読まない**")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tile", "side"] + keys)
        for t, ag, ar in rows:
            w.writerow([t, "gt"] + [f"{ag[k]:.6f}" for k in keys])
            w.writerow([t, "rough"] + [f"{ar[k]:.6f}" for k in keys])
    print(f"\n   CSV: {a.out}")


if __name__ == "__main__":
    main()
