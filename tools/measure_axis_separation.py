#!/usr/bin/env python3
"""**腕を分ける計算可能な軸を測る**(条例1 / 門 `judge_protocol.py` の `separation_auc`)。

人に判定を頼む前に回す。軸1本だけで腕の対を分けられるなら、**その対について人に聞いても
計算で既に出ている答が返る**(lesson 11 の安い点検を、材料を作る前に当てた形)。

出すのは**対ごとの AUC**(Mann-Whitney)。**プールごとに行を分ける**(規律: プールを跨いで
平均しない)。**落としたタイルは理由別に数えて印字する**(規律9)。

使い方:
  measure_axis_separation.py --root <腕の親ディレクトリ> --arms a,b,c --pools lineart,housei
終了コード: 0 = 測れた / 2 = 走れない。**棒は判定しない** —— 判定は門が持つ。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

AXES = ("mean_ink", "near_white_frac")


def axis_values(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return {
        "mean_ink": float((255.0 - g).mean() / 255.0),
        "near_white_frac": float((g >= 250).mean()),
    }


def auc(x, y):
    """x を正、y を負としたときの AUC。0.5 = 分けられない、1.0 = 完全に分かれる。"""
    if not len(x) or not len(y):
        return None
    v = np.concatenate([x, y])
    r = v.argsort().argsort().astype(np.float64) + 1
    # 同値は平均順位に直す(同値を無視すると分離を過大に読む)
    order = np.argsort(v, kind="mergesort")
    vs = v[order]
    i = 0
    while i < len(vs):
        j = i
        while j + 1 < len(vs) and vs[j + 1] == vs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    rx = r[:len(x)].sum()
    a = (rx - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))
    return max(a, 1 - a)          # 向きは問わない。**分かれているかだけを読む**


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arms", required=True, help="腕のディレクトリ名をカンマ区切り(親からの相対)")
    ap.add_argument("--pools", default="lineart,housei")
    ap.add_argument("--csv")
    a = ap.parse_args()
    root = Path(a.root)
    arms = [s.strip() for s in a.arms.split(",")]
    pools = [s.strip() for s in a.pools.split(",")]

    vals = {}        # (pool, arm, tile) -> {axis: v}
    dropped = {}
    for arm in arms:
        d = root / arm
        if not d.is_dir():
            print(f"**腕が無い**: {d}", file=sys.stderr)
            return 2
        for f in sorted(d.glob("*.png")):
            pool = next((p for p in pools if f.name.startswith(p)), None)
            if pool is None:
                dropped["プールに属さない名前"] = dropped.get("プールに属さない名前", 0) + 1
                continue
            vals[(pool, arm, f.name)] = axis_values(f)

    for reason, n in sorted(dropped.items()):
        print(f"落とした: {reason} {n} 枚")

    rows = []
    for pool in pools:
        tiles = {t for (p, _, t) in vals if p == pool}
        print()
        print(f"══ プール {pool}  (タイル {len(tiles)} 枚)")
        for axis in AXES:
            print(f"   軸 {axis}")
            print("     腕                     中央値   " + "  ".join(f"{x[:9]:>9s}" for x in arms))
            for arm in arms:
                v = np.array([vals[(pool, arm, t)][axis] for t in sorted(tiles) if (pool, arm, t) in vals])
                cells = []
                for other in arms:
                    if other == arm:
                        cells.append("        —")
                        continue
                    w = np.array([vals[(pool, other, t)][axis] for t in sorted(tiles) if (pool, other, t) in vals])
                    s = auc(v, w)
                    cells.append("        —" if s is None else f"{s:9.3f}")
                    if s is not None:
                        rows.append((pool, axis, arm, other, s))
                print(f"     {arm:22s} {np.median(v):7.4f}  " + "  ".join(cells))
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pool", "axis", "arm_a", "arm_b", "separation_auc"])
            w.writerows(rows)
        print(f"\nCSV: {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
