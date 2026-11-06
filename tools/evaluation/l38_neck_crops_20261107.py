"""**首で繋がっている面**を原寸 1:1 で切り出す(作者の裁定を仰ぐため)。

作者(2026-11-06): **「一部レイヤが不自然な結合をしているものもあった」**。
`l38_merge_cause_20261106.py` で **面の 12.0%(136/1,130)が細い首で繋がっている**と出た。
**割るべきかどうかは作者が決める**(規律6)。この器具は**判断の材料を原寸で出すだけ**で、
数も出さない。

**縮小したモンタージュで論じない**(規律6 / 2026-10-24 の方針)。だから:

- **`_face.png`** —— 面の外接矩形を**原寸 1:1**。面の画素を黒、同じコマの他の面を淡い灰
  (**どこまでが1つの面か**が見える)
- **`_neck.png`** —— **首の周りだけを原寸 1:1** で窓に切る(既定 700x700)。
  首の場所は計算で出す: **割れた半径で侵食 → 残った2片をそれぞれ膨らませ、
  面の中で両方に触る画素**が首である。印は付けない(画素そのものを見せる)

**首の場所の出し方は推定である。** 面と窓の画素は原寸のままなので、
**作者が見ているのは器具の判断ではなく原稿の画素**である。
"""
import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_arms import fill_mask                                   # noqa: E402
from l14_known_cells import A, PAD, comps                        # noqa: E402
from l17_known_cells import (DEL, MAX_FRAC, MIN_FRAC, arms,      # noqa: E402
                             jobs, zmap)

ROOT = '/home/sh1/deepl/lineart-fill-region'
SRC = f'{ROOT}/results/l38_merge_cause_20261106.json'
OUT = f'{ROOT}/results/l38_neck_crops_20261107'
MINPART = 500


def page_masks(fz):
    out, shape = [], None
    for lid, path in fz:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        ink = im < 128
        if shape is None:
            shape = ink.shape
        elif ink.shape != shape or not ink.any():
            continue
        if not ink.any():
            continue
        ys, xs = np.nonzero(ink)
        y0, y1 = max(0, ys.min() - PAD), min(shape[0], ys.max() + 1 + PAD)
        x0, x1 = max(0, xs.min() - PAD), min(shape[1], xs.max() + 1 + PAD)
        m = np.zeros(shape, bool)
        m[y0:y1, x0:x1] = fill_mask(ink[y0:y1, x0:x1], radius=4)
        out.append((int(lid), m))
    return out, shape


def neck_where(R, r):
    """割れた半径 `r` で侵食し、**残った片の膨張が面の中で出会う画素**を首として返す。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    e = cv2.erode(R.astype(np.uint8), k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(e, 8)
    big = [i for i in range(1, n) if st[i, A] >= MINPART]
    if len(big) < 2:
        return None
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * r + 3, 4 * r + 3))
    hit = np.zeros(R.shape, np.uint8)
    for i in big:
        hit += cv2.dilate((lab == i).astype(np.uint8), kd)
    return (hit >= 2) & R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-neck', type=int, default=2, help='首の半径の上限(既定 2 = 最も細い)')
    ap.add_argument('--n', type=int, default=2, help='出す面の数')
    ap.add_argument('--win', type=int, default=700, help='首の窓の一辺(原寸の画素)')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = [r for r in json.load(open(SRC, encoding='utf-8'))
            if r['neck'] is not None and r['neck'] <= a.max_neck]
    rows.sort(key=lambda r: -r['area'])
    # **同じ頁・同じ層に偏らせない**(1つの事故を n 回見せても材料にならない)
    pick, seen = [], set()
    for r in rows:
        k = (r['work'], r['pid'], r['top_lid'])
        if k in seen:
            continue
        seen.add(k)
        pick.append(r)
        if len(pick) >= a.n:
            break
    print(f'首が半径 {a.max_neck} 以下の面 {len(rows)} / そのうち出すのは **{len(pick)}**'
          f'(頁と層が重ならないように上から取った。面積の大きい順)', flush=True)
    for r in pick:
        print(f"  {r['work']} {r['pid']} comp={r['comp']} region={r['region']} "
              f"面積 {r['area']:,} 首の半径 {r['neck']} layer_id={r['top_lid']}")

    want = {(r['work'], r['pid']) for r in pick}
    zz = zmap()
    fill, line = arms('gated')
    js = [j for j in jobs(zz, fill, line) if (j[1], j[0]) in want]
    print(f'\n走る頁 {len(js)}', flush=True)
    made = 0
    skip = collections.Counter()
    for job in js:
        pid, work, fz, lz, sel = job
        lm, shape = page_masks(fz)
        if shape is None:
            continue
        F = np.zeros(shape, bool)
        for _, m in lm:
            F |= m
        H, W = shape
        for fr in sel:
            km = cv2.imread(f'{DEL}/{work}/koma/{fr["file"]}', cv2.IMREAD_GRAYSCALE)
            if km is None or km.shape != (H, W):
                continue
            n, lab, st, _ = cv2.connectedComponentsWithStats((km > 127).astype(np.uint8), 8)
            for ci in range(1, n):
                ar = int(st[ci, A])
                if ar < MIN_FRAC * H * W or ar > MAX_FRAC * H * W:
                    continue
                y0 = max(0, st[ci, cv2.CC_STAT_TOP] - 2)
                x0 = max(0, st[ci, cv2.CC_STAT_LEFT] - 2)
                y1 = min(H, y0 + st[ci, cv2.CC_STAT_HEIGHT] + 4)
                x1 = min(W, x0 + st[ci, cv2.CC_STAT_WIDTH] + 4)
                P = (lab[y0:y1, x0:x1] == ci)
                glab, gids, gstat = comps(F[y0:y1, x0:x1] & P, 8)
                for r in pick:
                    if (r['work'], r['pid'], r['comp']) != (work, pid, ci):
                        continue
                    if r['region'] not in gids:
                        continue
                    R = glab == r['region']
                    # **`(work, pid, comp)` は面を同定しない。** `comp`(= `ci`)は
                    # 枠ファイルごとに 1 から振り直されるので、**同じ `comp` を持つ別のコマの
                    # 同じ `region` 番号にも当たる**(2026-11-07、最初の版は 2 面を頼んで
                    # **4 回書こうとした**。同じ衝突を同じ日に3度踏んだ。条例4)。
                    # **面積で同定する** —— 鍵に依らない同一性で、行が持っている値である
                    if int(R.sum()) != r['area']:
                        skip['面積が合わない(`comp` が同じ別のコマ)'] += 1
                        continue
                    other = (glab > 0) & ~R
                    # ---- 面の外接矩形を原寸で
                    ys, xs = np.nonzero(R)
                    by0, by1 = ys.min(), ys.max() + 1
                    bx0, bx1 = xs.min(), xs.max() + 1
                    img = np.full((by1 - by0, bx1 - bx0, 3), 255, np.uint8)
                    img[other[by0:by1, bx0:bx1]] = (225, 225, 225)
                    img[R[by0:by1, bx0:bx1]] = (0, 0, 0)
                    base = (f"{work}_{pid.split('_', 3)[-1]}_c{ci}_r{r['region']}"
                            f"_area{r['area']}_neck{r['neck']}")
                    cv2.imwrite(f'{OUT}/{base}_face.png', img)
                    # ---- 首の周りを原寸で
                    nk = neck_where(R, r['neck'])
                    if nk is not None and nk.any():
                        ny, nx = np.nonzero(nk)
                        cy, cx = int(ny.mean()), int(nx.mean())
                        h = a.win // 2
                        wy0, wy1 = max(0, cy - h), min(R.shape[0], cy + h)
                        wx0, wx1 = max(0, cx - h), min(R.shape[1], cx + h)
                        w = np.full((wy1 - wy0, wx1 - wx0, 3), 255, np.uint8)
                        w[other[wy0:wy1, wx0:wx1]] = (225, 225, 225)
                        w[R[wy0:wy1, wx0:wx1]] = (0, 0, 0)
                        cv2.imwrite(f'{OUT}/{base}_neck.png', w)
                        print(f'  書いた {base}  面 {bx1-bx0}x{by1-by0} / '
                              f'首の窓 {wx1-wx0}x{wy1-wy0}(原寸)', flush=True)
                    else:
                        print(f'  **首の場所が出せなかった**: {base}', flush=True)
                    made += 1
    print(f'\n出した面 {made} / 頼んだ {len(pick)}  -> {OUT}')
    print('落としたもの(理由別。規律9):')
    for k, v in (skip.most_common() or [('(衝突は起きなかった)', 0)]):
        print(f'  {v}  {k}')
    if made != len(pick):
        print('**頼んだ数と出した数が違う。止める**')
        return 1


if __name__ == '__main__':
    main()
