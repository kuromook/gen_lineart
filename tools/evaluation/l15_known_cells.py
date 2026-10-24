# -*- coding: utf-8 -*-
"""L15 既知セル。PASS しなければ本体は起動しない(登録 §5)。"""
import csv, os, sys
import numpy as np, cv2
from skimage.morphology import skeletonize

SUP_PX, END_PX = 2, 8
K4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
K8 = np.ones((3, 3), np.uint8)


def endpoints(barrier):
    """線の端点: 細線化した骨格のうち、骨格の8近傍が1個だけの画素。"""
    if not barrier.any():
        return np.zeros_like(barrier)
    sk = skeletonize(barrier)
    n = cv2.filter2D(sk.astype(np.uint8), -1, K8, borderType=cv2.BORDER_CONSTANT)
    return sk & (n == 2)        # 自分自身を含むので 2 = 近傍1個


def boundary(m):
    return m & ~cv2.erode(m.astype(np.uint8), K4, 1).astype(bool)


def runs_of(R, barrier, sup_px=SUP_PX, end_px=END_PX, d=None, de=None):
    """開いた区間を取り、E/H/F/O に分類する。登録 §3 の定義そのまま。"""
    b = boundary(R)
    nb = int(b.sum())
    if nb == 0:
        return []
    if d is None:
        d = (cv2.distanceTransform((~barrier).astype(np.uint8), cv2.DIST_L2, 3)
             if barrier.any() else np.full(R.shape, 1e9, np.float32))
    if de is None:
        e = endpoints(barrier)
        de = (cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 3)
              if e.any() else np.full(R.shape, 1e9, np.float32))
    op = b & (d > sup_px)
    sup = b & ~op
    n, lab = cv2.connectedComponents(op.astype(np.uint8), 8)
    supd = cv2.dilate(sup.astype(np.uint8), K8, 1).astype(bool)
    out = []
    for k in range(1, n):
        u = lab == k
        mouth = u & supd
        ys, xs = np.nonzero(mouth)
        rec = dict(length=int(u.sum()), n_mouth=int(mouth.sum()), bnd_px=nb)
        if len(ys) == 0:
            rec.update(kind='O', near=0, width=0.0)
            out.append(rec); continue
        near = int((de[ys, xs] <= end_px).sum())
        if len(ys) > 1:
            P = np.stack([ys, xs], 1).astype(np.float64)
            dd = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
            w = float(np.sqrt(dd.max()))
        else:
            w = 0.0
        rec.update(kind=('E' if near == len(ys) else ('F' if near == 0 else 'H')),
                   near=near, width=round(w, 1))
        out.append(rec)
    return out


def _ring(H=400, cy=200, cx=200, r=100, t=3):
    yy, xx = np.ogrid[:H, :H]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (r2 <= r ** 2), ((r2 <= (r + t) ** 2) & (r2 > r ** 2))


def kc12():
    disk, ring = _ring()
    u0 = runs_of(disk, ring)
    a = len(u0) == 0
    print(f'KC12a 切れ目の無いリング: 区間 {len(u0)} (==0)  {"PASS" if a else "FAIL"}')
    g = ring.copy(); g[195:205, 290:310] = False
    u = runs_of(disk, g)
    b = len(u) == 1 and u[0]['kind'] == 'E' and abs(u[0]['width'] - 10) <= 3
    print(f'KC12b 10px の切れ目: 区間 {len(u)} 分類 '
          f'{u[0]["kind"] if u else "-"} 幅 {u[0]["width"] if u else 0} (E / 10±3)  '
          f'{"PASS" if b else "FAIL"}')
    return a and b


def kc13():
    disk, ring = _ring()
    half = ring.copy(); half[:, :200] = False          # 片側180度を消す
    u = runs_of(disk, half)
    big = max(u, key=lambda r: r['length']) if u else None
    a = big is not None and big['kind'] == 'E'
    print(f'KC13a 半分消したリング: 最長区間 {big["length"] if big else 0}px 分類 '
          f'{big["kind"] if big else "-"} (E)  {"PASS" if a else "FAIL"}')
    far = np.zeros((400, 400), bool); far[0:3, :] = True   # 障壁を遠くに置く
    u2 = runs_of(disk, far)
    b = len(u2) == 1 and u2[0]['kind'] == 'O'
    print(f'KC13b 障壁が遠い円盤: 区間 {len(u2)} 分類 '
          f'{u2[0]["kind"] if u2 else "-"} (O)  {"PASS" if b else "FAIL"}')
    return a and b


def kc14():
    l = np.zeros((200, 200), bool); l[100:103, 40:140] = True
    n1 = int(endpoints(l).sum())
    _, ring = _ring()
    n2 = int(endpoints(ring).sum())
    good = n1 == 2 and n2 == 0
    print(f'KC14 端点検出: 直線 {n1} (==2) / 閉じたリング {n2} (==0)  '
          f'{"PASS" if good else "FAIL"}')
    return good


if __name__ == '__main__':
    a, b, c = kc12(), kc13(), kc14()
    print('\n既知セル(合成)', 'PASS' if (a and b and c) else 'FAIL')
    sys.exit(0 if (a and b and c) else 1)
