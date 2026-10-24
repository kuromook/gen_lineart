# -*- coding: utf-8 -*-
"""L13 既知セル。これが PASS しなければ本体は起動しない(L12 の作法の失敗の修理)。

KC6 器具: 3px リングで囲んだ円盤 -> support 1.00 / open_runs 0
          同じリングに 10px の切れ目 -> support ~ 1-10/周長 / open_runs 1
KC7 データ経路: truth4 の連結成分(>=500px)の総数が L12 の per_panel.csv の n_gt 合計と一致
"""
import csv, os, sys
import numpy as np, cv2

TRUTH = os.environ['L13_TRUTH']
KOMA  = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
L12   = '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions/per_panel.csv'
MINREG, SUP_PX = 500, 2


def boundary(m):
    """4近傍の収縮との差 = 輪郭画素(領域の内側の1画素)。"""
    k = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    er = cv2.erode(m.astype(np.uint8), k, iterations=1).astype(bool)
    return m & ~er


def dist_to(barrier):
    """障壁までの距離。コマごとに1回だけ作って使い回す(領域ごとに作ると遅い)。"""
    if not barrier.any():
        return None
    return cv2.distanceTransform((~barrier).astype(np.uint8), cv2.DIST_L2, 3)


def support_of(region, barrier, sup_px=SUP_PX, dist=None):
    b = boundary(region)
    nb = int(b.sum())
    if nb == 0:
        return None
    d = dist_to(barrier) if dist is None else dist
    sup = (b & (d <= sup_px)) if d is not None else np.zeros_like(b)
    unsup = b & ~sup
    n, _ = cv2.connectedComponents(unsup.astype(np.uint8), connectivity=8)
    return dict(bnd_px=nb, sup_px_n=int(sup.sum()), support=int(sup.sum()) / nb,
                open_runs=max(n - 1, 0),
                bite=int((region & barrier).sum()) / nb)


def kc6():
    ok = True
    H = 400
    yy, xx = np.ogrid[:H, :H]
    r2 = (yy - 200) ** 2 + (xx - 200) ** 2
    disk = r2 <= 100 ** 2
    ring = (r2 <= 103 ** 2) & (r2 > 100 ** 2)          # 3px のリング
    s = support_of(disk, ring)
    good = s['support'] > 0.99 and s['open_runs'] == 0
    ok &= good
    print(f"KC6a 閉じたリング: support {s['support']:.4f} (>0.99) / open_runs "
          f"{s['open_runs']} (==0) / 輪郭 {s['bnd_px']}px  {'PASS' if good else 'FAIL'}")
    ring2 = ring.copy()
    ring2[195:205, 290:310] = False                    # 10px 幅の切れ目を1つ
    s2 = support_of(disk, ring2)
    exp = 1 - 10.0 / s['bnd_px']
    good2 = abs(s2['support'] - exp) < 0.02 and s2['open_runs'] == 1
    ok &= good2
    print(f"KC6b 10px の切れ目: support {s2['support']:.4f} (期待 {exp:.4f} ±0.02) / "
          f"open_runs {s2['open_runs']} (==1)  {'PASS' if good2 else 'FAIL'}")
    return ok


def kc7():
    want = sum(int(r['n_gt']) for r in csv.DictReader(open(L12)))
    man = {r['name']: r for r in csv.DictReader(open(f'{KOMA}/manifest.csv'))}
    got = 0; npanel = 0
    cache = {}
    for r in csv.DictReader(open(L12)):
        m = man[r['name']]
        pid = m['pid'].replace('__', '_'); w = r['work']
        if pid not in cache:
            cache.clear()
            cache[pid] = cv2.imread(os.path.join(TRUTH, f'{w}_{pid}_t4.png'),
                                    cv2.IMREAD_GRAYSCALE) > 127
        t = cache[pid]
        x0, y0, x1, y1 = (int(m[k]) for k in ('x0', 'y0', 'x1', 'y1'))
        sub = t[y0:y1, x0:x1]
        n, _, st, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), 8)
        got += sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MINREG)
        npanel += 1
    good = got == want
    print(f'KC7 データ経路: truth4 の面(>={MINREG}px) {got} / L12 の n_gt 合計 {want}'
          f'  ({npanel}コマ)  {"PASS" if good else "FAIL"}')
    return good


if __name__ == '__main__':
    a, b = kc6(), kc7()
    print('\n既知セル', 'PASS' if (a and b) else 'FAIL')
    sys.exit(0 if (a and b) else 1)
