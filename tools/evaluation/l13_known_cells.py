# -*- coding: utf-8 -*-
"""L13 既知セル。これが PASS しなければ本体は起動しない(L12 の作法の失敗の修理)。

KC6 器具: 3px リングで囲んだ円盤 -> support 1.00 / open_runs 0
          同じリングに 10px の切れ目 -> support ~ 1-10/周長 / open_runs 1
KC7 データ経路: truth4 の連結成分(>=500px)の数が、**コマごとに** L12 の per_panel.csv の
          n_gt と一致する(248コマ)。向きは両方、粒度はコマ。
          **2026-10-25 の訂正**: 以前は**合計 7,400 = 7,400** しか見ていなかった。
          2コマが +1 / -1 でずれても合計は通る(罠 T101 の「要約を中身と呼ぶ」形)。
          **同日発見**: `L13_TRUTH` が指す t4 画像は**ディスク上のどこにも無い**
          (`find /home/sh1/deepl /home/sh1/disk -name '*_t4.png'` が0件)。
          **この既知セルは現状回せない。** 以前の走行が残した定数は生きている ——
          `results/l12_panel_regions/page_truth.csv`(65頁の `t4_px`)と
          `per_panel.csv`(248コマの `n_gt`)。L12 を回し直すときは**この2本と
          コマ単位で照合する**(合計ではなく)。画像は `l12_panel_regions.py` の段1 が作る。
"""
import csv, os, sys
import numpy as np, cv2

TRUTH = os.environ.get('L13_TRUTH', '')   # 無いときは KC7 が理由を印字して FAIL する。
                                          # KeyError で import が死ぬと、**点検が無いのか
                                          # 環境が無いのかが呼び出し側から見えない**
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
    if not TRUTH or not os.path.isdir(TRUTH):
        print('KC7 回せない: L13_TRUTH が無い(t4 画像はディスク上に存在しない)。'
              '以前の走行の定数は page_truth.csv / per_panel.csv に在る')
        return False
    man = {r['name']: r for r in csv.DictReader(open(f'{KOMA}/manifest.csv'))}
    per = []          # (コマ名, 自前, L12 の n_gt)
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
        mine = sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MINREG)
        per.append((r['name'], mine, int(r['n_gt'])))
    off = [(nm, a, b) for nm, a, b in per if a != b]
    got, want = sum(a for _, a, _ in per), sum(b for _, _, b in per)
    good = not off
    print(f'KC7 データ経路: **コマ単位** 一致 {len(per) - len(off)}/{len(per)}  '
          f'(合計は {got} / {want} —— **合計の一致は中身の一致ではない**)  '
          f'{"PASS" if good else "FAIL"}')
    for nm, a, b in off[:5]:
        print(f'    ずれ {nm}: 自前 {a} / L12 {b}')
    return good


if __name__ == '__main__':
    a, b = kc6(), kc7()
    print('\n既知セル', 'PASS' if (a and b) else 'FAIL')
    sys.exit(0 if (a and b) else 1)
