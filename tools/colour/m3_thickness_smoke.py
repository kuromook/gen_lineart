"""M3: B5(太さ)の量が計算できて安いかの smoke test。

作者の判断 2026-10-24: **カラーの作品に線画層はあり、線画自身に色がついている
ケースもある。** ゆえに「彩度がある」ではベース層と線画層を分けられない。
分けるなら形の軸が要る —— **面は内接半径が大きく、線は小さい。**

    thick(層) := 描画画素の距離変換の最大値の分布
                 (drawn = 画素値 < 255、8近傍の外側を背景として L2 距離変換)
    報告する量: 描画画素の半径の中央値 r50 と p90、最大内接半径 rmax

**これはモノクロのプールで走らせる smoke test であって、カラーの測定ではない。**
ako6 の制作ベタ層は**面であることが制作側で確かめられている**唯一の集合なので、
「量が面に対して大きく出るか」と「原寸でいくらかかるか」を見るために使う。
**ここで出る数値をカラーの主張に持ち込まない**(プールを跨いで平均しない)。

既知セル(走らせる前に書いた): 半径 r の円盤は rmax = r、幅 w の帯は rmax = w/2。
"""
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ADD = Path('/home/sh1/deepl/lineart/data/incoming_manuscript')


def thickness(drawn):
    """描画画素の距離変換。drawn は bool。返り値は (r50, p90, rmax, px)。"""
    d = cv2.distanceTransform(drawn.astype(np.uint8), cv2.DIST_L2, 5)
    v = d[drawn]
    if v.size == 0:
        return (0.0, 0.0, 0.0, 0)
    return (float(np.median(v)), float(np.percentile(v, 90)),
            float(v.max()), int(v.size))


def known_cell():
    """円盤と帯。期待値は上に書いてある。"""
    ok = True
    img = np.zeros((400, 400), bool)
    yy, xx = np.ogrid[:400, :400]
    img[(yy - 200) ** 2 + (xx - 200) ** 2 <= 60 * 60] = True
    r50, p90, rmax, _ = thickness(img)
    good = abs(rmax - 60) <= 1.0
    ok &= good
    print(f'  円盤 r=60: rmax {rmax:.2f} (期待 60) r50 {r50:.2f} '
          f'-> {"PASS" if good else "FAIL"}')
    band = np.zeros((400, 400), bool)
    band[200:209, 50:350] = True          # 幅 9px
    r50b, _, rmaxb, _ = thickness(band)
    goodb = abs(rmaxb - 4.5) <= 0.6
    ok &= goodb
    print(f'  帯 w=9: rmax {rmaxb:.2f} (期待 4.5) r50 {r50b:.2f} '
          f'-> {"PASS" if goodb else "FAIL"}')
    return ok


def run(paths, label, limit, seed=0):
    # 先頭を取ると最初のページに偏るので、種を固定して全体から抽出する
    paths = list(paths)
    if limit < len(paths):
        idx = np.random.default_rng(seed).choice(len(paths), limit, replace=False)
        paths = [paths[int(i)] for i in sorted(idx)]
    rows = []
    t0 = time.time()
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        drawn = im < 255
        if not drawn.any():
            continue
        r50, p90, rmax, px = thickness(drawn)
        rows.append((Path(p).name, r50, p90, rmax, px, px / drawn.size))
    dt = time.time() - t0
    if not rows:
        print(f'{label}: 0 層')
        return rows
    a = np.array([[r[1], r[2], r[3], r[5]] for r in rows])
    print(f'\n{label}: {len(rows)} 層(全 {len(paths)} 中、種固定の無作為抽出), '
          f'{dt:.1f}s ({dt/len(rows):.2f}s/層, 原寸 7016x6543)')
    for i, name in enumerate(['r50', 'p90', 'rmax', 'drawn_ratio']):
        q = np.percentile(a[:, i], [10, 50, 90])
        print(f'  {name:<12} p10 {q[0]:8.4f}  中央 {q[1]:8.4f}  p90 {q[2]:8.4f}')
    return rows


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print('既知セル:')
    if not known_cell():
        print('再現しないので他を読まない')
        return 1

    fill = sorted(glob.glob(str(ADD / 'ako6/fill/*_beta[0-9]*.png')))
    print(f'\nako6/fill の層 PNG: {len(fill)} 枚')
    rf = run(fill, 'ako6 制作ベタ層(制作側が面と確かめた集合)', limit)

    # part_layers: 名前は選別器ではないので、全部まとめて1行で出す
    pl = sorted(glob.glob(str(ADD / 'part_layers/*_L[0-9]*.png')))
    print(f'\npart_layers の層 PNG: {len(pl)} 枚')
    rp = run(pl, 'part_layers 5ページの全層(名前で選別しない)', limit, seed=1)

    # 名前は falsifier として後ろに置く: 選別ではなく不一致の印字
    names = {}
    for j in glob.glob(str(ADD / 'part_layers/*_layers.json')):
        d = json.load(open(j, encoding='utf-8'))
        for l in d['layers']:
            names[l['file']] = l['name']
    if rp:
        line = [r for r in rp if names.get(r[0], '').startswith('線画')]
        other = [r for r in rp if not names.get(r[0], '').startswith('線画')]
        print('\n  名前での内訳(選別ではなく、量が名前とどれだけ食い違うかの印字):')
        for lab, g in (('名前が 線画*', line), ('それ以外', other)):
            if g:
                v = np.array([r[3] for r in g])
                print(f'    {lab:<12} {len(g):3d} 層  rmax 中央 {np.median(v):7.3f}  '
                      f'p10 {np.percentile(v,10):7.3f}  p90 {np.percentile(v,90):7.3f}')
    print('\n**この数値はモノクロのプールのもの。カラーの主張に持ち込まない。**')
    return 0


if __name__ == '__main__':
    sys.exit(main())
