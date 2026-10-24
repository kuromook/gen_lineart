"""M4: 濃淡(グラデーション)層が export でどう出てくるかの点検。

**なぜ M がこれを見るか**: 作者の構成は「**面を示すベース + その上に乗る濃淡レイヤ**」
(2026-10-24)。ベースを同定するには**濃淡の側も見えていなければならない**。

測る量(名前では決めない。名前は最後に不一致の印字として出すだけ):

    tile256(層) := 描画画素が「256グリッドに整列し、完全に充填された
                   256x256 タイルの和集合」であるか
                   (画素数が 65,536 の整数倍 / bbox の原点と寸法が 256 の倍数 /
                    bbox 内が全部 drawn)

7016x6543 のキャンバスでは 65,536 / 45,905,688 = **0.001428** なので、
`drawn_ratio` が 0.001428 とその整数倍なら 256 タイル単位を疑う。

**これはモノクロのプールの点検であって、カラーの測定ではない。**
"""
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ADD = Path('/home/sh1/deepl/lineart/data/incoming_manuscript')
P65 = Path('/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv')
TILE = 256
TILE_PX = TILE * TILE


def tile256(drawn):
    cnt = int(drawn.sum())
    if cnt == 0 or cnt % TILE_PX:
        return False, cnt
    ys, xs = np.nonzero(drawn)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    ok = (y0 % TILE == 0 and x0 % TILE == 0
          and (y1 - y0 + 1) % TILE == 0 and (x1 - x0 + 1) % TILE == 0
          and bool(drawn[y0:y1 + 1, x0:x1 + 1].all()))
    return ok, cnt


def scan(paths, label):
    rows = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        drawn = im < 255
        ok, cnt = tile256(drawn)
        rows.append((Path(p).name, ok, cnt, cnt / drawn.size))
    n_ok = sum(1 for r in rows if r[1])
    print(f'{label}: {len(rows)} 枚中 **{n_ok} 枚**が 256 タイル整列・完全充填')
    return rows


def main():
    print('=== 1. part_layers の層ラスタ(5ページ、全部) ===')
    names = {}
    for j in glob.glob(str(ADD / 'part_layers/*_layers.json')):
        for l in json.load(open(j, encoding='utf-8'))['layers']:
            names[l['file']] = l['name']
    rows = scan(sorted(glob.glob(str(ADD / 'part_layers/*_L[0-9]*.png'))),
                'part_layers')
    print('  名前との突き合わせ(選別ではなく不一致の印字):')
    for r in rows:
        if r[1]:
            print(f'    整列 {r[0]}  name={names.get(r[0])!r}  '
                  f'px {r[2]} = {r[2]//TILE_PX} タイル')
    out = [r for r in rows if not r[1] and 'グラデ' in (names.get(r[0]) or '')]
    print(f'  名前に グラデ を含むが整列しなかった層: {len(out)}')

    print('\n=== 2. ako6 の納品グラデマスク ===')
    # マスクも層ラスタと同じ約束(0 = 印の付いた面、255 = 外)。6枚を目視で確認した
    g = scan(sorted(glob.glob(str(ADD / 'ako6/grad/*_mask.png'))),
             'ako6/grad マスク')
    dr = sorted(r[3] for r in g if r[3] > 0)
    if dr:
        print(f'  マスクの面積率: n {len(dr)} 最小 {dr[0]:.5f} '
              f'中央 {dr[len(dr)//2]:.5f} 最大 {dr[-1]:.5f}')

    print('\n=== 3. priority65 の一覧(ラスタ無しで列だけ) ===')
    if P65.exists():
        r = list(csv.DictReader(open(P65, encoding='utf-8-sig')))
        gf = [x for x in r if x['has_gradation_fill'].strip().lower() in ('true', '1')]
        zero = sum(1 for x in gf if float(x['drawn_ratio']) == 0.0)
        print(f'  has_gradation_fill 真: {len(gf)} / {len(r)} 行、'
              f'rendered 真 {sum(1 for x in gf if x["rendered"]=="True")}')
        print(f'  うち drawn_ratio が厳密に 0.0: **{zero}**')
        nz = Counter(x['drawn_ratio'] for x in gf if float(x['drawn_ratio']) > 0)
        print('  0 でない drawn_ratio の多い順(上5):', nz.most_common(5))
        unit = TILE_PX / (7016 * 6543)
        print(f'  1タイル = {unit:.6f} なので、その整数倍かを見る:')
        for v, c in nz.most_common(6):
            k = float(v) / unit
            print(f'    {v:>10}  {c:3d} 行  / 1タイル = {k:7.3f}  '
                  f'{"整数倍に近い" if abs(k-round(k)) < 0.02 else "-"}')
    else:
        print('  priority65 CSV が見えない(Track L の木)。この節は飛ばす')

    print('\n=== 4. cv2.distanceTransform の罠(背景画素が無い層) ===')
    full = np.ones((1000, 1000), np.uint8)
    for lab, m in (('mask=5', 5), ('mask=3', 3),
                   ('PRECISE', cv2.DIST_MASK_PRECISE)):
        d = cv2.distanceTransform(full, cv2.DIST_L2, m)
        print(f'  全面 drawn / {lab:>8}: max {float(d.max()):.6g}  '
              f'(正解は中心までの ~500)')
    f2 = np.ones((1000, 1000), np.uint8)
    f2[0, :] = 0
    print(f'  背景を1行足すと mask=5: max '
          f'{float(cv2.distanceTransform(f2, cv2.DIST_L2, 5).max()):.6g} (期待 999)')
    print('  => **背景画素が1つも無い層では距離変換は黙って壊れ、値は mask 依存。**')
    print('     カラーのベース層は紙面をほぼ覆う可能性があるので、太さを測る前に')
    print('     背景画素の有無を点検すること(M3 の B5 の前提)。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
