"""M35: 件1 の作り直し。**成分が窓より大きいと、1:1 は面の形を示さない**(基盤の差し戻し)。

差し戻しの理由: `illust201710\\page0003` L50 の成分は **1,620,984px** で、
窓 1400×1400 = 1,960,000px に**収まっていない**。絵はほぼ全面が黒く、
マゼンタは窓の縁で切れた断片になっていて、**作者は「どこで始まってどこで終わる面か」が
見えないまま「1つの面ですか」と聞かれる**ことになっていた。

**これは今夜3つめの同族**: (1) 選別の根拠と見せる絵が別の対象 /
(2) 輪郭の描き方が見せたい対象を壊す(網点) / **(3) 成分が窓より大きい**。

**作り直しの形(基盤の提案。役割を分ける)**:

| 枚 | 中身 | 役割 |
|---|---|---|
| **(a) `_shape_1of<N>`** | **成分全体の輪郭を縮小で1枚**(縮小率を明記) | **形の把握**。判定の対象ではない |
| **(b) `_waist_1to1`** | **融けていると疑う「くびれ」を 1:1 で1枚** | **判定の対象**(原寸) |

**くびれの決め方(走らせる前に登録する。結果を見てから選び直さない)**:

> **成分を半径 r で収縮していき、2つ以上に割れる最小の r を探す。**
> **割れる直前に残っていた「橋」の位置を窓の中心にする。**
> r は 4・8・16・32・64・128 px(原寸)で試す。**割れなければ「くびれ無し」と印字して
> (b) を作らない** —— 作れなかったことも結果である。

**既知セル**: このページの標準が **ベース7 / 成分46 / 面46**(`m27` の走行と同じ)。
"""
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                         # noqa: E402
import m16_score_corrected_standard as m16               # noqa: E402
import m27_author_crops as m27                           # noqa: E402

PAGE = '000_2018_illust201710_page0003'
LAYER = 50
OUT = Path('results/m27_author_crops')
EXPECT = {'bases': 7, 'components': 46, 'areas': 46}
RADII = (4, 8, 16, 32, 64, 128)
WIN = 1400


def main():
    rows = list(csv.DictReader(open(m27.CSV872, encoding='utf-8')))
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r['page_path']].append(r)
    p = [json.loads(l) for l in
         open(m27.DELIV / 'colour_layers_pages.jsonl', encoding='utf-8')
         if l.strip() and json.loads(l)['page'] == PAGE][0]
    m12.D = m27.DELIV
    s = int(round(p['preview']['縮小率']))
    pw, ph = p['preview']['size']
    grid = (ph * s, pw * s)
    floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
    comps = []
    (_, _), (_, garea_a), st = m16.measure_standard(
        p, byp[p['source']], grid, floor_px, out_comps=comps)
    print(f'== 既知セル ==')
    ok = all((st['bases'], st['components'], len(garea_a))[i] ==
             (EXPECT['bases'], EXPECT['components'], EXPECT['areas'])[i]
             for i in range(3))
    print(f'  ベース {st["bases"]}/{EXPECT["bases"]} 成分 '
          f'{st["components"]}/{EXPECT["components"]} 面 '
          f'{len(garea_a)}/{EXPECT["areas"]}  '
          f'{"PASS" if ok else "**FAIL**"}')
    if not ok:
        print('**既知セルが合わない。切り出しを書かない。**')
        return 1

    c = max((x for x in comps if x['layer_id'] == LAYER),
            key=lambda x: x['area'])
    y, x, h, w = c['box']
    H, W = grid
    files = {l['layer_id']: l['file'] for l in p['layers']}
    zmap = {q['layer_id']: q['z'] for q in comps}
    page_ids = [i for i, _ in sorted(
        ((l['layer_id'], zmap.get(l['layer_id'], 0)) for l in p['layers']),
        key=lambda t: t[1])]
    print(f'\n成分 L{LAYER} 面積 {c["area"]} / 箱 y{y} x{x} {w}×{h}'
          f'(窓 {WIN}×{WIN} に収まらない)')

    # --- くびれ: 割れる最小の r ---
    m = c['mask'].astype(np.uint8)
    waist = None
    for r in RADII:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        er = cv2.erode(m, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(er, connectivity=8)
        big = [i for i in range(1, n)
               if stats[i, cv2.CC_STAT_AREA] >= 0.02 * c['area']]
        print(f'  収縮 r={r:>4}px → 2%以上の塊 {len(big)}')
        if len(big) >= 2:
            a1 = (lab == big[0]).astype(np.uint8)
            a2 = (lab == big[1]).astype(np.uint8)
            kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * r + 3, 2 * r + 3))
            br = (cv2.dilate(a1, kk) > 0) & (cv2.dilate(a2, kk) > 0) & (m > 0)
            if br.sum() == 0:
                br = (cv2.dilate(a1, kk) > 0) & (cv2.dilate(a2, kk) > 0)
            yy, xx = np.nonzero(br)
            if yy.size:
                waist = (y + int(yy.mean()), x + int(xx.mean()), r)
                break
    if waist is None:
        print('  → **くびれ無し。(b) は作らない**(作れなかったことも結果)')
    else:
        print(f'  → **くびれ: 原寸 y{waist[0]} x{waist[1]}(割れた収縮半径 '
              f'{waist[2]}px)**')

    # --- (a) 形(縮小)---
    pv = cv2.imread(str(m27.DELIV / p['preview']['file']), cv2.IMREAD_COLOR)
    full = np.zeros((H, W), bool)
    full[y:y + h, x:x + w] = c['mask']
    small = full[::s, ::s].astype(np.uint8)
    sh, sw = min(small.shape[0], pv.shape[0]), min(small.shape[1], pv.shape[1])
    shape = pv.copy()
    cs, _ = cv2.findContours(small[:sh, :sw], cv2.RETR_CCOMP,
                             cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(shape, cs, -1, (255, 0, 255), 2)
    base = f'case1b_{PAGE}_L{LAYER}'
    if waist is not None:
        cv2.circle(shape, (waist[1] // s, waist[0] // s), 26, (0, 180, 255), 3)
    cv2.imwrite(str(OUT / f'{base}_shape_1of{s}.png'), shape)
    print(f'\nwrote {base}_shape_1of{s}.png(形の把握。判定の対象ではない)')

    # --- (b) くびれ 1:1 ---
    if waist is not None:
        cy, cx, _ = waist
        y0 = max(0, min(H - WIN, cy - WIN // 2))
        x0 = max(0, min(W - WIN, cx - WIN // 2))
        y1, x1 = min(H, y0 + WIN), min(W, x0 + WIN)
        img = m27.composite(page_ids, files, c['box'], (y0, x0, y1, x1))
        sub = full[y0:y1, x0:x1].astype(np.uint8)
        mk = img.copy()
        cs2, _ = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mk, cs2, -1, (255, 0, 255), 3)
        cv2.imwrite(str(OUT / f'{base}_waist_1to1_x{x0}_y{y0}_raw.png'), img)
        cv2.imwrite(str(OUT / f'{base}_waist_1to1_x{x0}_y{y0}_marked.png'), mk)
        print(f'wrote {base}_waist_1to1_x{x0}_y{y0}_raw.png / _marked.png'
              f'(**判定の対象。原寸1:1**)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
