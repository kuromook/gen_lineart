"""`page0002` の K8r の12面を、線と (b) 標準の面に重ねて**原寸で**切り出す(規律6)。点数は出さない。

追補197 §E-1(作者の選択 2026-11-02「1」)。問い: **マスクの面と線の閉じた面が1対1にならないのは、
線が閉じていないからか、マスクが複数の面を束ねているからか** —— 作者が目で見る。
層ごとに1枚(その層の床超の面の外接矩形 + 余白)。**縮小しない。**

色: 黒 = 線(`線画` の墨)/ 灰 = 線以外の境界(標準の境界)/ 緑 = 標準の閉じた面 /
赤の縁と薄い赤 = マスクの面(K8r)。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m12_score_a0_nulls as m12                                     # noqa: E402
import m16_score_corrected_standard as m16                           # noqa: E402
import m102_b_standard_v2 as m102                                    # noqa: E402
import m106_why_page0005_fails as m106                               # noqa: E402

OUT = Path('results/m118_page0002_k8_crops')
MD = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/dataset_colour_pilot_masks_20261101')
LAYERS = {59: 'CLOTH', 61: 'HAIR', 63: 'EYE', 65: 'TEETH', 67: 'UNDERWARE', 69: 'SKIN'}
PAD = 48


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    byp = m16.page_rows()
    p = next(p for p in json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
             if p['page'].endswith('page0002'))
    ink, bnd, floor_px, _, _ = m106.build(p, byp, kinds)
    srows, _ = m106.faces(bnd, ink, floor_px)
    H, W = bnd.shape
    _, lab, _, _ = cv2.connectedComponentsWithStats((~bnd).astype(np.uint8), connectivity=4)
    std = np.isin(lab, [r['face'] for r in srows])
    del lab
    print(f'標準の閉じた面 {len(srows)}: ' + ', '.join(
        f'face {r["face"]} {r["area"]/(H*W)*100:.3f}% bbox y{r["y0"]}-{r["y1"]} x{r["x0"]}-{r["x1"]}'
        for r in srows))
    mm = {int(r['layer_id']): r for r in json.load(open(MD / 'layer_masks_manifest.json',
                                                        encoding='utf-8'))['layers']
          if r['page_path'] == p['source']}
    for lid, nm in LAYERS.items():
        g = cv2.imread(str(MD / mm[lid]['mask_file']), cv2.IMREAD_UNCHANGED)
        g = (g[..., 0] if g.ndim == 3 else g)[:H, :W] <= 127      # **反転。<=127 が見える側**
        n, lb, st, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), connectivity=8)
        big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= floor_px]
        face = np.isin(lb, big)
        del lb
        ys, xs = np.nonzero(face)
        y0, y1 = max(0, ys.min() - PAD), min(H, ys.max() + 1 + PAD)
        x0, x1 = max(0, xs.min() - PAD), min(W, xs.max() + 1 + PAD)
        f = face[y0:y1, x0:x1]
        im = np.full(f.shape + (3,), 255, np.uint8)
        im[std[y0:y1, x0:x1]] = (170, 230, 170)
        im[f] = (im[f] * 0.55 + np.array((90, 90, 255)) * 0.45).astype(np.uint8)
        edge = f & ~cv2.erode(f.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        im[cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = (0, 0, 230)
        im[bnd[y0:y1, x0:x1] & ~ink[y0:y1, x0:x1]] = (150, 150, 150)
        im[ink[y0:y1, x0:x1]] = (0, 0, 0)
        out = OUT / f'page0002_L{lid}_{nm}_x{x0}_y{y0}.png'
        cv2.imwrite(str(out), im)
        print(f'L{lid} {nm}: 面 {len(big)} / 切り出し {x1-x0}x{y1-y0}(原寸)/ '
              f'標準の面の画素 {int(std[y0:y1, x0:x1].sum())} → {out}')


if __name__ == '__main__':
    main()
