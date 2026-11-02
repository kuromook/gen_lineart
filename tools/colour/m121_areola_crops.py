"""追補203: `gakusai3` の乳輪2か所を並べる(点数ではない)。

左 = 完成画のプレビュー(**1/4 縮小を最近傍で 4倍に戻したもの。原寸ではない**)/
中 = 線の層 L22 の墨(原寸)/ 右 = 原寸の重ね(黒 = L22 の墨、青 = L29 `SKIN` マスクの縁、橙 = クリップ無しの塗り L64 `deep` の縁)。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m12_score_a0_nulls as m12                                     # noqa: E402
import m116_mask_material_adopted4 as m116                           # noqa: E402

OUT = Path('results/m121_areola_crops')
WINS = {'left': (2915, 4585), 'right': (4087, 4728)}
HALF = 340
K4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


def edge(v):
    return v & ~cv2.erode(v.astype(np.uint8), K4, borderValue=1).astype(bool)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = next(p for p in json.load(open(m116.CM, encoding='utf-8'))['pages'] if 'gakusai3' in p['source'])
    files = {int(l['layer_id']): l['file'] for l in p['layers'] if l.get('file')}
    mm = {int(r['layer_id']): r for r in json.load(open(m116.MM, encoding='utf-8'))['layers']
          if r['page_path'] == p['source']}
    pv = cv2.imread(str(m116.D / p['preview']['file']))
    s = int(round(p['preview']['縮小率']))
    ink = cv2.imread(str(m116.D / files[22]), cv2.IMREAD_UNCHANGED)[..., 3] >= m12.ALPHA_DRAWN
    deep = cv2.imread(str(m116.D / files[64]), cv2.IMREAD_UNCHANGED)[..., 3] >= m12.ALPHA_DRAWN
    g = cv2.imread(str(m116.D / mm[29]['mask_file']), cv2.IMREAD_UNCHANGED)
    mask = (g[..., 0] if g.ndim == 3 else g) <= 127                  # **反転。<=127 が見える側**
    for nm, (cx, cy) in WINS.items():
        y0, y1, x0, x1 = cy - HALF, cy + HALF, cx - HALF, cx + HALF
        a = cv2.resize(pv[y0 // s:y1 // s, x0 // s:x1 // s], (x1 - x0, y1 - y0),
                       interpolation=cv2.INTER_NEAREST)
        b = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
        b[ink[y0:y1, x0:x1]] = 0
        c = b.copy()
        c[cv2.dilate(edge(deep[y0:y1, x0:x1]).astype(np.uint8), K4).astype(bool)] = (0, 140, 255)
        c[cv2.dilate(edge(mask[y0:y1, x0:x1]).astype(np.uint8), K4).astype(bool)] = (230, 80, 0)
        c[ink[y0:y1, x0:x1]] = 0
        sep = np.full((y1 - y0, 8, 3), 200, np.uint8)
        out = OUT / f'gakusai3_areola_{nm}_x{x0}_y{y0}.png'
        cv2.imwrite(str(out), np.hstack([a, sep, b, sep, c]))
        print(f'{out}  {3 * (x1 - x0) + 16}x{y1 - y0}')


if __name__ == '__main__':
    main()
