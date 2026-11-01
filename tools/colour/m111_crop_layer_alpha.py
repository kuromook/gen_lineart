"""指定した層のアルファを、**指定した窓で原寸1:1**に切り出す(規律6)。

**窓は呼ぶ側が渡す。この器具は窓を選ばない** —— `m109` が決めた窓をそのまま使うため。
層の同定は `layer_id`(構造の鍵)で行い、名前では引かない(条例4)。

使い方: `python -I tools/colour/m111_crop_layer_alpha.py <頁の部分文字列> <layer_id> <x0> <y0> <幅>`
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m12_score_a0_nulls as m12                                     # noqa: E402
import m87_split_standard_by_occlusion as m87                        # noqa: E402

OUT = Path('results/m111_layer_crops')


def main(argv):
    page, lid, x0, y0, win = argv[1], int(argv[2]), *map(int, argv[3:6])
    OUT.mkdir(parents=True, exist_ok=True)
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    p = [q for q in pages if page in q['page']][0]
    files = {l['layer_id']: l['file'] for l in p['layers']}
    s = int(round(p['preview']['縮小率']))
    pw, ph = p['preview']['size']
    grid = (ph * s, pw * s)
    a = m87.alpha_of(lid, files, grid, {})
    if a is None:
        print(f'**層 {lid} にラスタが無い。出さない**')
        return 1
    full = np.zeros(grid, bool)
    full[:a.shape[0], :a.shape[1]] = a
    print(f'{p["page"]} 層 {lid} / 画布 {grid[1]}×{grid[0]} / '
          f'このアルファ {int(full.sum()):,}px = {full.mean()*100:.4f}%')
    crop = (~full[y0:y0+win, x0:x0+win]).astype(np.uint8) * 255
    f = OUT / f'{p["page"]}__layer{lid}__x{x0}_y{y0}_{win}.png'
    cv2.imwrite(str(f), crop)
    black = int((crop < 128).sum())
    print(f'  窓 x{x0}–{x0+win}/y{y0}–{y0+win} / 窓の中の黒 {black:,}px = '
          f'{black/crop.size*100:.3f}%')
    print(f'  {f}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
