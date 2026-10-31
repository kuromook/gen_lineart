#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**M98: 「線そのもの」と「線で区切られた面」の違いを絵にする。下見。**

**作者の問い(2026-10-31)**: 「これは線画だが**線の区切られた面を探す**ということ?」
→ **いまの標準は線そのものを数えている**(ベース層の画素の8連結)。
**線が囲んだ領域は一度も計算していない。** その差を見せる。

**これは測定ではない**: **棒も既知セルも持たず、どの数も登録に載せない。**
**線画の層を名前(`線画` を含む)で選んでいる** —— **登録の同定ではない**(条例4)。
**下見として、名前で選んだと書いた上で出す。**

  (a) 線そのもの = 墨の **8連結**(いまの標準と同じ数え方)
  (b) 線で区切られた面 = 墨の**補集合の 4連結**(前景8連結の双対。
      **背景も8連結にすると、斜めの線の隙間から漏れて領域が消える**)。
      **外周に通じる領域は「囲まれていない」**ので分けて数える

  <venv>/python tools/colour/m98_line_enclosed_regions.py
"""
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                                   # noqa: E402
import m16_score_corrected_standard as m16                         # noqa: E402
import m77_standard_visible as m77                                 # noqa: E402

OUT = pathlib.Path('results/m98_line_enclosed')
PAGE = 'skima12H'
WINDOW = (2174, 772, 561, 807)        # `m97` の L140 の窓(そのまま比べられる)
PALETTE = np.array([(255, 0, 255), (0, 140, 255), (0, 170, 0), (255, 140, 0),
                    (0, 0, 230), (160, 0, 160), (0, 200, 200), (200, 80, 0),
                    (120, 60, 220), (0, 110, 255), (90, 160, 0), (230, 0, 120)],
                   np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    p = [q for q in pages if PAGE in q['page']][0]
    rows_vis, _ = m77.visible_rows(byp[p['source']])
    bases, _ = m16.tree_bases(rows_vis)
    files = {l['layer_id']: l['file'] for l in p['layers']}
    s = int(round(p['preview']['縮小率']))
    pw, ph = p['preview']['size']
    grid = (ph * s, pw * s)
    floor = max(1, round(m12.FLOOR * grid[0] * grid[1]))

    sel = [lid for lid, r in bases.items() if '線画' in r['name'] and lid in files]
    print(f'{p["page"]}  画布 {grid[1]}x{grid[0]}  床 {floor}px')
    print(f'線画の名前のベース {len(sel)} 枚(**名前で選んだ。登録の同定ではない**)')

    ink = np.zeros(grid, bool)
    for lid in sel:
        im = cv2.imread(str(m12.D / files[lid]), cv2.IMREAD_UNCHANGED)
        im = im[:grid[0], :grid[1]]
        ink[:im.shape[0], :im.shape[1]] |= (im[..., 3] >= m12.ALPHA_DRAWN)

    # **`connectivity` は必ずキーワードで渡す** —— 第2の位置引数は `labels` なので、
    # 位置で渡すと**黙って既定の8連結**になる(罠 `T-M28`。初版がそれで、
    # 背景を8連結で数えて囲まれた領域が 48 → 38、面積率 2.63% → 1.02% に化けた)
    n, _, st, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    a_cnt = int((st[1:, cv2.CC_STAT_AREA] >= floor).sum())
    nb, lb, sb, _ = cv2.connectedComponentsWithStats(
        (~ink).astype(np.uint8), connectivity=4)
    border = set(np.unique(np.r_[lb[0], lb[-1], lb[:, 0], lb[:, -1]])) - {0}
    kept = [i for i in range(1, nb) if sb[i, cv2.CC_STAT_AREA] >= floor]
    enc = [i for i in kept if i not in border]
    out = [i for i in kept if i in border]
    tot = sum(int(sb[i, cv2.CC_STAT_AREA]) for i in enc)
    px = grid[0] * grid[1]
    print(f'  墨 {ink.mean()*100:.3f}%')
    print(f'  (a) 線そのもの(8連結、床以上)      **{a_cnt}**')
    print(f'  (b) 線で区切られた面(床以上)        {len(kept)}'
          f'  / 外に通じる {len(out)} / **囲まれている {len(enc)}**')
    print(f'      外に通じる最大が画布の {max(sb[i, cv2.CC_STAT_AREA] for i in out)/px*100:.2f}%')
    print(f'      **囲まれた領域の合計は画布の {tot/px*100:.2f}%**')

    # ---- 絵。囲まれた領域だけ塗る。開いた領域は白のまま
    canvas = np.full((grid[0], grid[1], 3), 255, np.uint8)
    for k, i in enumerate(sorted(enc, key=lambda i: -sb[i, cv2.CC_STAT_AREA])):
        canvas[lb == i] = PALETTE[k % len(PALETTE)]
    canvas[ink] = 0
    cv2.imwrite(str(OUT / f'{PAGE}_fullpage_native_enclosed{len(enc)}.png'), canvas)
    x, y, w, h = WINDOW
    cv2.imwrite(str(OUT / f'{PAGE}_L140window_native_x{x}_y{y}_{w}x{h}.png'),
                canvas[y:y + h, x:x + w])
    print(f'\nwrote {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
