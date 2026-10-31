#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**M97: 追補147 §F —— P1 の 15島を原寸の切り出しにして作者に問う。**

**問いは1つ**: **この島は面ですか。**
**聞く単位は島**であって、名前でも層でもない(**罠 `T-M27`**: 裁定を名前で引くと、
聞いていない単位に広がる)。

**`L140` と `L146` を分けて出す** —— 作者が 2026-10-26 に見たのは
**追補38 の問0 = `L140` の1成分(11,714px)だけ**で、**`L146` は一度も見せていない**。
**作者が見た単位と、こちらが名前で広げた単位を混ぜない。**

**この頁は `P0444` で、IC6(CLIP 自身の `_preview.png` との引き算)に落ちている**
(平均差 **7.43** 対 棒 3.0)。**修正5-A(ユーザーの決定、2026-10-31)により、
こちらで層から原寸の合成を組み直す道は閉じている。**
→ **原寸で出すのは「その層自身の画素」だけで、下に他の層を敷かない。**
**仕上がった絵は CLIP 自身の `_preview.png`(1/4)で示し、「どこの話か」専用とする**
(**画素を根拠にする主張に使わない**。規律6 / `m27` の3枚目と同じ扱い)。

**描き方は `m27` に合わせる**: 縮小しない / 窓は最大 1400 /
**`_raw`(描き込み無し)と印を付けた版の2枚組**。

**ただし外周の線では描けない**(初版で踏んだ): **この層は線画なので、島の外周を
3px で描くと線そのものがマゼンタに埋まり、絵が消えた。**
→ **島は「その画素自身を色で塗り分ける」**(`m95` の③と同じ考え)。
**線に入らなかった画素は黒のまま残る**ので、**どの筆が1つの島か**が見える。

**門(落ちたら絵を書かない)**: この頁の標準の島が **220**、`L140` が **12**、
`L146` が **3**(追補146・147 の値)。

  <venv>/python tools/colour/m97_hair_island_crops.py
"""
import collections
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                                   # noqa: E402
import m16_score_corrected_standard as m16                         # noqa: E402
import m77_standard_visible as m77                                 # noqa: E402
import m87_split_standard_by_occlusion as m87                      # noqa: E402

OUT = pathlib.Path('results/m97_hair_islands')
PAGE = 'skima12H'
ASK_BASES = (140, 146)          # 140 = 作者が見た / 146 = こちらが名前で広げた
KC = {'page_islands': 220, 140: 12, 146: 3}
WIN = 1400                      # `m27` と同じ上限
MARGIN = 120
LINE = (255, 0, 255)            # マゼンタ(`m27` / `m93` と同じ。文脈の図で使う)
# **島の塗り分け**(BGR)。白地で見分けがつき、黒い線とも違う色
PALETTE = [(255, 0, 255), (0, 140, 255), (0, 170, 0), (255, 140, 0),
           (0, 0, 230), (160, 0, 160), (0, 200, 200), (200, 80, 0),
           (120, 60, 220), (0, 110, 255), (90, 160, 0), (230, 0, 120)]
FRAME = (0, 180, 255)


def layer_over_white(path, grid):
    """**その層自身の画素だけ**を白地に置く。**他の層を敷かない**(合成しない)。"""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    gh, gw = grid
    im = im[:gh, :gw]
    out = np.full((gh, gw, 3), 255, np.uint8)
    h, w = im.shape[:2]
    if im.shape[2] == 4:
        a = (im[..., 3:4].astype(np.float32) / 255.0)
        out[:h, :w] = (im[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    else:
        out[:h, :w] = im[..., :3]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    p = [q for q in pages if PAGE in q['page']][0]
    rows_vis, _ = m77.visible_rows(byp[p['source']])
    bases, _ = m16.tree_bases(rows_vis)
    s = int(round(p['preview']['縮小率']))
    pw, ph = p['preview']['size']
    grid = (ph * s, pw * s)
    floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
    print(f'{p["page"]}  画布 {grid[1]}x{grid[0]}  preview {pw}x{ph} (1/{s})  '
          f'床 {floor_px}px')

    comps = []
    (_, _), (lab, _), _ = m16.measure_standard(p, rows_vis, grid, floor_px,
                                               out_comps=comps)
    comps = [c for c in comps if c['area'] >= floor_px]
    areas = m87.areas_from(lab, comps, floor_px)

    # ---------------------------------------------------------------- 門
    got = {'page_islands': len(areas)}
    for b in ASK_BASES:
        got[b] = sum(1 for a in areas if a['base'] == b)
    ok = True
    print('\n== 門(落ちたら絵を書かない)==')
    for k, want in KC.items():
        g = got[k]
        ok &= want == g
        print(f'  {str(k):14s} 期待 {want:>4} / 実測 {g:>4}  '
              f'{"PASS" if want == g else "**FAIL**"}')
    if not ok:
        print('\n**門が落ちた。絵を書かない**')
        return 1

    files = {l['layer_id']: l['file'] for l in p['layers']}
    prev = cv2.imread(str(m12.D / p['preview']['file']), cv2.IMREAD_COLOR)
    sheet = []

    for b in ASK_BASES:
        g = sorted([a for a in areas if a['base'] == b],
                   key=lambda a: -a['area'])
        ys = min(a['box'][0] for a in g)
        xs = min(a['box'][1] for a in g)
        y2 = max(a['box'][0] + a['box'][2] for a in g)
        x2 = max(a['box'][1] + a['box'][3] for a in g)
        y0 = max(0, ys - MARGIN); x0 = max(0, xs - MARGIN)
        y1 = min(grid[0], y2 + MARGIN); x1 = min(grid[1], x2 + MARGIN)
        if y1 - y0 > WIN or x1 - x0 > WIN:          # `m27` の上限
            y1 = min(y1, y0 + WIN); x1 = min(x1, x0 + WIN)
        name = bases[b]['name']
        seen = '作者が見た層(追補38 問0)' if b == 140 else '**未提示の層**(名前で広げた分)'
        print(f'\n== base L{b} 「{name}」 島 {len(g)} —— {seen} ==')
        print(f'   窓 x{x0} y{y0}  {x1-x0}x{y1-y0}(原寸 1:1)')

        raw = layer_over_white(m12.D / files[b], grid)[y0:y1, x0:x1]
        marked = raw.copy()
        full = np.zeros(grid, np.uint8)
        labels = []
        for i, a in enumerate(g, start=1):
            yy, xx, hh, ww = a['box']
            full[yy:yy + hh, xx:xx + ww][a['mask']] = 1
            sub = np.zeros((y1 - y0, x1 - x0), bool)
            fy0, fx0 = max(yy, y0), max(xx, x0)
            fy1, fx1 = min(yy + hh, y1), min(xx + ww, x1)
            sub[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] = \
                a['mask'][fy0 - yy:fy1 - yy, fx0 - xx:fx1 - xx]
            # **島はその画素自身を塗る。外周の線で描かない**(線が埋まる)
            marked[sub] = PALETTE[(i - 1) % len(PALETTE)]
            labels.append((i, xx - x0, yy - y0, PALETTE[(i - 1) % len(PALETTE)]))
            fill = a['area'] / (hh * ww)
            print(f'   {i:>3}  面積 {a["area"]:>6}px  '
                  f'ページ比 {a["area"]/(grid[0]*grid[1])*100:7.4f}%  '
                  f'箱 {ww}x{hh}  充填率 {fill:.3f}')
            sheet.append((b, i, a['area'], ww, hh, fill))

        # 番号は最後に(島の色を隠さないよう白地を敷く)
        for i, tx, ty, col in labels:
            t = str(i)
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            ox, oy = max(2, min(tx, x1 - x0 - tw - 4)), max(th + 4, ty - 4)
            cv2.rectangle(marked, (ox - 2, oy - th - 3), (ox + tw + 2, oy + 3),
                          (255, 255, 255), -1)
            cv2.putText(marked, t, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        col, 2, cv2.LINE_AA)

        tag = f'L{b}_{len(g)}islands_x{x0}_y{y0}_{x1-x0}x{y1-y0}'
        cv2.imwrite(str(OUT / f'{tag}_1_raw_native.png'), raw)
        cv2.imwrite(str(OUT / f'{tag}_2_islands_coloured_native.png'), marked)

        # 3枚目: **CLIP 自身の preview**(1/4)。場所を示すためだけ
        ctx = prev.copy()
        small = full[::s, ::s]
        sh, sw = min(small.shape[0], ctx.shape[0]), min(small.shape[1], ctx.shape[1])
        cs2, _ = cv2.findContours(small[:sh, :sw], cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ctx, cs2, -1, LINE, 2)
        cv2.rectangle(ctx, (x0 // s, y0 // s), (x1 // s, y1 // s), FRAME, 2)
        cv2.imwrite(str(OUT / f'{tag}_3_whereonpage_CLIPpreview_1of{s}.png'), ctx)

        # 4枚目: preview の同じ窓を4倍(**原寸ではない**。名前に書く)
        pv = prev[y0 // s:y1 // s, x0 // s:x1 // s]
        up = np.repeat(np.repeat(pv, s, 0), s, 1)
        cv2.imwrite(str(OUT / f'{tag}_4_drawing_PREVIEWx{s}_NOT_native.png'), up)

    print(f'\nwrote {OUT} ({len(list(OUT.glob("*.png")))} 枚)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
