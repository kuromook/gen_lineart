#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**M105: 追補159 §J の3つの問いを、原寸の切り出しで作者に見せる。**

**規律6**: **絵が何を描いているかは作者が決める。疑わしいときはモンタージュで論じず
原寸の切り出しを出す。** **縮小しない。合成しない**(原寸の合成は IC6 で閉じている。`m100`)。

**窓の選び方(内容で選ばない。<ins>構造の規則</ins>で先に決める)**:

  `page0005`  問い(i)  **`rendering_lineart_middle0001` のアルファの外接箱の中心**
  `page0019`  問い(ii) **R6 が落とした層(ルビ、L33)のアルファが<ins>最も多く入る</ins>窓**
                       (**初版は外接箱の中心にしたが、文字が無い人物の顔に落ちた** ——
                        **外接箱の中心は対象が写ることを保証しない。窓の規則を直して作り直した**)
  `page0002`  問い(ii) **T4 の閉じた面のうち<ins>最大</ins>の面の重心**(面が3枚しか無い頁)
  `page0003`  打診     **T4 の閉じた面のうち<ins>最大</ins>の面の重心**

**窓は 1024x1024**(`m100` と同じ。作者が既に見ている大きさ)。**画布の端で切れる場合は寄せる。**

**出す絵**(頁ごと):

  `_1_ink`        **この企画が「線」と呼んでいる墨だけ**(名前に `線画`)。黒
  `_2_faces_T4`   墨(黒)+ **遮蔽物の縁**(灰)+ **T4 の閉じた面を色で塗る**
  `page0005` だけ:
  `_3_ink_with3d` **墨 + `rendering_lineart`**(= **作者の定義での「線」**)。
                  **足した分を<ins>赤</ins>で描く**(どこが 3D の線か分かるように)
  `_4_faces_T5`   **作者の定義で数えた面**(T5)

**出力は `results/`(gitignore)。原稿の画素なので追跡しない。`SendUserFile` で直接送る。**

  <venv>/python tools/colour/m105_ruling_crops.py
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
import m99_b_standard as m99                                       # noqa: E402
import m102_b_standard_v2 as m102                                  # noqa: E402
import m104_b_standard_v3 as m104                                  # noqa: E402

OUT = pathlib.Path('results/m105_ruling_crops')
WIN = 1024
GREY = (170, 170, 170)
RED = (0, 0, 255)                 # **3D の線だけに使う。面の色には赤を入れない**
PALETTE = [(255, 0, 255), (0, 140, 255), (0, 170, 0), (255, 140, 0),
           (160, 0, 160), (0, 200, 200), (200, 80, 0),
           (120, 60, 220), (0, 110, 255), (90, 160, 0), (230, 0, 120)]
# 頁 -> 出す段。**`page0019` は R6 の前と後を同じ窓で並べる**(148面 → 49面)
STAGES = {'000_2017_illustration_sunshine2_page0019': ('T3', 'T4'),
          '000_2017_illustration_sunshine2_page0002': ('T4',),
          '000_2018_illust201710_illust201710_page0003': ('T4',),
          '000_2018_illust201710_illust201710_page0005': ('T4', 'T5')}
# 頁 -> (窓の決め方, その引数)。**走る前に書いた。内容で選ばない**
ANCHOR = {
    '000_2018_illust201710_illust201710_page0005': ('layer_bbox', 'rendering_lineart'),
    '000_2017_illustration_sunshine2_page0019': ('layer_kind_densest', 'text'),
    '000_2017_illustration_sunshine2_page0002': ('largest_face', None),
    '000_2018_illust201710_illust201710_page0003': ('largest_face', None),
}


def clamp(cx, cy, grid):
    x0 = int(min(max(0, cx - WIN // 2), max(0, grid[1] - WIN)))
    y0 = int(min(max(0, cy - WIN // 2), max(0, grid[0] - WIN)))
    return x0, y0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    for p in pages:
        if p['page'] not in ANCHOR:
            continue
        how, arg = ANCHOR[p['page']]
        rows_vis, _ = m77.visible_rows(byp[p['source']])
        _, kids = m16.build_tree(rows_vis)
        cbase = m87.clip_base_map(rows_vis)
        files = {l['layer_id']: l['file'] for l in p['layers']}
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        px = grid[0] * grid[1]
        floor_px = max(1, int(round(px * m12.FLOOR)))
        cache = {}
        body = [r for r in rows_vis if r['layer_type'] != m16.PAPER_TYPE]
        skip = {int(r['layer_id']) for r in body
                if kinds.get((p['source'], str(r['layer_id']))) == m104.TEXT_KIND}

        ink = np.zeros(grid, bool)
        ink3d = np.zeros(grid, bool)      # `rendering_lineart` だけ(赤で描く分)
        for r in body:
            nm = (r['name'] or '').lower()
            line = m99.is_line(r)
            if not line and m104.RENDER_LINE not in nm:
                continue
            if line and m102.is_sketch(r):
                continue
            a = m104.fit(m87.alpha_of(r['layer_id'], files, grid, cache), grid)
            if a is None:
                continue
            if line:
                ink |= a
            elif r['folder'] != 'True':
                ink3d |= a

        edge3 = np.zeros(grid, bool)      # R6 を当てない縁(= 第2版 T3。灰で描く)
        edge4 = np.zeros(grid, bool)      # R6 を当てた縁(= 第3版 T4)
        dd = collections.Counter()
        for r in body:
            if (r.get('opacity') or '0') == '0' or (r.get('composite') or '0') == '2':
                continue
            m = m87.occluder_mask(r, kids, files, grid, cache, cbase, True, dd)
            if m is None:
                continue
            leaf = r['folder'] != 'True'
            rect, boxr = m102.rect_of(m104.fit(m87.alpha_of(r['layer_id'], files,
                                                            grid, cache), grid), px) \
                if leaf else (0.0, 0.0)
            kind = kinds.get((p['source'], str(r['layer_id'])))
            mat = leaf and (kind == 'image_material'
                            or (not m102.is_tone_correction(r)
                                and rect >= m102.RECT_BAR and boxr < m102.FULL_CANVAS))
            if (r['clip'] == '1' or m102.is_named(r) or mat or m102.is_sketch(r)):
                continue
            e3 = m99.inner_edge(m)
            edge3 |= e3
            has_text = (int(r['layer_id']) in skip
                        or any(i in skip for i in m16.descendants(int(r['layer_id']),
                                                                  kids)))
            if not has_text:
                edge4 |= e3
            else:
                m6 = m104.occluder_mask_r6(r, kids, files, grid, cache, cbase, skip, dd)
                if m6 is not None:
                    edge4 |= m99.inner_edge(m6)
                    del m6
            del m, e3

        # ---- 窓の中心(**走る前に決めた規則のとおり**)
        if how == 'layer_bbox':
            tgt = next((r for r in body if arg in (r['name'] or '').lower()
                        and r['folder'] != 'True'), None)
            a = m104.fit(m87.alpha_of(tgt['layer_id'], files, grid, cache), grid)
            ys, xs = np.nonzero(a)
            cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
            why = f'層 L{tgt["layer_id"]} {tgt["name"]} の外接箱の中心'
        elif how == 'layer_kind_densest':
            tgt = next(r for r in body if int(r['layer_id']) in skip
                       and r['folder'] != 'True')
            a = m104.fit(m87.alpha_of(tgt['layer_id'], files, grid, cache), grid)
            # **その層の画素が最も多く入る 1024 窓**(16画素に粗めてから箱で畳む)
            q = 16
            small = cv2.boxFilter(
                cv2.resize(a.astype(np.float32), (grid[1] // q, grid[0] // q),
                           interpolation=cv2.INTER_AREA),
                -1, (WIN // q, WIN // q), normalize=False)
            iy, ix = np.unravel_index(int(np.argmax(small)), small.shape)
            cx, cy = ix * q, iy * q
            why = (f'R6 が落とした層 L{tgt["layer_id"]} {tgt["name"]} の画素が'
                   f'最も多く入る窓({int(small.max())*q*q} 画素ぶん)')
        else:
            cx = cy = None
            why = 'T4 の最大の閉じた面の重心'
        cache.clear()

        made = {'T3': (ink | (edge3 & m104.open_from(ink)), edge3, None),
                'T4': (ink | (edge4 & m104.open_from(ink)), edge4, None),
                'T5': ((ink | ink3d) | (edge4 & m104.open_from(ink | ink3d)),
                       edge4, ink3d)}
        for tag in STAGES[p['page']]:
            bnd, edge, add = made[tag]
            n, lab, st, cent = cv2.connectedComponentsWithStats(
                (~bnd).astype(np.uint8), connectivity=4)
            border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
            closed = [i for i in range(1, n)
                      if i not in border and int(st[i, cv2.CC_STAT_AREA]) >= floor_px]
            if cx is None:
                big = max(closed, key=lambda i: int(st[i, cv2.CC_STAT_AREA]))
                cx, cy = int(cent[big][0]), int(cent[big][1])
                why += f'(面 {big}、{int(st[big, cv2.CC_STAT_AREA])} 画素)'
            x0, y0 = clamp(cx, cy, grid)
            sub = lab[y0:y0 + WIN, x0:x0 + WIN]
            here = [i for i in closed if (sub == i).any()]
            img = np.full((min(WIN, grid[0]), min(WIN, grid[1]), 3), 255, np.uint8)
            img[edge[y0:y0 + WIN, x0:x0 + WIN]] = GREY
            for j, i in enumerate(sorted(here, key=lambda i: -int(st[i, cv2.CC_STAT_AREA]))):
                img[sub == i] = PALETTE[j % len(PALETTE)]
            img[ink[y0:y0 + WIN, x0:x0 + WIN]] = (0, 0, 0)
            if add is not None:
                img[add[y0:y0 + WIN, x0:x0 + WIN]] = RED
            cv2.imwrite(str(OUT / f'{p["page"]}_faces_{tag}_native.png'), img)
            # 墨だけ(同じ窓)
            ib = np.full_like(img, 255)
            ib[ink[y0:y0 + WIN, x0:x0 + WIN]] = (0, 0, 0)
            if add is not None:
                ib[add[y0:y0 + WIN, x0:x0 + WIN]] = RED
                cv2.imwrite(str(OUT / f'{p["page"]}_ink_with3d_native.png'), ib)
            else:
                cv2.imwrite(str(OUT / f'{p["page"]}_ink_native.png'), ib)
            print(f'{p["page"]:44s} {tag} 窓 ({x0},{y0}) {WIN}x{WIN} / '
                  f'画布 {grid[1]}x{grid[0]} / 窓の中の閉じた面 {len(here)} / '
                  f'頁全体 {len(closed)} / 窓の決め方: {why}', flush=True)
        del ink, ink3d, edge3, edge4, lab, st


if __name__ == '__main__':
    main()
