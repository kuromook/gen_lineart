#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**M101: ある面の境目を<ins>何が</ins>作ったのかを層まで辿る。**

**作者の問い(2026-10-31、`m100` の12枚を見て)**:

  - `skima12chara`「**ベルト・パンツ周辺の 5,9 あたりがなぜ割れたかは気になる**」
  - `skima12H`「**矩形がみえる これを分けているのはなにか不明**」
  - `page0002` / `page0005`「**線で区切られた面を、さらに何かのルールで分割されている**」

**これは測定ではない。** 棒も既知セルも持たない。**問いに答えるための診断**である。

やること: `m100` と**同じ経路**で窓と面を作り直し(種・窓・番号が一致する)、
**各面の境目の画素を、`線の墨` / `遮蔽物の縁` に分け、縁ならどの層のものかを数える。**

**層は名前で選んでいない** —— 遮蔽物の集合は追補122 §B の規則どおり。
**名前は答を読むために印字するだけ。**

  <venv>/python tools/colour/m101_why_split.py
"""
import collections
import pathlib
import sys
import json

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                                   # noqa: E402
import m16_score_corrected_standard as m16                         # noqa: E402
import m77_standard_visible as m77                                 # noqa: E402
import m87_split_standard_by_occlusion as m87                      # noqa: E402
import m99_b_standard as m99                                       # noqa: E402
import m100_b_face_crops as m100                                   # noqa: E402

ASK = {                       # 頁 -> (窓x, 窓y, 聞かれた面の番号)
    '000_2018_skima11_skima11_skima12chara': (1187, 880, [1, 2, 3, 4, 5, 9]),
    '000_2018_skima11_skima11_skima12H': (890, 255, [1, 2, 3]),
    # 作者「ざらざらした画面に色が乗っている これは面ではない」(page0019)
    '000_2017_illustration_sunshine2_page0019': (3544, 3039, [1, 2, 3, 4]),
    # 作者「線で区切られた面をさらに何かのルールで分割されている」(page0002)
    '000_2017_illustration_sunshine2_page0002': (2145, 4259, [1, 2, 3]),
}
WIN = m100.WIN


def main():
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    for p in pages:
        if p['page'] not in ASK:
            continue
        x0, y0, want = ASK[p['page']]
        rows_vis, _ = m77.visible_rows(byp[p['source']])
        _, kids = m16.build_tree(rows_vis)
        cbase = m87.clip_base_map(rows_vis)
        files = {l['layer_id']: l['file'] for l in p['layers']}
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        cache = {}

        ink = np.zeros(grid, bool)
        for r in rows_vis:
            if r['layer_type'] == m16.PAPER_TYPE or not m99.is_line(r):
                continue
            a = m87.alpha_of(r['layer_id'], files, grid, cache)
            if a is not None:
                ink[:a.shape[0], :a.shape[1]] |= a
        occ = np.zeros(grid, bool)
        edges = {}                       # 層 -> 窓の中の縁(bool)
        drop = collections.Counter()
        for r in rows_vis:
            if r['layer_type'] == m16.PAPER_TYPE:
                continue
            if (r.get('opacity') or '0') == '0' or (r.get('composite') or '0') == '2':
                continue
            m = m87.occluder_mask(r, kids, files, grid, cache, cbase, True, drop)
            if m is None:
                continue
            e = m99.inner_edge(m)
            occ |= e
            sub = e[y0:y0 + WIN, x0:x0 + WIN]
            if sub.any():
                edges[(r['layer_id'], r['name'], r['folder'])] = \
                    np.ascontiguousarray(sub)
            del m, e
        cache.clear()

        boundary = ink | occ
        n, lab, st, _ = cv2.connectedComponentsWithStats(
            (~boundary).astype(np.uint8), connectivity=4)
        border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
        cset = {i for i in range(1, n)
                if i not in border and st[i, cv2.CC_STAT_AREA] >= floor_px}
        sub = lab[y0:y0 + WIN, x0:x0 + WIN]
        ids = [i for i in np.unique(sub)
               if i in cset and int((sub == i).sum()) >= floor_px]
        ids = sorted(ids, key=lambda i: -int((sub == i).sum()))   # m100 と同じ順
        isub = ink[y0:y0 + WIN, x0:x0 + WIN]
        osub = occ[y0:y0 + WIN, x0:x0 + WIN]
        k3 = np.ones((3, 3), np.uint8)

        print(f'\n=== {p["page"]} 窓 x{x0} y{y0} / 面 {len(ids)} ===')
        for k in want:
            if k > len(ids):
                print(f'  面 {k}: 窓に無い')
                continue
            m = (sub == ids[k - 1])
            ring = (cv2.dilate(m.astype(np.uint8), k3).astype(bool) & ~m
                    & (isub | osub))
            nr = int(ring.sum())
            n_ink = int((ring & isub).sum())
            n_occ = int((ring & osub & ~isub).sum())
            print(f'  面 {k:>2}  面積 {int(m.sum()):>7}px  境目 {nr:>6}px  '
                  f'**線 {n_ink/max(1,nr)*100:5.1f}% / 遮蔽物の縁だけ '
                  f'{n_occ/max(1,nr)*100:5.1f}%**')
            hits = []
            for (lid, nm, fo), e in edges.items():
                c = int((ring & e).sum())
                if c:
                    hits.append((c, lid, nm, fo))
            for c, lid, nm, fo in sorted(hits, reverse=True)[:5]:
                print(f'        {c/max(1,nr)*100:5.1f}%  L{lid:>4} '
                      f'{(nm or "")[:34]:34s} folder={fo}')
            del m, ring
        del lab, st, sub, ink, occ, edges
    return 0


if __name__ == '__main__':
    sys.exit(main())
