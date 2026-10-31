#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**M103: C1 が通らなかった2頁で、T3 の面を<ins>何が</ins>割っているのかを層まで辿る。**

**これは測定ではない。棒も既知セルも持たない。** `m102` の登録(`doc/prereg_b_standard_v2_20261031.md`)
§Pass criteria C1 の読み **「通らない頁が在る = その頁では裁定の実装が効いていない。
原因を印字して作者に返す」**を実行する診断である。

対象: **`page0019` と `page0005`**(**T3 の線率の中央が 0.00% で C1 FAIL**)。
`page0003`(**PASS、線率 100%**)を**対照として同じ経路で**出す。

量: **T3 で閉じた面のうち線率 < 0.5 のものを集めた領域 `M`** に対し、
**各層の縁が `M` に接している画素数と、接している面の数**。
**層は名前で選んでいない**(`m102` と同じ R1〜R4 の後の集合)。**名前は読むために印字するだけ。**

  <venv>/python tools/colour/m103_why_t3_fails.py
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

WANT = ('000_2017_illustration_sunshine2_page0019',
        '000_2018_illust201710_illust201710_page0005',
        '000_2018_illust201710_illust201710_page0003')


def main():
    kinds = m102.load_kinds()
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    k3 = np.ones((3, 3), np.uint8)

    for p in pages:
        if p['page'] not in WANT:
            continue
        rows_vis, _ = m77.visible_rows(byp[p['source']])
        _, kids = m16.build_tree(rows_vis)
        cbase = m87.clip_base_map(rows_vis)
        files = {l['layer_id']: l['file'] for l in p['layers']}
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        px = grid[0] * grid[1]
        floor_px = max(1, round(m12.FLOOR * px))
        cache = {}
        body = [r for r in rows_vis if r['layer_type'] != m16.PAPER_TYPE]

        ink = np.zeros(grid, bool)
        for r in body:
            if not m99.is_line(r):
                continue
            a = m87.alpha_of(r['layer_id'], files, grid, cache)
            if a is not None:
                ink[:a.shape[0], :a.shape[1]] |= a

        # R1 の「外周に通じている領域」
        _, labl, _, _ = cv2.connectedComponentsWithStats(
            (~ink).astype(np.uint8), connectivity=4)
        bl = set(np.unique(np.r_[labl[0], labl[-1], labl[:, 0], labl[:, -1]])) - {0}
        open_mask = np.isin(labl, np.fromiter(bl, int, len(bl)))
        del labl

        # T3 の縁(R2・R3・R4 を通った層だけ)。**層ごとに持たず、2回走査する**
        def surviving():
            for r in body:
                if (r.get('opacity') or '0') == '0' or \
                        (r.get('composite') or '0') == m102.MULTIPLY:
                    continue
                if r['clip'] == '1' or m102.is_named(r) or m102.is_sketch(r):
                    continue
                m = m87.occluder_mask(r, kids, files, grid, cache, cbase, True,
                                      collections.Counter())
                if m is None:
                    continue
                leaf = r['folder'] != 'True'
                if leaf:
                    rect, boxr = m102.rect_of(
                        m87.alpha_of(r['layer_id'], files, grid, cache), px)
                    kind = kinds.get((p['source'], str(r['layer_id'])))
                    if kind == 'image_material' or (
                            not m102.is_tone_correction(r)
                            and rect >= m102.RECT_BAR and boxr < m102.FULL_CANVAS):
                        continue
                yield r, m

        occ = np.zeros(grid, bool)
        n_occ = 0
        for r, m in surviving():
            occ |= (m99.inner_edge(m) & open_mask)
            n_occ += 1
            del m
        cache.clear()
        boundary = ink | occ
        del occ

        n, lab, st, _ = cv2.connectedComponentsWithStats(
            (~boundary).astype(np.uint8), connectivity=4)
        border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
        low, nclosed = [], 0
        H, W = grid
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            if a < floor_px or i in border:
                continue
            nclosed += 1
            x0 = max(0, int(st[i, cv2.CC_STAT_LEFT]) - 1)
            y0 = max(0, int(st[i, cv2.CC_STAT_TOP]) - 1)
            x1 = min(W, x0 + int(st[i, cv2.CC_STAT_WIDTH]) + 3)
            y1 = min(H, y0 + int(st[i, cv2.CC_STAT_HEIGHT]) + 3)
            m = (lab[y0:y1, x0:x1] == i)
            ring = cv2.dilate(m.astype(np.uint8), k3).astype(bool) & ~m \
                & boundary[y0:y1, x0:x1]
            nr = int(ring.sum())
            if nr and int((ring & ink[y0:y1, x0:x1]).sum()) / nr < 0.5:
                low.append((i, a, nr))
            del m, ring
        print(f'\n=== {p["page"]} / T3 で閉じた面 {nclosed} / '
              f'**線率 < 0.5 の面 {len(low)}** / 縁を出した層 {n_occ} ===')
        if not low:
            print('  **線率 < 0.5 の面が無い。この頁では裁定が効いている**')
            del lab, st, ink, boundary, open_mask
            continue
        ids = np.fromiter((i for i, _, _ in low), int, len(low))
        M = np.isin(lab, ids)
        print(f'  その面の合計面積 {int(M.sum())/px*100:5.2f}% / '
              f'最大 {max(a for _, a, _ in low)/px*100:5.2f}%')
        hits = []
        for r, m in surviving():
            e = m99.inner_edge(m) & open_mask
            d = cv2.dilate(e.astype(np.uint8), k3).astype(bool) & M
            c = int(d.sum())
            if c:
                hits.append((c, len(set(np.unique(lab[d])) - {0}), r['layer_id'],
                             r['name'], r['folder'], '/'.join(m102.anc_parts(r))))
            del m, e, d
        cache.clear()
        print(f'  {"px":>9} {"面":>5}  層')
        for c, nf, lid, nm, fo, anc in sorted(hits, reverse=True)[:12]:
            print(f'  {c:>9} {nf:>5}  L{lid:>4} {str(nm)[:26]:26s} '
                  f'folder={fo:5s} /{anc}')
        del lab, st, ink, boundary, open_mask, M
    return 0


if __name__ == '__main__':
    sys.exit(main())
