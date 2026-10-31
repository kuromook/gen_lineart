#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**M100: (b) の面を原寸の切り出しにして作者に問う。**

**問いは1つ**: **この面はあなたの「面」ですか。**
**本当の棒は人の判定**(第1条・棒は2本)。追補152 §G1 の「次」。

**選び方(走らせる前に commit する。結果を見てから選び直さない。`m27` / `m92` と同じ規律)**:

  母集団 = **`pop_6pages_lineart` の 6頁**(追補152 の走行と同じ)
  面     = **S3(線 + 遮蔽物「作者の集合」)の<ins>閉じた</ins>面**。床 1e-5
  窓     = **1頁につき 1枚。1024x1024 を一様無作為**(**内容で選ばない。NC1 の誤り**)。
           **種 `SEED = 20261101` を固定。** 窓に閉じた面が1つも無ければ**引き直し**、
           **引き直した回数を印字する**(規律9)。**上限 200 回**
  問う面 = **その窓の中に画素を持つ閉じた面のうち、窓内の面積が床以上のもの全部**
           (**大きいものを選ばない。小さいものを捨てない**)

**絵は合成しない。** 6頁のうち3頁(`P0166` / `P0151` / `P0444`)は **IC6 に落ちている**ので、
**こちらで原寸の合成を組み直す道は閉じている**(修正5-A)。
→ **原寸で出すのは「線の墨」と「遮蔽物の縁」だけ**、つまり **(b) が面を決めた材料そのもの**。

  `_1_lines_native`   線の墨だけ(黒)。**描き込み無し。判定はこちらで**
  `_2_faces_native`   線の墨(黒)+ **遮蔽物の縁(灰)** + **面を色で塗る**(番号つき)

**線が細いので外周の輪郭では描かない**(`m97` の初版で線が塗り潰された)。**面自身を塗る。**

  <venv>/python tools/colour/m100_b_face_crops.py
"""
import collections
import csv
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

OUT = pathlib.Path('results/m100_b_faces')
SEED = 20261101
WIN = 1024
MAX_DRAW = 200
GREY = (170, 170, 170)
PALETTE = [(255, 0, 255), (0, 140, 255), (0, 170, 0), (255, 140, 0),
           (0, 0, 230), (160, 0, 160), (0, 200, 200), (200, 80, 0),
           (120, 60, 220), (0, 110, 255), (90, 160, 0), (230, 0, 120)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    byp = m16.page_rows()
    pages = json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
    keep = []
    for p in pages:
        rows_vis, _ = m77.visible_rows(byp[p['source']])
        if any(r['layer_type'] != m16.PAPER_TYPE and m99.is_line(r)
               for r in rows_vis):
            keep.append(p)
    print(f'母集団 {len(keep)} 頁 / 種 {SEED} / 窓 {WIN}x{WIN} 一様無作為')

    sheet = []
    for p in keep:
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
        drop = collections.Counter()
        for r in rows_vis:
            if r['layer_type'] == m16.PAPER_TYPE:
                continue
            if (r.get('opacity') or '0') == '0':
                continue
            if (r.get('composite') or '0') == '2':      # 乗算は遮蔽物でない
                continue
            m = m87.occluder_mask(r, kids, files, grid, cache, cbase, True, drop)
            if m is None:
                continue
            occ |= m99.inner_edge(m)
            del m
        cache.clear()

        boundary = ink | occ
        n, lab, st, _ = cv2.connectedComponentsWithStats(
            (~boundary).astype(np.uint8), connectivity=4)
        border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
        closed = [i for i in range(1, n)
                  if i not in border and st[i, cv2.CC_STAT_AREA] >= floor_px]
        print(f'\n{p["page"]:44s} 画布 {grid[1]}x{grid[0]}  閉じた面 {len(closed)}')
        if not closed:
            print('   閉じた面が0。窓を引かない')
            continue

        cset = set(closed)
        redraws, box = 0, None
        for _ in range(MAX_DRAW):
            y0 = int(rng.integers(0, max(1, grid[0] - WIN)))
            x0 = int(rng.integers(0, max(1, grid[1] - WIN)))
            sub = lab[y0:y0 + WIN, x0:x0 + WIN]
            ids = [i for i in np.unique(sub) if i in cset
                   and int((sub == i).sum()) >= floor_px]
            if ids:
                box = (x0, y0, ids)
                break
            redraws += 1
        if box is None:
            print(f'   {MAX_DRAW} 回引いても閉じた面が入らなかった。この頁は窓なし')
            continue
        x0, y0, ids = box
        print(f'   窓 x{x0} y{y0} {WIN}x{WIN}(引き直し {redraws} 回)  問う面 {len(ids)}')

        sub = lab[y0:y0 + WIN, x0:x0 + WIN]
        bsub = boundary[y0:y0 + WIN, x0:x0 + WIN]
        isub = ink[y0:y0 + WIN, x0:x0 + WIN]
        osub = occ[y0:y0 + WIN, x0:x0 + WIN]
        raw = np.full((sub.shape[0], sub.shape[1], 3), 255, np.uint8)
        raw[isub] = 0
        marked = np.full((sub.shape[0], sub.shape[1], 3), 255, np.uint8)
        marked[osub] = GREY
        labels = []
        for k, i in enumerate(sorted(ids, key=lambda i: -int((sub == i).sum())),
                              start=1):
            m = (sub == i)
            marked[m] = PALETTE[(k - 1) % len(PALETTE)]
            # **番号は面の<ins>内側</ins>に置く** —— 外接箱の角に置くと、面から離れた
            # 白地に数字が浮き、**どれについて答えるのかが曖昧になる**(初版がそれ)。
            # 最も内側の点は距離変換の最大。**縁に0を1列足す**(T-M2: 背景画素が無いと壊れる)
            pad = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
            pad[1:-1, 1:-1] = m
            d = cv2.distanceTransform(pad, cv2.DIST_L2, 3)
            cy, cx = np.unravel_index(int(d.argmax()), d.shape)
            labels.append((k, int(cx) - 1, int(cy) - 1,
                           PALETTE[(k - 1) % len(PALETTE)]))
            del pad, d
            a_win = int(m.sum())
            sheet.append((p['page'], k, int(st[i, cv2.CC_STAT_AREA]), a_win,
                          x0, y0))
            print(f'      {k:>2}  窓内 {a_win:>7}px  全体 '
                  f'{int(st[i, cv2.CC_STAT_AREA]):>8}px')
            del m
        marked[isub] = 0                                   # **線は最後に上書き**
        for k, tx, ty, col in labels:
            t = str(k)
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            ox = max(2, min(tx - tw // 2, marked.shape[1] - tw - 4))
            oy = max(th + 4, min(ty + th // 2, marked.shape[0] - 4))
            cv2.rectangle(marked, (ox - 2, oy - th - 3), (ox + tw + 2, oy + 3),
                          (255, 255, 255), -1)
            cv2.putText(marked, t, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        col, 2, cv2.LINE_AA)
        tag = f'{p["page"][-14:]}_x{x0}_y{y0}_{WIN}x{WIN}_{len(ids)}faces'
        cv2.imwrite(str(OUT / f'{tag}_1_lines_native.png'), raw)
        cv2.imwrite(str(OUT / f'{tag}_2_faces_native.png'), marked)
        del lab, st, sub, boundary, ink, occ

    with (OUT / 'faces_asked.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['page', 'n', 'area_full_px', 'area_in_window_px', 'win_x', 'win_y'])
        w.writerows(sheet)
    print(f'\n問う面 合計 {len(sheet)} / wrote {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
