# -*- coding: utf-8 -*-
"""**分離した片**(と元のままの面)を原寸 1:1 で切り出す(2026-11-08)。

**割る規則の実体は1つ** —— `l38_separate_problem_a_20261108.pieces_in_face()` を呼ぶ。
**規則の写しを持たない**(条例4: 名前・鍵は含意を持つが、含意は点検ではない)。

- 問う単位(片か面)を**淡い橙で塗り、輪郭を濃い橙**で引く。**同じ面だった他の片は淡い灰** ——
  **何が切り離されたのかが見える**
- 線層の墨を先に置く(**シルエットでは絵として読めない**。2026-11-07 の作者の指摘)
- **窓に線が入るまで余白を広げる**(落とすと一様でなくなる。規律9)
- **点検を2つ、器具の中で回す**(登録 `doc/l38_problem_b_registration_20261108.md` の NC1/NC2):
  1. **印の付いた画素の数が、頼まれた単位の面積と一致すること**(= 親に印を付けていないこと)
  2. **輪郭が1画素以上在ること**(= 面が見えない切り出しを出さないこと)

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_piece_crops_20261108.py \
        --list data/l38_problemb_20261108/faces.json --out-name problemb20261108
"""
import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import A, comps                                # noqa: E402
from l17_known_cells import (DEL, MAX_FRAC, MIN_FRAC, arms,         # noqa: E402
                             jobs, zmap)
from l38_merge_cause_20261106 import drop_wip, layer_masks          # noqa: E402
from l38_neck_crops_20261107 import MININK, line_ink, render        # noqa: E402
from l38_separate_problem_a_20261108 import pieces_in_face          # noqa: E402

ROOT = '/home/sh1/deepl/lineart-fill-region'
OUT = f'{ROOT}/results'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True, help='出す単位を名指しした JSON(`fid` つき)')
    ap.add_argument('--out-name', required=True)
    ap.add_argument('--margin', type=int, default=400)
    a = ap.parse_args()
    out = f'{OUT}/{a.out_name}'
    os.makedirs(out, exist_ok=True)
    want = json.load(open(a.list, encoding='utf-8'))
    byface = collections.defaultdict(list)
    for r in want:
        # 片は**親の**(comp, region, 面積)でコマの中の面を引く。元のままの面は自分自身
        p = r['parent'] if r['origin'] == 'piece' else [r['work'], r['pid'], r['comp'],
                                                        r['region'], r['area']]
        byface[(r['work'], r['pid'], int(p[2]), int(p[3]), int(p[4]))].append(r)
    print(f'頼まれた単位 {len(want)} / 面 {len(byface)}', flush=True)

    zz = zmap()
    fill, line = arms('gated')
    cut = collections.Counter()
    js = drop_wip(jobs(zz, fill, line), cut)
    pages = {(r['work'], r['pid']) for r in want}
    js = [j for j in js if (j[1], j[0]) in pages]
    print(f'走る頁 {len(js)}', flush=True)

    made, skip = 0, collections.Counter()
    for i, job in enumerate(js, 1):
        pid, work, fz, lz, sel = job
        lm, shape = layer_masks(fz, 'solid', cut)
        if shape is None:
            skip['塗り層が1枚も読めない頁'] += 1
            continue
        zs = {}
        for lid, _ in lm:
            zs[lid] = zz.get((pid, str(lid)))
        order = sorted(lm, key=lambda t: (zs[t[0]] if zs[t[0]] is not None else -1),
                       reverse=True)
        lines = line_ink(lz, shape)
        F = np.zeros(shape, bool)
        for _, m in lm:
            F |= m
        H, W = shape
        for fr in sel:
            km = cv2.imread(f'{DEL}/{work}/koma/{fr["file"]}', cv2.IMREAD_GRAYSCALE)
            if km is None or km.shape != (H, W):
                continue
            n, lab, st, _ = cv2.connectedComponentsWithStats((km > 127).astype(np.uint8), 8)
            for ci in range(1, n):
                ar = int(st[ci, A])
                if ar < MIN_FRAC * H * W or ar > MAX_FRAC * H * W:
                    continue
                y0 = max(0, st[ci, cv2.CC_STAT_TOP] - 2)
                x0 = max(0, st[ci, cv2.CC_STAT_LEFT] - 2)
                y1 = min(H, y0 + st[ci, cv2.CC_STAT_HEIGHT] + 4)
                x1 = min(W, x0 + st[ci, cv2.CC_STAT_WIDTH] + 4)
                P = (lab[y0:y1, x0:x1] == ci)
                glab, gids, gstat = comps(F[y0:y1, x0:x1] & P, 8)
                win = [(lid, m[y0:y1, x0:x1] & P) for lid, m in order]
                for key, rs in byface.items():
                    if key[:3] != (work, pid, ci) or key[3] not in gids:
                        continue
                    R = glab == key[3]
                    # **`comp` は枠ファイルごとに1から振り直される。面積で同定する**
                    if int(R.sum()) != key[4]:
                        skip['面積が合わない(`comp` が同じ別のコマ)'] += 1
                        continue
                    big, _ = pieces_in_face(R, win)
                    LN = lines[y0:y1, x0:x1]
                    for r in rs:
                        if r['origin'] == 'piece':
                            hit = [m for lid, aa, m in big
                                   if lid == r['piece_lid'] and aa == r['area']]
                            if len(hit) != 1:
                                skip[f'片が1つに決まらない({len(hit)} 個当たった)'] += 1
                                continue
                            M = hit[0]
                        else:
                            M = R
                        if int(M.sum()) != r['area']:
                            skip['**印の画素が単位の面積と合わない**(NC1)'] += 1
                            continue
                        other = (glab > 0) & ~M
                        ys, xs = np.nonzero(M)
                        mg, grew = a.margin, 0
                        while True:
                            by0 = max(0, ys.min() - mg)
                            by1 = min(M.shape[0], ys.max() + 1 + mg)
                            bx0 = max(0, xs.min() - mg)
                            bx1 = min(M.shape[1], xs.max() + 1 + mg)
                            if LN[by0:by1, bx0:bx1].mean() >= MININK:
                                break
                            if (by1 - by0, bx1 - bx0) == M.shape:
                                skip['**コマ全体でも線が足りない**(広げきった)'] += 1
                                break
                            mg, grew = mg * 2, grew + 1
                        if grew:
                            skip[f'余白を {grew} 回広げた'] += 1
                        img = render(M, other, LN, (by0, by1, bx0, bx1))
                        # **NC2 の点検: 輪郭(濃い橙 BGR=(0,90,255))が1画素以上在ること**
                        c = ((img[:, :, 0] < 60) & (img[:, :, 1] > 50)
                             & (img[:, :, 1] < 140) & (img[:, :, 2] > 200)).sum()
                        if c == 0:
                            skip['**輪郭が1画素も無い切り出し**(出さない。NC2)'] += 1
                            continue
                        cv2.imwrite(f'{out}/{r["fid"]}.png', img)
                        made += 1
        print(f'  {i}/{len(js)} 頁  出した {made}', flush=True)

    print(f'\n出した単位 {made} / 頼んだ {len(want)}  -> {out}')
    print('落としたもの(理由別。規律9):')
    for k, v in skip.most_common():
        print(f'  {v:>4}  {k}')
    for k, v in cut.most_common():
        print(f'  {v:>4}  {k}')
    if made != len(want):
        print('**数が合わない。合うまで聞き取りに出さない**(登録の NC1/NC2)')


if __name__ == '__main__':
    main()
