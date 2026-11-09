# -*- coding: utf-8 -*-
"""重ね塗りの疑い 4 面の**片**を、親ごとに**同じ窓**で原寸に切り出す(登録
`doc/l38_overlap_registration_20261109.md`)。

割る規則は `pieces_in_face()`、描き方は `render()` を呼ぶ(写しを持たない)。
窓 = 親の外接矩形 + 余白 400(線が足りなければ倍々に広げる。`l38_piece_crops_20261108.py` と同じ)。
点検: 片の印の画素数が片の面積と一致 / 片が1つに決まる / 出した数 = 9。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_overlap_crops_20261109.py
"""
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
from l17_known_cells import DEL, MAX_FRAC, MIN_FRAC, arms, jobs, zmap  # noqa: E402
from l38_merge_cause_20261106 import drop_wip, layer_masks          # noqa: E402
from l38_neck_crops_20261107 import MININK, line_ink, render        # noqa: E402
from l38_separate_problem_a_20261108 import pieces_in_face          # noqa: E402

ROOT = '/home/sh1/deepl/lineart-fill-region'
POOL = f'{ROOT}/results/l38_faces_separated_20261108.json'
OUT = f'{ROOT}/results/overlap20261109'
RATIO, MARGIN = 1.5, 400


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = json.load(open(POOL, encoding='utf-8'))
    g = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r['origin'] == 'piece':
            g[tuple(r['parent'])].append((i, r))
    sus = {k: v for k, v in g.items() if sum(r['area'] for _, r in v) / k[4] > RATIO}
    want = sum(len(v) for v in sus.values())
    print(f'和/親 > {RATIO} の親 {len(sus)} / 片 {want}(全数)', flush=True)

    zz = zmap()
    fill, line = arms('gated')
    cut = collections.Counter()
    js = [j for j in drop_wip(jobs(zz, fill, line), cut)
          if any(k[0] == j[1] and k[1] == j[0] for k in sus)]
    made, skip, meta = 0, collections.Counter(), []
    for pid, work, fz, lz, sel in js:
        lm, shape = layer_masks(fz, 'solid', cut)
        zs = {lid: zz.get((pid, str(lid))) for lid, _ in lm}
        order = sorted(lm, key=lambda t: (zs[t[0]] if zs[t[0]] is not None else -1), reverse=True)
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
                P = lab[y0:y1, x0:x1] == ci
                glab, gids, _ = comps(F[y0:y1, x0:x1] & P, 8)
                win = [(lid, m[y0:y1, x0:x1] & P) for lid, m in order]
                for key, ps in sus.items():
                    if key[:3] != (work, pid, ci) or key[3] not in gids:
                        continue
                    R = glab == key[3]
                    if int(R.sum()) != key[4]:
                        skip['面積が合わない(`comp` が同じ別のコマ)'] += 1
                        continue
                    big, _ = pieces_in_face(R, win)
                    LN = lines[y0:y1, x0:x1]
                    ys, xs = np.nonzero(R)
                    mg = MARGIN
                    while True:
                        b = (max(0, ys.min() - mg), min(R.shape[0], ys.max() + 1 + mg),
                             max(0, xs.min() - mg), min(R.shape[1], xs.max() + 1 + mg))
                        if LN[b[0]:b[1], b[2]:b[3]].mean() >= MININK or \
                                (b[1] - b[0], b[3] - b[2]) == R.shape:
                            break
                        mg *= 2
                    masks = []
                    for i, r in ps:
                        hit = [m for lid, aa, m in big if lid == r['piece_lid'] and aa == r['area']]
                        if len(hit) != 1 or int(hit[0].sum()) != r['area']:
                            skip[f'片が1つに決まらない / 面積が合わない({len(hit)})'] += 1
                            continue
                        masks.append((i, r, hit[0]))
                    for k, (i, r, M) in enumerate(masks):
                        other = np.zeros_like(M)
                        for j, (_, _, M2) in enumerate(masks):
                            if j != k:
                                other |= M2
                        other &= ~M
                        img = render(M, other, LN, b)
                        fid = f'{work}_{pid[-4:]}_c{ci}r{key[3]}_p{k + 1}'
                        cv2.imwrite(f'{OUT}/{fid}.png', img)
                        ov = [int((M & m2).sum()) for j, (_, _, m2) in enumerate(masks) if j != k]
                        meta.append(dict(fid=fid, row=i, work=work, pid=pid, comp=int(ci),
                                         region=int(key[3]), parent_area=int(key[4]),
                                         piece_lid=r['piece_lid'], area=r['area'],
                                         overlap_with_others=ov, margin=mg,
                                         win=[int(b[0] + y0), int(b[1] + y0),
                                              int(b[2] + x0), int(b[3] + x0)]))
                        made += 1
                        print(f'  {fid}  層 {r["piece_lid"]} 面積 {r["area"]:,}  '
                              f'他の片との重なり {ov}  窓 {b[1] - b[0]}x{b[3] - b[2]}', flush=True)
    json.dump(meta, open(f'{OUT}/meta.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n出した片 {made} / 頼んだ {want} -> {OUT}')
    for k, v in skip.most_common():
        print(f'  落とした {v}  {k}')
    if made != want:
        sys.exit('**数が合わない。合うまで聞き取りに出さない**')


if __name__ == '__main__':
    main()
