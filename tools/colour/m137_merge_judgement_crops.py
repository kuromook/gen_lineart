"""追補229 §B H6: τ = 6.0 の W が結合した組 8 + 対照(隣接・未結合)4 を各頁から選び、混ぜて切り出す。

作者にはどちらの組か伝えない。対応表は `results/m137_merge_crops/key.json`(判定ページには出さない)。
左 = 2つの面の全体(最大 600px に縮小)/ 右 = 境目の中央の原寸 ±150px。
色: 黒 = 線 / 橙 = 面 A / 青 = 面 B / 薄い灰 = 他の面。点数は出さない。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m102_b_standard_v2 as m102                                    # noqa: E402
import m119_mask_faces_and_lines as m119                             # noqa: E402
import m130_t_junctions as m130                                      # noqa: E402
import m136_thin_line_merge as m136                                  # noqa: E402

OUT = Path('results/m137_merge_crops')
SRC = m136.OUT
N_W, N_C, HALF, BIG = 8, 4, 150, 600
CA, CB, CO = (40, 140, 245), (230, 120, 30), (225, 225, 225)


def paint(lab, ink, a, b, box):
    y0, y1, x0, x1 = box
    L = lab[y0:y1, x0:x1]
    im = np.full(L.shape + (3,), 255, np.uint8)
    im[L > 0] = CO
    im[L == a] = CA
    im[L == b] = CB
    im[ink[y0:y1, x0:x1]] = 0
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    items = []
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, _ = loader(kinds)
        key, H, W = pg['key'], pg['H'], pg['W']
        _, inkP = m136.bases(pg)
        segsT = json.load(open(m130.OUT / f'{key}_segs.json'))['segs']
        ink = m130.draw_segs(inkP, [tuple(map(tuple, s)) for s in segsT])
        lab = np.load(SRC / f'{key}_tau6_lab.npy')
        pr = json.load(open(SRC / f'{key}_tau6_pairs.json'))
        merged = [tuple(p) for p in pr['merged']]
        ctrl = [tuple(p) for p in pr['cand'] if tuple(p) not in set(merged)]
        rng = np.random.default_rng(0)
        pick = [('W', merged[i]) for i in rng.permutation(len(merged))[:N_W]] + \
               [('対照', ctrl[i]) for i in rng.permutation(len(ctrl))[:N_C]]
        print(f'{key}: 結合 {len(merged)} 組から {sum(k == "W" for k, _ in pick)} / 対照の候補 {len(ctrl)} 組から {sum(k == "対照" for k, _ in pick)}')
        for kind, (a, b) in pick:
            ma, mb = lab == a, lab == b
            # 境目の中央: 面 B の画素のうち面 A に最も近いものと、その相手の中点
            da, ia = cv2.distanceTransformWithLabels((~ma).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            yb, xb = np.nonzero(mb)
            j = int(np.argmin(da[yb, xb]))
            ys_, xs_ = np.nonzero(ma)
            # ia の番号は「0 の画素」を走査順に数えたもの = ma の画素の走査順
            k = ia[yb[j], xb[j]] - 1
            cy, cx = (int(yb[j]) + int(ys_[k])) // 2, (int(xb[j]) + int(xs_[k])) // 2
            del da, ia
            yy, xx = np.nonzero(ma | mb)
            by0, by1, bx0, bx1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
            pad = max(40, (by1 - by0 + bx1 - bx0) // 20)
            box = (max(0, by0 - pad), min(H, by1 + pad), max(0, bx0 - pad), min(W, bx1 + pad))
            whole = paint(lab, ink, a, b, box)
            s = min(1.0, BIG / max(whole.shape[:2]))
            whole = cv2.resize(whole, (max(1, int(whole.shape[1] * s)), max(1, int(whole.shape[0] * s))), interpolation=cv2.INTER_AREA)
            cv2.circle(whole, (int((cx - box[2]) * s), int((cy - box[0]) * s)), 14, (0, 0, 230), 3)
            nb = (max(0, cy - HALF), min(H, cy + HALF), max(0, cx - HALF), min(W, cx + HALF))
            close = paint(lab, ink, a, b, nb)
            h = max(whole.shape[0], close.shape[0])
            pad_ = lambda im: np.vstack([im, np.full((h - im.shape[0], im.shape[1], 3), 255, np.uint8)])  # noqa: E731
            im = np.hstack([pad_(whole), np.full((h, 12, 3), 255, np.uint8), pad_(close)])
            items.append(dict(page=key, kind=kind, a=int(a), b=int(b), width=pr['width'][f'{a}-{b}'], img=im))
        del pg, lab, ink, inkP
    order = np.random.default_rng(0).permutation(len(items))
    keyrows = []
    for n, i in enumerate(order, 1):
        it = items[i]
        cv2.imwrite(str(OUT / f'm{n:02d}.png'), it['img'])
        keyrows.append(dict(n=n, id=f'm{n:02d}', page=it['page'], kind=it['kind'], a=it['a'], b=it['b'],
                            width=it['width'], w=int(it['img'].shape[1]), h=int(it['img'].shape[0])))
    json.dump(keyrows, open(OUT / 'key.json', 'w'), ensure_ascii=False, indent=0)
    print(f'切り出し {len(keyrows)} 枚')


if __name__ == '__main__':
    main()
