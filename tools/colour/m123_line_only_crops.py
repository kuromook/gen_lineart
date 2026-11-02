"""追補205 §F: L4 の提案を原寸で作者に見せる(第2の棒の材料。点数は出さない)。

頁ごとに1枚、評価域(マスクの面の和)の外接矩形 + 余白。**縮小しない。**
色: 黒 = 線 / 塗り = 域内の L4 の提案(提案ごとに色)/ 緑の縁 = F1@0.5 で正解の面に当たった提案 /
橙の縁 = 当たらなかった域内の提案 / 青の細線 = マスクの面の縁(正解の側)/ 白 = 域外(数えていない)。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m102_b_standard_v2 as m102                                    # noqa: E402
import m119_mask_faces_and_lines as m119                             # noqa: E402
import m122_line_only_faces as m122                                  # noqa: E402

OUT = Path('results/m123_line_only_crops')
PAD = 48
K3 = np.ones((3, 3), np.uint8)


def edge(m):
    return m & ~cv2.erode(m.astype(np.uint8), K3, borderValue=1).astype(bool)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    rng = np.random.default_rng(0)
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, _ = loader(kinds)
        Fl, Tl, _, _ = m122.build_truth(pg, kinds)
        U = Fl > 0
        lab, _, _ = m122.regions(pg['ink'], m122.MAIN_K, pg['floor_px'])
        keep, _ = m122.in_domain(lab, U)
        r = m122.score(lab, Tl, keep)[m122.MAIN]
        ys, xs = np.nonzero(U)
        Y0, Y1 = max(0, ys.min() - PAD), min(pg['H'], ys.max() + 1 + PAD)
        X0, X1 = max(0, xs.min() - PAD), min(pg['W'], xs.max() + 1 + PAD)
        L = lab[Y0:Y1, X0:X1]
        col = np.zeros((int(lab.max()) + 1, 3), np.uint8)
        col[keep] = rng.integers(150, 250, (len(keep), 3))
        isin = np.zeros(len(col), bool)
        isin[keep] = True
        im = np.full(L.shape + (3,), 255, np.uint8)
        im[isin[L]] = col[L[isin[L]]]
        hit = np.zeros(len(col), bool)
        hit[list(r['pids'])] = True
        e = cv2.dilate((L != cv2.erode(L.astype(np.float32), K3).astype(np.int32)).astype(np.uint8)
                       | (L != cv2.dilate(L.astype(np.float32), K3).astype(np.int32)).astype(np.uint8), K3).astype(bool)
        im[e & isin[L] & ~hit[L]] = (0, 140, 255)
        im[e & hit[L]] = (40, 170, 40)
        fe = edge(Fl[Y0:Y1, X0:X1] > 0) | (Fl[Y0:Y1, X0:X1] != cv2.erode(
            Fl[Y0:Y1, X0:X1].astype(np.float32), K3).astype(np.int32))
        im[fe & (Fl[Y0:Y1, X0:X1] > 0)] = (230, 80, 0)
        im[pg['ink'][Y0:Y1, X0:X1]] = 0
        out = OUT / f'{pg["key"]}_L{m122.MAIN_K}_x{X0}_y{Y0}.png'
        cv2.imwrite(str(out), im)
        print(f'{out}  {X1 - X0}x{Y1 - Y0}(原寸)  域内 {len(keep)} / 当たり {r["hit"]}')
        del pg, Fl, Tl, lab


if __name__ == '__main__':
    main()
