"""追補208 §E: GA(m=9)が実際に束ねた組だけを原寸で切り出す(第2の棒 H2 の材料。点数は出さない)。

組ごとに1枚。外接矩形 + 余白、**縮小しない**。**画布の大半を持つ組(域外)は外接矩形が画布全体になるので、
束ねた境目(墨の無い接触)の周りだけを切り出す。**
色: 黒 = 線 / 塗り = 組の中の L4 の提案(提案ごとに色)/ 赤 = 束ねた境目(墨を挟まない接触)/ 青の細線 = マスクの面の縁。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m102_b_standard_v2 as m102                                    # noqa: E402
import m119_mask_faces_and_lines as m119                             # noqa: E402
import m122_line_only_faces as m122                                  # noqa: E402
import m124_group_faces as m124                                      # noqa: E402

OUT = Path('results/m125_grouped_crops')
PAD = 160
K3 = np.ones((3, 3), np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    rng = np.random.default_rng(0)
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, _ = loader(kinds)
        Fl, _, _, _ = m122.build_truth(pg, kinds)
        L4, _, _ = m122.regions(pg['ink'], m122.MAIN_K, pg['floor_px'])
        merged = [(p, q, c) for (p, q), c in m124.contacts(L4).items() if c >= m124.MAIN_M]
        print(f'{pg["key"]}: 束ねた組 {len(merged)}')
        col = rng.integers(150, 250, (int(L4.max()) + 1, 3)).astype(np.uint8)
        for j, (p, q, c) in enumerate(merged):
            # 束ねた境目 = p と q が墨を挟まずに接する画素
            seam = np.zeros(L4.shape, bool)
            a, b = L4[:, :-1], L4[:, 1:]
            seam[:, :-1] |= ((a == p) & (b == q)) | ((a == q) & (b == p))
            a, b = L4[:-1, :], L4[1:, :]
            seam[:-1, :] |= ((a == p) & (b == q)) | ((a == q) & (b == p))
            ys, xs = np.nonzero(seam)
            Y0, Y1 = max(0, ys.min() - PAD), min(pg['H'], ys.max() + 1 + PAD)
            X0, X1 = max(0, xs.min() - PAD), min(pg['W'], xs.max() + 1 + PAD)
            L = L4[Y0:Y1, X0:X1]
            im = np.full(L.shape + (3,), 255, np.uint8)
            for i in (p, q):
                im[L == i] = col[i]
            F = Fl[Y0:Y1, X0:X1]
            fe = (F > 0) & (F != cv2.erode(F.astype(np.float32), K3).astype(np.int32))
            im[fe] = (230, 80, 0)
            im[cv2.dilate(seam[Y0:Y1, X0:X1].astype(np.uint8), K3, iterations=2).astype(bool)] = (0, 0, 230)
            im[pg['ink'][Y0:Y1, X0:X1]] = 0
            out = OUT / f'{pg["key"]}_pair{j}_P{p}_Q{q}_x{X0}_y{Y0}.png'
            cv2.imwrite(str(out), im)
            print(f'  {out}  {X1 - X0}x{Y1 - Y0}(原寸)  接触 {c} px')
        del pg, Fl, L4


if __name__ == '__main__':
    main()
