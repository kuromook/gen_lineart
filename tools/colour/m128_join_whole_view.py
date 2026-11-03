"""追補214 の 20(`page0002` P1–Q16 のつなぎ)を全体で見せる(作者の求め「全体をみたい」)。点数は出さない。

1枚目: 頁全体を 1/4 に縮小(**縮小と明記**)。2枚目: Q16 の外接矩形 + 余白を原寸。
色: 黒 = 線 / 紫 = P1 / 水色 = Q16 / 薄い灰 = その他の提案 / 赤 = 2つが墨を挟まず接する所(縮小図では赤い円で囲む)/ 青 = マスクの面の縁。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m102_b_standard_v2 as m102                                    # noqa: E402
import m119_mask_faces_and_lines as m119                             # noqa: E402
import m122_line_only_faces as m122                                  # noqa: E402

OUT = Path('results/m128_join_whole_view')
P, Q = 1, 16
PAD = 120
K3 = np.ones((3, 3), np.uint8)


def paint(lab, Fl, ink, seam):
    im = np.full(lab.shape + (3,), 255, np.uint8)
    im[lab > 0] = (225, 225, 225)
    im[lab == P] = (240, 160, 170)
    im[lab == Q] = (215, 240, 120)
    im[(Fl > 0) & (Fl != cv2.erode(Fl.astype(np.float32), K3).astype(np.int32))] = (230, 80, 0)
    im[cv2.dilate(seam.astype(np.uint8), K3, iterations=2).astype(bool)] = (0, 0, 230)
    im[ink] = 0
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    pg, _ = m119.load_page0002(kinds)
    Fl, _, _, _ = m122.build_truth(pg, kinds)
    lab = np.load('results/m126_line_ends/page0002_labB.npy')
    seam = np.zeros(lab.shape, bool)
    a, b = lab[:, :-1], lab[:, 1:]
    seam[:, :-1] |= ((a == P) & (b == Q)) | ((a == Q) & (b == P))
    a, b = lab[:-1, :], lab[1:, :]
    seam[:-1, :] |= ((a == P) & (b == Q)) | ((a == Q) & (b == P))
    sy, sx = np.nonzero(seam)
    print(f'P1 {int((lab == P).sum())} px / Q16 {int((lab == Q).sum())} px / 接触 {len(sy)} px  at x {sx.mean():.0f} y {sy.mean():.0f}')
    im = paint(lab, Fl, pg['ink'], seam)
    small = cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4), interpolation=cv2.INTER_AREA)
    cv2.circle(small, (int(sx.mean()) // 4, int(sy.mean()) // 4), 30, (0, 0, 230), 4)
    cv2.imwrite(str(OUT / 'page0002_P1_Q16_whole_quarter_scale.png'), small)
    ys, xs = np.nonzero(lab == Q)
    Y0, Y1 = max(0, ys.min() - PAD), min(lab.shape[0], ys.max() + 1 + PAD)
    X0, X1 = max(0, xs.min() - PAD), min(lab.shape[1], xs.max() + 1 + PAD)
    cv2.imwrite(str(OUT / f'page0002_P1_Q16_Q16extent_native_x{X0}_y{Y0}.png'), im[Y0:Y1, X0:X1])
    print(f'Q16 の範囲 {X1 - X0}x{Y1 - Y0}(原寸)')


if __name__ == '__main__':
    main()
