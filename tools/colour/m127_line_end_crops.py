"""追補212 §E H3: B の橋と BJ のつなぎを原寸で切り出す(点数は出さない)。

各頁、橋は無作為に最大12本(seed 0)、つなぎは12以下なら全部・超えれば無作為に12。**縮小しない。**
色 —— 橋: 黒 = 線 / **赤 = 足した橋** / 青の細線 = マスクの面の縁。
       つなぎ: 黒 = 線 / 塗り2色 = つないだ2つの提案 / **赤 = 2つが墨を挟まず接する所(ここでつないだ)** / 青の細線 = マスクの面の縁。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m102_b_standard_v2 as m102                                    # noqa: E402
import m119_mask_faces_and_lines as m119                             # noqa: E402
import m122_line_only_faces as m122                                  # noqa: E402
import m126_line_ends as m126                                        # noqa: E402

SRC = m126.OUT
OUT = Path('results/m127_line_end_crops')
PAD = 120
NMAX = 12
K3 = np.ones((3, 3), np.uint8)


def edges(F):
    return (F > 0) & (F != cv2.erode(F.astype(np.float32), K3).astype(np.int32))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, _ = loader(kinds)
        key, H, W, ink = pg['key'], pg['H'], pg['W'], pg['ink']
        Fl, _, _, _ = m122.build_truth(pg, kinds)
        arms = json.load(open(SRC / f'{key}_arms.json'))
        labB = np.load(SRC / f'{key}_labB.npy')
        rng = np.random.default_rng(0)
        br = arms['bridges']
        sel = sorted(rng.permutation(len(br))[:NMAX].tolist()) if len(br) > NMAX else list(range(len(br)))
        print(f'{key}: 橋 {len(br)} 本 → 切り出し {len(sel)}')
        for n, t in enumerate(sel):
            (ay, ax), (by, bx) = br[t]
            Y0, Y1 = int(max(0, min(ay, by) - PAD)), int(min(H, max(ay, by) + PAD + 1))
            X0, X1 = int(max(0, min(ax, bx) - PAD)), int(min(W, max(ax, bx) + PAD + 1))
            im = np.full((Y1 - Y0, X1 - X0, 3), 255, np.uint8)
            im[edges(Fl[Y0:Y1, X0:X1])] = (230, 80, 0)
            im[ink[Y0:Y1, X0:X1]] = 0
            cv2.line(im, (int(ax) - X0, int(ay) - Y0), (int(bx) - X0, int(by) - Y0), (0, 0, 230), 3)
            cv2.imwrite(str(OUT / f'{key}_bridge{n + 1:02d}_x{X0}_y{Y0}.png'), im)
        jn = arms['joins']
        sel = sorted(rng.permutation(len(jn))[:NMAX].tolist()) if len(jn) > NMAX else list(range(len(jn)))
        print(f'{key}: つなぎ {len(jn)} 組 → 切り出し {len(sel)}')
        col = np.random.default_rng(1).integers(150, 250, (int(labB.max()) + 1, 3)).astype(np.uint8)
        for n, t in enumerate(sel):
            p, q = jn[t]
            seam = np.zeros(labB.shape, bool)
            a, b = labB[:, :-1], labB[:, 1:]
            seam[:, :-1] |= ((a == p) & (b == q)) | ((a == q) & (b == p))
            a, b = labB[:-1, :], labB[1:, :]
            seam[:-1, :] |= ((a == p) & (b == q)) | ((a == q) & (b == p))
            ys, xs = np.nonzero(seam)
            Y0, Y1 = max(0, ys.min() - PAD), min(H, ys.max() + 1 + PAD)
            X0, X1 = max(0, xs.min() - PAD), min(W, xs.max() + 1 + PAD)
            L = labB[Y0:Y1, X0:X1]
            im = np.full(L.shape + (3,), 255, np.uint8)
            for i in (p, q):
                im[L == i] = col[i]
            im[edges(Fl[Y0:Y1, X0:X1])] = (230, 80, 0)
            im[cv2.dilate(seam[Y0:Y1, X0:X1].astype(np.uint8), K3, iterations=2).astype(bool)] = (0, 0, 230)
            im[ink[Y0:Y1, X0:X1]] = 0
            cv2.imwrite(str(OUT / f'{key}_join{n + 1:02d}_P{p}_Q{q}_x{X0}_y{Y0}.png'), im)
        del pg, Fl, labB


if __name__ == '__main__':
    main()
