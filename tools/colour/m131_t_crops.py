"""追補219 §D H4: T が足した延長を原寸で切り出す(各頁、無作為に最大12本 seed 0)。点数は出さない。

色: 黒 = 線(B の橋を含む)/ **赤 = T が足した延長** / 青の細線 = マスクの面の縁。
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
import m130_t_junctions as m130                                      # noqa: E402

OUT = Path('results/m131_t_crops')
PAD = 100
NMAX = 12
K3 = np.ones((3, 3), np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, _ = loader(kinds)
        key, H, W = pg['key'], pg['H'], pg['W']
        Fl, _, _, _ = m122.build_truth(pg, kinds)
        inkB, _, _, _, _ = m130.build_b(pg['ink'])
        segs = json.load(open(m130.OUT / f'{key}_segs.json'))['segs']
        rng = np.random.default_rng(0)
        sel = sorted(rng.permutation(len(segs))[:NMAX].tolist()) if len(segs) > NMAX else list(range(len(segs)))
        print(f'{key}: 延長 {len(segs)} 本 → 切り出し {len(sel)}')
        edge = (Fl > 0) & (Fl != cv2.erode(Fl.astype(np.float32), K3).astype(np.int32))
        for n, t in enumerate(sel):
            (ay, ax), (by, bx) = segs[t]
            Y0, Y1 = max(0, min(ay, by) - PAD), min(H, max(ay, by) + PAD + 1)
            X0, X1 = max(0, min(ax, bx) - PAD), min(W, max(ax, bx) + PAD + 1)
            im = np.full((Y1 - Y0, X1 - X0, 3), 255, np.uint8)
            im[edge[Y0:Y1, X0:X1]] = (230, 80, 0)
            im[inkB[Y0:Y1, X0:X1]] = 0
            cv2.line(im, (ax - X0, ay - Y0), (bx - X0, by - Y0), (0, 0, 230), 3)
            cv2.imwrite(str(OUT / f'{key}_t{n + 1:02d}_x{X0}_y{Y0}.png'), im)
        del pg, Fl, inkB, edge


if __name__ == '__main__':
    main()
