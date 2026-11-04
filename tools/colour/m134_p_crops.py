"""追補223 §D H5: P(ズレた組を繋ぐ)の線を、m132 と同じ文脈つきの形で切り出す。点数は出さない。

各頁から無作為に最大 12 本(seed 0)。1本につき 2×2 の合成画像:
  左上 = 文脈(延長の周り ±400 px を 1/2 に縮小。灰色の枠 = 右上の範囲)
  右上 = 原寸(±100 px、m131 と同じ)
  左下 = 面: P なし(B)   右下 = 面: P あり(B+P)  (下段は左上と同じ範囲・縮尺)
赤い丸 = P の線の位置(上段)。
色: 黒 = 線 / 赤 = P の線 / 青の細線 = マスクの面の縁(上段)/ 下段は提案の面ごとに色、
    **灰色の斜線 = 背景へ漏れて評価域から外れた面**。
    右下は延長で変わらない面を左下と同じ色にし、割れてできた片だけ新しい色にする。
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
import m133_offset_pairs as m133                                     # noqa: E402

OUT = Path('results/m134_p_crops')
PAD, WIDE, NMAX = 100, 400, 12
K3 = np.ones((3, 3), np.uint8)


def palette(n, seed=1):
    """彩度の高い色だけ(灰色と紛れないように)。"""
    rng = np.random.default_rng(seed)
    hsv = np.stack([rng.integers(0, 180, n), rng.integers(110, 200, n), rng.integers(190, 250, n)], 1)
    return cv2.cvtColor(hsv.astype(np.uint8)[None], cv2.COLOR_HSV2BGR)[0]


def match_colours(labB, labT, nB):
    """B+T の面に色番号を振る: 最も重なる B の面の色を、最初の1つだけ引き継ぐ。残り(割れた片)は新しい色。"""
    both = (labB > 0) & (labT > 0)
    nT = int(labT.max()) + 1
    key = labT[both].astype(np.int64) * (nB + 1) + labB[both]
    u, c = np.unique(key, return_counts=True)
    best = np.zeros(nT, np.int64); cnt = np.zeros(nT, np.int64)
    for k, v in zip(u, c):
        t, b = divmod(int(k), nB + 1)
        if v > cnt[t]:
            best[t], cnt[t] = b, v
    col = np.zeros(nT, np.int64); used = set(); nxt = nB + 1
    for t in np.argsort(-cnt):
        if t == 0:
            continue
        if best[t] and best[t] not in used:
            col[t] = best[t]; used.add(int(best[t]))
        else:
            col[t] = nxt; nxt += 1
    return col


def face_panel(lab, inside, ink, col, box, pal):
    Y0, Y1, X0, X1 = box
    L = lab[Y0:Y1, X0:X1]
    im = pal[col[L] % len(pal)].copy()
    yy, xx = np.indices(L.shape)
    out = ~inside[L]
    im[out] = 225
    im[out & (((yy + xx) // 6) % 3 == 0)] = 150
    im[L == 0] = 255
    im[ink[Y0:Y1, X0:X1]] = 0
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    pal = palette(4096)
    for loader in (m119.load_page0002, m119.load_gakusai3):
        pg, _ = loader(kinds)
        key, H, W = pg['key'], pg['H'], pg['W']
        Fl, _, _, _ = m122.build_truth(pg, kinds)
        U = Fl > 0
        inkB, _, _, _, _ = m130.build_b(pg['ink'])
        segs = json.load(open(m133.OUT / f'{key}_segs.json'))['segs']
        inkT = m130.draw_segs(inkB, [tuple(map(tuple, s)) for s in segs])
        labs = {}
        for name, ink_, src in (('B', inkB, Path('results/m126_line_ends') / f'{key}_labB.npy'),
                                ('T', inkT, m133.OUT / f'{key}_labP.npy')):
            lab = np.load(src)
            keep, nout = m122.in_domain(lab, U)
            inside = np.zeros(int(lab.max()) + 1, bool)
            inside[keep] = True
            labs[name] = (lab, inside, ink_)
            print(f'{key} {name}: 面 {int(lab.max())} / 評価域の外 {nout}')
        nB = int(labs['B'][0].max())
        colB = np.arange(nB + 1)
        colT = match_colours(labs['B'][0], labs['T'][0], nB)
        rng = np.random.default_rng(0)
        sel = sorted(rng.permutation(len(segs))[:NMAX].tolist()) if len(segs) > NMAX else list(range(len(segs)))
        edge = (Fl > 0) & (Fl != cv2.erode(Fl.astype(np.float32), K3).astype(np.int32))
        for n, t in enumerate(sel):
            (ay, ax), (by, bx) = segs[t]
            cy, cx = (ay + by) // 2, (ax + bx) // 2
            # 原寸(m131 と同じ範囲)
            Y0, Y1 = max(0, min(ay, by) - PAD), min(H, max(ay, by) + PAD + 1)
            X0, X1 = max(0, min(ax, bx) - PAD), min(W, max(ax, bx) + PAD + 1)
            def lines(box, thick):
                y0, y1, x0, x1 = box
                im = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
                im[edge[y0:y1, x0:x1]] = (230, 80, 0)
                im[inkB[y0:y1, x0:x1]] = 0
                cv2.line(im, (ax - x0, ay - y0), (bx - x0, by - y0), (0, 0, 230), thick)
                cv2.circle(im, (cx - x0, cy - y0), 12 * thick // 3 + 14, (0, 0, 230), max(2, thick // 2))  # P は短いので位置に丸
                return im
            near = lines((Y0, Y1, X0, X1), 3)
            wy0, wy1 = max(0, cy - WIDE), min(H, cy + WIDE)
            wx0, wx1 = max(0, cx - WIDE), min(W, cx + WIDE)
            wide = lines((wy0, wy1, wx0, wx1), 7)
            cv2.rectangle(wide, (X0 - wx0, Y0 - wy0), (X1 - wx0, Y1 - wy0), (120, 120, 120), 4)
            wide = cv2.resize(wide, ((wx1 - wx0) // 2, (wy1 - wy0) // 2), interpolation=cv2.INTER_AREA)
            wb = (wy0, wy1, wx0, wx1)
            half = ((wx1 - wx0) // 2, (wy1 - wy0) // 2)
            fb = face_panel(*labs['B'], colB, wb, pal)
            ft = face_panel(*labs['T'], colT, wb, pal)
            cv2.line(ft, (ax - wx0, ay - wy0), (bx - wx0, by - wy0), (0, 0, 230), 7)
            fb = cv2.resize(fb, half, interpolation=cv2.INTER_NEAREST)
            ft = cv2.resize(ft, half, interpolation=cv2.INTER_NEAREST)
            h = max(wide.shape[0], near.shape[0])
            def padto(im, hh, ww):
                o = np.full((hh, ww, 3), 255, np.uint8)
                o[:im.shape[0], :im.shape[1]] = im
                return o
            w0 = max(wide.shape[1], fb.shape[1])
            w1 = max(near.shape[1], ft.shape[1])
            gap = 12
            top = np.hstack([padto(wide, h, w0), np.full((h, gap, 3), 255, np.uint8), padto(near, h, w1)])
            hb = fb.shape[0]
            bot = np.hstack([padto(fb, hb, w0), np.full((hb, gap, 3), 255, np.uint8), padto(ft, hb, w1)])
            if bot.shape[1] < top.shape[1]:
                bot = padto(bot, hb, top.shape[1])
            elif top.shape[1] < bot.shape[1]:
                top = padto(top, h, bot.shape[1])
            im = np.vstack([top, np.full((gap, top.shape[1], 3), 255, np.uint8), bot])
            cv2.imwrite(str(OUT / f'{key}_p{n + 1:02d}.png'), im)
        del pg, Fl, inkB, inkT, labs, edge


if __name__ == '__main__':
    main()
