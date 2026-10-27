# -*- coding: utf-8 -*-
"""L21 の帰無値: **前処理器 (`LineartAnimeDetector`) が同じコマから作る線の太さ**。

規律「評価のベースラインは常に前処理器の素の出力」をここにも当てる。
作者の材料の太さを測っただけでは「きれい/汚い」は言えない —— **下流が実際に
食べている線の太さ**が比較対象になる。入力は「コマの見た目」(門を通った層の墨を
黒、それ以外を白)を学習解像度に落としたもの。器具は本体と同じ `ridge_width`。

  python tools/evaluation/l21_null_preproc.py --n 24
"""
import argparse, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l21_dirtiness import (TRAIN_SIDE, WCUT, acc_w, add_H, cum, quant,
                           pages_with_window, ridge_width)
from l17_known_cells import assign_top, jobs, zmap
from l17_known_cells import arms  # T131: 腕と標準は1か所(KC57)

OUT = '/home/sh1/deepl/lineart-fill-region/results/l21_dirtiness'
SEED = 20261025


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=24, help='コマの枚数 (プールごとに半分)')
    a = ap.parse_args()
    import torch
    from PIL import Image
    from controlnet_aux import LineartAnimeDetector
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    det = LineartAnimeDetector.from_pretrained('lllyasviel/Annotators').to(dev)
    zz = zmap(); fill, line = arms('gated')
    js = jobs(zz, fill, line)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(js))
    H = {'ako6': acc_w(np.zeros(0, np.float32)), 'ako7': acc_w(np.zeros(0, np.float32))}
    Hm = {'ako6': acc_w(np.zeros(0, np.float32)), 'ako7': acc_w(np.zeros(0, np.float32))}
    n = {'ako6': 0, 'ako7': 0}
    os.makedirs(f'{OUT}/preproc', exist_ok=True)
    for oi in order:
        if min(n.values()) >= a.n // 2:
            break
        job = js[int(oi)]
        for (pid, work, ci, P, glab, gids, gstat,
             sub, (y0, x0, y1, x1), fl, lines) in pages_with_window(job):
            if n[work] >= a.n // 2:
                break
            lab, keys = assign_top(sub, P.shape)
            inkP = (lab > 0) & P
            if inkP.sum() < 50000:
                continue
            # 「コマの見た目」を学習解像度に落とす
            img = np.full(P.shape, 255, np.uint8)
            img[inkP] = 0
            img[~P] = 255
            h, w = img.shape
            s = TRAIN_SIDE / max(h, w)
            small = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                               interpolation=cv2.INTER_AREA)
            # 材料そのものを同じ縮小で見たときの太さ (比較の片方)
            acc_w(ridge_width(small < 128), Hm[work])
            rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
            out = det(Image.fromarray(rgb), detect_resolution=max(small.shape),
                      image_resolution=max(small.shape))
            o = np.array(out.convert('L'))
            if o.shape != small.shape:
                o = cv2.resize(o, (small.shape[1], small.shape[0]),
                               interpolation=cv2.INTER_AREA)
            acc_w(ridge_width(o > 127), H[work])     # 前処理器は白線・黒地
            if n[work] == 0:
                cv2.imwrite(f'{OUT}/preproc/{work}_{pid}_c{ci}_material.png', small)
                cv2.imwrite(f'{OUT}/preproc/{work}_{pid}_c{ci}_preproc.png', 255 - o)
            n[work] += 1
            print(f'  {work} {pid} コマ{ci}  {small.shape[1]}x{small.shape[0]}', flush=True)
    print(f'\nコマ {n}')
    print('帰無値: 学習解像度 (長辺 %d) での線の太さ' % TRAIN_SIDE)
    for work in ('ako6', 'ako7'):
        for nm, A in (('材料(作者の層をそのまま縮小)', Hm[work]),
                      ('前処理器の出力', H[work])):
            if A['len'].sum() == 0:
                continue
            for key, u in (('len', '長さ'), ('ink', '墨')):
                q = quant(A, key); cc = cum(A, key, WCUT)
                print(f'  {work}  {nm:<24}{u}で重み  中央 {q[1]:.0f}px   '
                      + '  '.join(f'<={t}px {v:.4f}' for t, v in zip(WCUT, cc)))
    print(f'-> {OUT}/preproc/ (両方の見本画像)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
