# -*- coding: utf-8 -*-
"""L37 の **判別できない層**(`ov >= 0.80` で標準に入っている)を**原寸1:1で切り出す**。

登録 §3 の帰結そのまま: **作者に出す候補を作る。出す順はユーザー。
こちらの判断で標準から外さない。**

1層に 1枚。**縮小しない**(規律6 / `feedback_send_native_crops`)。配色:

  灰   = その層より下に在る墨(全面の層は積まない。L37 本体と同じ `below`)
  赤   = **その層の墨のうち、下の墨の上に在る画素**(= `ov` の分子)
  青   = その層の墨のうち、白紙の上に在る画素

**窓は層の墨の外接矩形**。`MAX_WIN` より大きければ中心で切り、
**切ったことをファイル名と log に書く**(窓が全部でないことを隠さない)。
"""
import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import COV_MAX, gated_layers
from l17_known_cells import zmap
from pid_work import work_of

INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
KEEP4 = '/home/sh1/deepl/lineart-fill-region/results/l10_keep4_all_layers/keep4.csv'
BASE = '/home/sh1/deepl/lineart/data/incoming_manuscript'
OUT = '/home/sh1/deepl/lineart-fill-region/results/l37_white_over_ink/crops'
CSV_IN = '/home/sh1/deepl/lineart-fill-region/results/l37_white_over_ink/over_ink.csv'
MAX_WIN = 1800
HI = 0.80


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [r for r in csv.DictReader(open(CSV_IN))
            if r['in_fill'] == '1' and float(r['ov']) >= HI]
    print(f'**標準に入っていて `ov >= {HI}` の層  {len(rows)} 層** -> 原寸1:1で切り出す')
    if not rows:
        print('  **0 層。この切り出しは空**')
        return

    zz = zmap()
    _gfill, line_arm = gated_layers()
    cov = {(r['page_id'], r['layer_id']): float(r['cov'])
           for r in csv.DictReader(open(KEEP4))}

    # 対象の頁の層だけを集める
    pids = {r['pid'] for r in rows}
    per = {}
    for r in csv.DictReader(open(INV, encoding='utf-8-sig')):
        pid = r['page_id']
        if pid not in pids or r['folder'] != 'False' or r['rendered'] != 'True':
            continue
        dr = r['drawn_ratio'].strip()
        if not dr or float(dr) <= 0:
            continue
        z = zz.get((pid, r['layer_id']))
        if z is None:
            continue
        w = work_of(pid, r.get('page_path'))
        p = f'{BASE}/{w}/all_layers/{pid}_L{r["layer_id"]}.png'
        if not os.path.exists(p):
            continue
        cv_ = cov.get((pid, r['layer_id']))
        per.setdefault(pid, []).append(
            (z, r['layer_id'], p, cv_ is not None and cv_ >= COV_MAX))

    for pid in sorted(pids):
        layers = sorted(per[pid], key=lambda t: t[0])
        want = {r['lid']: r for r in rows if r['pid'] == pid}
        below = None
        lines = None
        for k, v in line_arm.items():
            if k[0] != pid:
                continue
            for lid_, path_ in v:
                im_ = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)
                if im_ is None:
                    continue
                if lines is None:
                    lines = np.zeros(im_.shape, bool)
                if im_.shape == lines.shape:
                    lines |= im_ < 128
        print(f'  線の腕の層 {sum(len(v) for k, v in line_arm.items() if k[0] == pid)} '
              f'(背景に使う。**0 ならその旨を書く**)')
        for z, lid, path, wide in layers:
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            ink = im < 128
            if below is None:
                below = np.zeros(ink.shape, bool)
            if ink.shape != below.shape:
                continue
            if lid in want:
                r = want[lid]
                ys, xs = np.nonzero(ink)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                bh, bw = y1 - y0, x1 - x0
                cropped = bh > MAX_WIN or bw > MAX_WIN
                if cropped:
                    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
                    y0 = max(0, cy - MAX_WIN // 2); y1 = min(ink.shape[0], y0 + MAX_WIN)
                    x0 = max(0, cx - MAX_WIN // 2); x1 = min(ink.shape[1], x0 + MAX_WIN)
                w_ink = ink[y0:y1, x0:x1]
                w_bel = below[y0:y1, x0:x1]
                img = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
                img[w_bel] = (170, 170, 170)                      # 灰: 下の墨
                img[w_ink & ~w_bel] = (255, 80, 0)                # 青(BGR): 白紙の上
                img[w_ink & w_bel] = (0, 0, 220)                  # 赤(BGR): 下の墨の上
                tag = 'cut' if cropped else 'bbox'
                fn = (f'{pid}_L{lid}_{r["name"].replace(" ", "_")}'
                      f'_ov{float(r["ov"]):.4f}_{tag}_{x0}_{y0}_{x1 - x0}x{y1 - y0}.png')
                cv2.imwrite(f'{OUT}/{fn}', img)
                # **2枚目: 作者が「何が描いてあるか」を見られる窓**(規律6)。
                # 1枚目の灰は `below` —— この頁では `color` / `over lay` の塊で、
                # **線が1本も見えない。判定の材料にならない。**
                # 背景は**器具自身の線の腕**(`gated_layers()` の line)にする。
                # **名前で選ばない**(問い2で名前から材質は決まらないと作者が答えている)。
                # 対象の層は**塗り潰さず輪郭だけ赤**にして、下の絵を隠さない。
                img2 = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
                if lines is not None:
                    img2[lines[y0:y1, x0:x1]] = (40, 40, 40)
                ring = cv2.dilate(w_ink.astype(np.uint8), np.ones((9, 9), np.uint8))
                img2[(ring > 0) & ~w_ink] = (0, 0, 220)
                fn2 = fn[:-4] + '_line.png'
                cv2.imwrite(f'{OUT}/{fn2}', img2)
                print(f'    -> {OUT}/{fn2}  (**線の腕を背景に、対象は輪郭だけ**)')
                print(f'  ov {float(r["ov"]):.4f}  墨 {int(r["ink_px"]):>9,}  '
                      f'外接 {bw}x{bh}  窓 {x1 - x0}x{y1 - y0} '
                      f'{"(**中心で切った。窓は層の全部ではない**)" if cropped else "(外接矩形の全部)"}')
                print(f'    -> {OUT}/{fn}')
            if wide:
                continue
            below |= ink
    print()
    print('**灰 = 下の墨 / 赤 = その層の墨が下の墨の上に在る画素 / '
          '青 = その層の墨が白紙の上に在る画素。縮小していない(1:1)**')


if __name__ == '__main__':
    main()
