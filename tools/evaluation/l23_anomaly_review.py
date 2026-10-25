# -*- coding: utf-8 -*-
"""L23: 作者が「異常」と印を付けた面を、**UI の見せ方と原稿の見た目を並べて**出す。

作者の問い(2026-10-25):
> 表示されているのは複数のレイヤの合成だったのだろうか?
> しかし髪は合成されていないものや背景3Dが合成されているものなど事故に見えるものにチェックをいれた

L19 の切り出しは、こう作ってある:
  薄い灰 = **その頁の LINE 層を全部合成**(3D の線画層 `rendering_lineart_*` も含む)
  ほぼ黒 = **聞いている面1つだけ**。**他のベタ層は描いていない**
  **z 順を無視して、聞いている面を最後に塗る**(= 必ず手前に見える)

だから「髪のベタが無い」「3D が混ざる」「手前に在るのがおかしい」は、
**材料の事故ではなく見せ方の結果でありうる。** ここはそれを選り分けるために、
同じ面を 3 通りで原寸 1:1 に出す:

  1  UI             L19 とまったく同じ絵
  2  門を通った全層   FILL も LINE も、**z_global の順**に塗る。面を赤枠
  3  門が落とした層も  下絵・フキダシ・トーン・色も足して z 順。面を赤枠

**3 は「原稿の見た目」ではない。** 掃き出された png には `用紙`(全面が黒い紙の層)や
マスクが混ざっていて、素朴に z 順で塗ると**頁が真っ黒になる**(2026-10-25 に実際そうなった)。
ここでは **墨が頁の 95% を超える層を除いて**いるだけで、クリップもブレンドも再現していない。

  python tools/evaluation/l23_anomaly_review.py
"""
import csv, json, os, sys
from collections import defaultdict

import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l21_dirtiness import pages_with_window
from l17_known_cells import (D1, adj_sets, assign_top, gated_layers, jobs, zmap)
from l15_known_cells import boundary

INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
BASE = '/home/sh1/deepl/lineart/data/incoming_manuscript'
OUT = '/home/sh1/deepl/lineart-fill-region/results/l23_anomaly'
Q = '/home/sh1/deepl/lineart-fill-region/results/l19_ui/queue.json'
FLAG = '/home/sh1/deepl/lineart-fill-region/labels/fill_anomaly.jsonl'
LABF = '/home/sh1/deepl/lineart-fill-region/labels/fill_words.jsonl'
PAD, MAXSIDE = 80, 1800
PAL = [(60, 60, 255), (60, 200, 60), (255, 140, 0), (200, 60, 220), (0, 200, 220)]
SPECIAL = ('rendering', 'MASK', 'effect')      # 3D / マスク / 効果 は別の色で示す


def flags():
    got = {}
    for ln in open(FLAG):
        r = json.loads(ln)
        if r.get('note') == '__undo__':
            got.pop(r['order'], None)
        else:
            got[r['order']] = r
    return got


def labels():
    out = {}
    for ln in open(LABF):
        r = json.loads(ln)
        if r['choice'] == 'free' and r.get('free'):
            r['choice'], r['word'] = 'word', r['free'].strip()
        out[r['order']] = r
    return out


def all_layers(pid, work, names):
    """掃き出された層のうち、**全面を覆う層 (用紙など) を除いた**もの。

    除かないと `用紙` が頁を真っ黒に塗る。ここは「原稿の見た目」ではなく
    「門が落とした層も見えるようにした絵」である。
    """
    d = f'{BASE}/{work}/all_layers'
    out, drop = [], []
    for (p, lid), nm in names.items():
        if p != pid:
            continue
        f = f'{d}/{pid}_L{lid}.png'
        if not os.path.exists(f):
            continue
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        ink = im < 128
        if ink.sum() >= 0.95 * ink.size:
            drop.append(nm)
            continue
        out.append((lid, nm, ink))
    if drop:
        print(f'    (全面の層を除いた: {", ".join(x for x in drop if x)})')
    return out


def paint(layers, zz, pid, win, shape, mark=()):
    """z_global の低い方から塗り重ねる。mark に当たる層は色を変える。"""
    y0, x0, y1, x1 = win
    img = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    order = sorted(layers, key=lambda t: zz.get((pid, str(t[0])), 0))
    for lid, nm, ink in order:
        sub = ink[y0:y1, x0:x1]
        if not sub.any():
            continue
        c = (40, 40, 40)
        for i, s in enumerate(SPECIAL):
            if nm and nm.startswith(s):
                c = PAL[i % len(PAL)]
        img[sub] = c
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    names = {(r['page_id'], r['layer_id']): r['name']
             for r in csv.DictReader(open(INV, encoding='utf-8-sig'))
             if r['folder'] == 'False' and r['rendered'] == 'True'}
    fl = flags(); lab = labels()
    q = json.load(open(Q))['queue']
    byorder = {r['order']: r for r in q}
    want = {byorder[o]['crop']: o for o in fl}
    pages = {byorder[o]['pid'] for o in fl}
    print(f'異常の印 {len(fl)} 件 / 頁 {len(pages)}')
    zz = zmap(); fill, line = gated_layers()
    rows = []
    for job in jobs(zz, fill, line):
        if job[0] not in pages:
            continue
        pid0, work0 = job[0], job[1]
        every = all_layers(pid0, work0, names)
        k = 0
        for (pid, work, ci, P, glab, gids, gstat,
             sub, (y0, x0, y1, x1), flr, lines) in pages_with_window(job):
            lab2, keys = assign_top(sub, P.shape)
            inkP = (lab2 > 0) & P
            for gi in gids:
                R = glab == gi
                b = boundary(R)
                vals, cnt = adj_sets(R, lab2, P, ds=(D1,), bnd=b)[D1]
                if cnt.sum() == 0:
                    continue
                k += 1
                nm = f'{work}_{pid}_c{ci}_r{gi}_{k:03d}.png'
                if nm not in want:
                    continue
                o = want[nm]
                ys, xs = np.nonzero(R)
                cy0 = max(0, int(ys.min()) - PAD); cy1 = min(P.shape[0], int(ys.max()) + 1 + PAD)
                cx0 = max(0, int(xs.min()) - PAD); cx1 = min(P.shape[1], int(xs.max()) + 1 + PAD)
                for lo, hi, mx in ((cy0, cy1, P.shape[0]), (cx0, cx1, P.shape[1])):
                    pass
                if cy1 - cy0 > MAXSIDE:
                    c = (cy0 + cy1) // 2; cy0, cy1 = max(0, c - MAXSIDE // 2), max(0, c - MAXSIDE // 2) + MAXSIDE
                if cx1 - cx0 > MAXSIDE:
                    c = (cx0 + cx1) // 2; cx0, cx1 = max(0, c - MAXSIDE // 2), max(0, c - MAXSIDE // 2) + MAXSIDE
                win = (cy0, cx0, cy1, cx1)
                # --- 1. UI とまったく同じ絵
                a = np.full((cy1 - cy0, cx1 - cx0, 3), 255, np.uint8)
                a[inkP[cy0:cy1, cx0:cx1]] = (205, 205, 205)
                a[R[cy0:cy1, cx0:cx1]] = (26, 26, 26)
                dt = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2,
                                           cv2.DIST_MASK_PRECISE)
                near = (dt <= D1) & P
                order2 = np.argsort(-cnt)
                for i, oo in enumerate(order2[:5]):
                    a[(near & (lab2 == int(vals[oo])))[cy0:cy1, cx0:cx1]] = PAL[i % len(PAL)]
                # --- 2. 門を通った全層を z 順
                g = [(lid, names.get((pid, str(lid))), ink[y0:y1, x0:x1])
                     for lid, ink in flr] + \
                    [(lid, names.get((pid, str(lid))), ink) for lid, ink, z in lines]
                bimg = paint(g, zz, pid, win, P.shape)
                # --- 3. 納品の全層を z 順 (= 原稿の見た目)
                ev = [(lid, nm2, ink[y0:y1, x0:x1]) for lid, nm2, ink in every]
                cimg = paint(ev, zz, pid, win, P.shape)
                for im in (bimg, cimg):
                    im[~P[cy0:cy1, cx0:cx1]] = (248, 248, 248)
                    cv2.rectangle(im, (int(xs.min()) - cx0 - 6, int(ys.min()) - cy0 - 6),
                                  (int(xs.max()) - cx0 + 6, int(ys.max()) - cy0 + 6),
                                  (0, 0, 255), 4)
                note = fl[o]['note']
                v = lab[o]
                tag = f'Q{o + 1:03d}'
                for sfx, im in (('1_UI', a), ('2_門を通った全層_z順', bimg),
                                ('3_門が落とした層も_z順', cimg)):
                    cv2.imwrite(f'{OUT}/{tag}_{sfx}.png', im)
                # --- 層の表
                zr = [(zz.get((pid, str(lid)), 0), lid, names.get((pid, str(lid))),
                       int((ink[y0:y1, x0:x1] & P).sum()) if ink.shape != P.shape
                       else int((ink & P).sum()))
                      for lid, nm2, ink in every]
                zr = [t for t in zr if t[3] > 0]
                zr.sort(key=lambda t: -t[0])
                zreg = max((zz.get((pid, str(lid)), 0) for lid, ink in flr
                            if (ink[y0:y1, x0:x1] & R).any()), default=None)
                rows.append(dict(order=o, q=o + 1, crop=nm, note=note,
                                 answer=(v['word'] if v['choice'] == 'word' else v['choice']),
                                 pid=pid, work=work, comp=ci, region=gi,
                                 area=int(R.sum()), z_region=zreg,
                                 n_layer_in_panel=len(zr),
                                 above=[t[2] for t in zr if zreg is not None and t[0] > zreg][:6],
                                 special=[t[2] for t in zr
                                          if t[2] and t[2].startswith(SPECIAL)][:6]))
                print(f'  Q{o + 1:03d}  {v["word"] or v["choice"]:<8}{note[:34]:<36}'
                      f'  層 {len(zr)}  面の z {zreg}')
    rows.sort(key=lambda r: r['q'])
    json.dump(rows, open(f'{OUT}/review.json', 'w'), ensure_ascii=False, indent=1)
    print(f'\n-> {OUT}/  (Q***_1_UI / _2_門を通った全層_z順 / _3_門が落とした層も_z順)')
    print(f'-> {OUT}/review.json')
    print('\n面より手前 (z_global が大きい) に在る層 —— **UI では手前に描けていなかったもの**')
    for r in rows:
        if r['above']:
            print(f'  Q{r["q"]:03d}  {", ".join(x for x in r["above"] if x)}')
    print('\n3D / MASK / effect がコマに在る問')
    for r in rows:
        if r['special']:
            print(f'  Q{r["q"]:03d}  {", ".join(x for x in r["special"] if x)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
