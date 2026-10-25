# -*- coding: utf-8 -*-
"""L21: 起案に付ける**原寸 1:1** の切り出しを作る (規律6 / `feedback_send_native_crops`)。

L19 の切り出しは長辺 1400 で縮めてあるので証拠に使えない。ここは縮めない。

  python tools/evaluation/l21_crops.py --crops <L19の切り出し名> ...
  python tools/evaluation/l21_crops.py --answers 黒消し 隙間埋め --max 4

出力 results/l21_dirtiness/crops/<name>.png  (原寸。長辺が大きすぎる面は中心 1600px)
"""
import argparse, json, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l21_dirtiness import pages_with_window, NAMES
from l17_known_cells import assign_top, gated_layers, jobs, zmap, adj_sets, D1
from l15_known_cells import boundary

OUT = '/home/sh1/deepl/lineart-fill-region/results/l21_dirtiness/crops'
Q = '/home/sh1/deepl/lineart-fill-region/results/l19_ui/queue.json'
LAB = '/home/sh1/deepl/lineart-fill-region/labels/fill_words.jsonl'
MAXSIDE, PAD = 1600, 40


def answers():
    q = json.load(open(Q))['queue']
    crop_of = {r['order']: r['crop'] for r in q}
    out = {}
    for ln in open(LAB):
        r = json.loads(ln)
        if r['choice'] == 'free' and r.get('free'):
            r['choice'], r['word'] = 'word', r['free'].strip()
        out[crop_of[r['order']]] = r['word'] if r['choice'] == 'word' else r['choice']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', nargs='*', default=[])
    ap.add_argument('--answers', nargs='*', default=[])
    ap.add_argument('--max', type=int, default=4, help='答ごとの枚数')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ans = answers()
    want = set(a.crops)
    if a.answers:
        got = {k: 0 for k in a.answers}
        for c, v in sorted(ans.items()):
            if v in got and got[v] < a.max:
                want.add(c); got[v] += 1
    if not want:
        print('切り出しの指定が無い'); return 1
    pages = {c.split('_c')[0].split('_', 1)[1] for c in want}
    print(f'切り出し {len(want)} 枚 / 頁 {len(pages)}')
    zz = zmap(); fill, line = gated_layers()
    done = 0
    for job in jobs(zz, fill, line):
        if job[0] not in pages:
            continue
        k = 0
        for (pid, work, ci, P, glab, gids, gstat,
             sub, (y0, x0, y1, x1), fl, lines) in pages_with_window(job):
            lab, keys = assign_top(sub, P.shape)
            inkP = (lab > 0) & P
            for gi in gids:
                R = glab == gi
                b = boundary(R)
                vals, cnt = adj_sets(R, lab, P, ds=(D1,), bnd=b)[D1]
                if cnt.sum() == 0:
                    continue
                k += 1
                name = f'{work}_{pid}_c{ci}_r{gi}_{k:03d}.png'
                if name not in want:
                    continue
                ys, xs = np.nonzero(R)
                cy0 = max(0, int(ys.min()) - PAD); cy1 = min(P.shape[0], int(ys.max()) + 1 + PAD)
                cx0 = max(0, int(xs.min()) - PAD); cx1 = min(P.shape[1], int(xs.max()) + 1 + PAD)
                if cy1 - cy0 > MAXSIDE:            # **縮めない。中心を切る**
                    c = (cy0 + cy1) // 2
                    cy0, cy1 = c - MAXSIDE // 2, c + MAXSIDE // 2
                if cx1 - cx0 > MAXSIDE:
                    c = (cx0 + cx1) // 2
                    cx0, cx1 = c - MAXSIDE // 2, c + MAXSIDE // 2
                img = np.full((cy1 - cy0, cx1 - cx0, 3), 255, np.uint8)
                img[inkP[cy0:cy1, cx0:cx1]] = (40, 40, 40)      # 線墨 = 濃い灰
                img[R[cy0:cy1, cx0:cx1]] = (0, 0, 0)            # 聞いている面 = 黒
                img[~P[cy0:cy1, cx0:cx1]] = (245, 245, 245)     # コマの外
                tag = ans.get(name, '?')
                p = f'{OUT}/{tag}_{name}'
                cv2.imwrite(p, img)
                print(f'  {tag:<16}{img.shape[1]}x{img.shape[0]} 原寸  {p}')
                done += 1
    print(f'-> {done} 枚  {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
