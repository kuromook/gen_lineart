# -*- coding: utf-8 -*-
"""L21: 起案に付ける**原寸 1:1** の切り出しを作る (規律6 / `feedback_send_native_crops`)。

L19 の切り出しは長辺 1400 で縮めてあるので証拠に使えない。ここは縮めない。

  python tools/evaluation/l21_crops.py --crops <L19の切り出し名> ...
  python tools/evaluation/l21_crops.py --answers 黒消し 隙間埋め --max 4
  python tools/evaluation/l21_crops.py --anomaly        UI の「⚑ 異常を報告」の全件

出力 results/l21_dirtiness/crops/<name>.png  (原寸。長辺が大きすぎる面は中心 1600px)
"""
import argparse, json, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l21_dirtiness import pages_with_window, NAMES
from l17_known_cells import assign_top, jobs, zmap, adj_sets, D1
from l15_known_cells import boundary
from l17_known_cells import arms  # T131: 腕と標準は1か所(KC57)

OUT = '/home/sh1/deepl/lineart-fill-region/results/l21_dirtiness/crops'
Q = '/home/sh1/deepl/lineart-fill-region/results/l19_ui/queue.json'
LAB = '/home/sh1/deepl/lineart-fill-region/labels/fill_words.jsonl'
FLAG = '/home/sh1/deepl/lineart-fill-region/labels/fill_anomaly.jsonl'
MAXSIDE, PAD = 1600, 40
PANEL_MAX = 3200          # コマ全体の切り出しもこの長辺までは**縮めない**


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
    ap.add_argument('--anomaly', action='store_true',
                    help='UI で「⚑ 異常を報告」が付いた面を全部。コマ全体も原寸で出す')
    ap.add_argument('--max', type=int, default=4, help='答ごとの枚数')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ans = answers()
    want = set(a.crops)
    note = {}
    if a.anomaly:
        if not os.path.exists(FLAG):
            print(f'{FLAG} が無い (まだ報告されていない)'); return 1
        for ln in open(FLAG):
            r = json.loads(ln)
            if r.get('note') == '__undo__':
                note.pop(r['crop'], None)
            else:
                note[r['crop']] = r['note']
        want |= set(note)
        print(f'異常の報告 {len(note)} 件')
    if a.answers:
        got = {k: 0 for k in a.answers}
        for c, v in sorted(ans.items()):
            if v in got and got[v] < a.max:
                want.add(c); got[v] += 1
    if not want:
        print('切り出しの指定が無い'); return 1
    pages = {c.split('_c')[0].split('_', 1)[1] for c in want}
    print(f'切り出し {len(want)} 枚 / 頁 {len(pages)}')
    zz = zmap(); fill, line = arms('gated')
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
                img[inkP[cy0:cy1, cx0:cx1]] = (40, 40, 40)      # 門が通した層の墨 = 濃い灰
                img[R[cy0:cy1, cx0:cx1]] = (0, 0, 0)            # 聞いている面 = 黒
                img[~P[cy0:cy1, cx0:cx1]] = (245, 245, 245)     # コマの外
                tag = ('異常_' + ans.get(name, '?')) if name in note else ans.get(name, '?')
                p = f'{OUT}/{tag}_{name}'
                cv2.imwrite(p, img)
                print(f'  {tag:<16}{img.shape[1]}x{img.shape[0]} 原寸  {p}'
                      + (f'   報告: {note[name]}' if name in note else ''))
                done += 1
                if name in note:
                    # **コマ全体も原寸で**。何が混じっているかは面だけでは見えない
                    ph, pw = P.shape
                    sy0, sy1, sx0, sx1 = 0, ph, 0, pw
                    if ph > PANEL_MAX:
                        c = (int(ys.min()) + int(ys.max())) // 2
                        sy0 = max(0, min(ph - PANEL_MAX, c - PANEL_MAX // 2)); sy1 = sy0 + PANEL_MAX
                    if pw > PANEL_MAX:
                        c = (int(xs.min()) + int(xs.max())) // 2
                        sx0 = max(0, min(pw - PANEL_MAX, c - PANEL_MAX // 2)); sx1 = sx0 + PANEL_MAX
                    pan = np.full((sy1 - sy0, sx1 - sx0, 3), 255, np.uint8)
                    pan[inkP[sy0:sy1, sx0:sx1]] = (40, 40, 40)
                    pan[R[sy0:sy1, sx0:sx1]] = (0, 0, 0)
                    pan[~P[sy0:sy1, sx0:sx1]] = (245, 245, 245)
                    cv2.rectangle(pan, (cx0 - sx0 - 3, cy0 - sy0 - 3),
                                  (cx1 - sx0 + 2, cy1 - sy0 + 2), (0, 0, 255), 5)
                    pp = f'{OUT}/異常コマ_{name}'
                    cv2.imwrite(pp, pan)
                    print(f'  {"":<16}{pan.shape[1]}x{pan.shape[0]} 原寸(コマ)  {pp}')
    print(f'-> {done} 枚  {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
