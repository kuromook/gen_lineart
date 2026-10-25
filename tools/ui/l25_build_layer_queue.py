# -*- coding: utf-8 -*-
"""L25: **層ごとの正解**を作る待ち行列と切り出し (Track L、2026-10-25)。

事前登録は `doc/work_log.md § 2026-10-25 L25 事前登録`(`e87a70c`)。
**標本の取り方はそこで固定済み。答を見てから選び直さない。**

見せるのは**画素だけ** —— 原寸 1:1 の切り出しと、頁の中での位置。
**層名も `preview_dark` も `keep4` も `black_among_drawn` も画面に出さない**
(どれも答を予測させてしまう)。

  python tools/ui/l25_build_layer_queue.py
"""
import csv, json, os, sys
from multiprocessing import Pool

import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, os.path.join(HERE, '..', 'evaluation'))
import l14_known_cells as K

INV = K.INV
BASE = '/home/sh1/deepl/lineart/data/incoming_manuscript'
OUT = '/home/sh1/deepl/lineart-fill-region/results/l25_layer_ui'
SEED = 20261025
MAXSIDE, CTXSIDE = 1500, 1100
# **事前登録の階級**(上から順に当てはめ、層は1つの階級にしか入れない)
STRATA = [('A_preview_dark', 20), ('B_keep4', 20), ('C_bad_max', 20), ('D_anchor', 15)]
WORDS = ['線', 'ベタ', '両方が混じっている', '絵ではない(マスク・紙・下地)', '分からない']


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def stratum(r, kk):
    pd = fnum(r['preview_dark'])
    ba = fnum(r['black_among_drawn'])
    k4 = fnum(kk['keep4']) if kk else None
    if pd is not None and 0.30 <= pd < 0.70:
        return 'A_preview_dark'
    if k4 is not None and 0.90 <= k4 <= 1.00:
        return 'B_keep4'
    if ba is not None and 0.95 <= ba <= 1.00:
        return 'C_bad_max'
    return 'D_anchor'


def render(job):
    pid, work, lid, sid = job
    f = f'{BASE}/{work}/all_layers/{pid}_L{lid}.png'
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    ink = im < 128
    n = int(ink.sum())
    if n == 0:
        return None
    ys, xs = np.nonzero(ink)
    H, W = ink.shape
    y0, y1 = max(0, int(ys.min()) - 30), min(H, int(ys.max()) + 31)
    x0, x1 = max(0, int(xs.min()) - 30), min(W, int(xs.max()) + 31)
    # ---- 原寸 1:1。大きすぎる層は**縮めず**、墨の重心のまわりを切る
    cy, cx = int(np.median(ys)), int(np.median(xs))
    if y1 - y0 > MAXSIDE:
        y0 = max(0, min(H - MAXSIDE, cy - MAXSIDE // 2)); y1 = y0 + MAXSIDE
    if x1 - x0 > MAXSIDE:
        x0 = max(0, min(W - MAXSIDE, cx - MAXSIDE // 2)); x1 = x0 + MAXSIDE
    sub = ink[y0:y1, x0:x1]
    img = np.full(sub.shape + (3,), 255, np.uint8)
    img[sub] = (20, 20, 20)
    name = f'{work}_{pid}_L{lid}.png'
    cv2.imwrite(f'{OUT}/crops/{name}', img)
    # ---- 頁の中での位置 (縮小してよい。**位置を見せるためのもの**)
    s = CTXSIDE / max(H, W)
    ctx = np.full((int(H * s), int(W * s), 3), 255, np.uint8)
    small = cv2.resize(ink.astype(np.uint8) * 255, (ctx.shape[1], ctx.shape[0]),
                       interpolation=cv2.INTER_AREA)
    ctx[small > 40] = (20, 20, 20)
    cv2.rectangle(ctx, (int(x0 * s) - 2, int(y0 * s) - 2),
                  (int(x1 * s) + 2, int(y1 * s) + 2), (0, 0, 255), 2)
    cv2.imwrite(f'{OUT}/crops/ctx_{name}', ctx)
    return dict(work=work, pid=pid, lid=int(lid), crop=name, ctx=f'ctx_{name}',
                area=n, h=int(y1 - y0), w=int(x1 - x0), page_h=H, page_w=W,
                scale=1.0, ctx_scale=round(s, 4), why=sid,
                frac_of_page=round(n / (H * W), 5), cropped=int(sub.shape != ink.shape))


def main():
    os.makedirs(f'{OUT}/crops', exist_ok=True)
    k4 = {(r['page_id'], r['layer_id']): r for r in csv.DictReader(open(K.KEEP4))}
    pool_rows = []
    for r in csv.DictReader(open(INV, encoding='utf-8-sig')):
        if r['folder'] != 'False' or r['rendered'] != 'True':
            continue
        if (fnum(r['drawn_ratio']) or 0) <= 0:
            continue
        kk = k4.get((r['page_id'], r['layer_id']))
        pool_rows.append((r, kk, stratum(r, kk)))
    by = {}
    for r, kk, s in pool_rows:
        by.setdefault(s, []).append((r, kk))
    print('母集団(階級ごと):', {k: len(v) for k, v in sorted(by.items())})
    rng = np.random.default_rng(SEED)
    picked = []
    for sid, want in STRATA:
        cand = sorted(by.get(sid, []), key=lambda t: (t[0]['page_id'], int(t[0]['layer_id'])))
        if not cand:
            print(f'  {sid}: 候補 0。この階級は空', flush=True)
            continue
        idx = rng.permutation(len(cand))[:min(want, len(cand))]
        for i in idx:
            r, kk = cand[int(i)]
            w = 'ako6' if '_ako6_' in r['page_id'] else 'ako7'
            picked.append((r['page_id'], w, r['layer_id'], sid))
        print(f'  {sid}: 候補 {len(cand)} -> 抽出 {min(want, len(cand))}')
    print(f'抽出 {len(picked)} 層。切り出しを作る', flush=True)
    rows = []
    with Pool(4) as p:
        for i, d in enumerate(p.imap_unordered(render, picked), 1):
            if d is not None:
                rows.append(d)
            if i % 20 == 0:
                print(f'  {i}/{len(picked)}', flush=True)
    rows.sort(key=lambda r: (r['work'], r['pid'], r['lid']))   # **並びを決定的に**
    rng2 = np.random.default_rng(SEED)
    order = rng2.permutation(len(rows))
    q = [rows[int(i)] for i in order]
    for i, r in enumerate(q):
        r['order'] = i
        r['cands'] = []            # 層名も統計も渡さない
        r['word_share'] = {}
        r['top_word'] = ''
        r['purity'] = 0.0; r['n_lay'] = 0; r['n_lay90'] = 0
        r['auto'] = 0
        r['comp'] = 0; r['region'] = r['lid']
    import collections
    cnt = collections.Counter(r['why'] for r in q)
    json.dump(dict(n=len(q), n_ask=len(q), n_check=0, n_auto=0, n_all=len(q),
                   words=WORDS,
                   ask='このレイヤは何?',
                   help='1枚のレイヤの墨だけを原寸 1:1 で出している。'
                        'C で頁のどこにあるか(赤枠が今見ている範囲)。'
                        '迷ったら M で保留して後に回せる。',
                   others=[['mixed', '保留(後で見直す)', 'M']],
                   queue=q),
              open(f'{OUT}/queue.json', 'w'), ensure_ascii=False)
    print(f'-> {OUT}/queue.json  行列 {len(q)}  階級 {dict(cnt)}')
    print('**階級は UI に送らない**(`why` はサーバが落とす)')


if __name__ == '__main__':
    main()
