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
SEED2 = 20261026        # L27: 追加の無作為60層(**枠は L25 の E_random と同じ**)
MAXSIDE, CTXSIDE = 1500, 1100
# **事前登録の階級**(上から順に当てはめ、層は1つの階級にしか入れない)
STRATA = [('A_preview_dark', 20), ('B_keep4', 20), ('C_bad_max', 20),
          ('D_anchor', 15), ('E_random', 15)]
REPEAT = 5          # 既知セル: 同じ層をもう一度出して、**人の答の再現性**を測る
# **答は2つの軸を混ぜない。** 形(線/ベタ)と役割(完成の絵か、工程のものか)は別。
# 2026-10-25 作者の指摘「下絵は『絵ではない』でいいのだろうか」で足した ——
# ペンで描いた下絵は**形は線、役割は完成の絵ではない**ので、どちらの箱にも入らなかった。
WORDS = ['線(完成の線画)', 'ベタ(完成の塗り)', '線とベタが混じっている',
         '下絵・ラフ(ペンでも。完成の絵ではない)',
         '絵ではない(マスク・紙・下地・トーン)', '分からない']


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def stratum(r, kk):
    """**上から順に当てはめ、層は1つの階級にしか入れない。**

    `E_random` はここでは付かない —— **境界と無関係に母集団から引く**階級なので、
    抽出の側で別に引く(基盤 `reply_l_layer_truth_registration_20261025.md` 条件2:
    無作為の階級が無いと、**境界から離れた所の誤り率が最後まで分からない**)。
    """
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


def already_shown():
    """L25 で既に見せた層。**追加の無作為はここを除いて引く**。"""
    p = f'{OUT}/queue.json'
    if not os.path.exists(p):
        return set()
    return {(r['pid'], str(r['lid'])) for r in json.load(open(p))['queue']}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--extend-random', type=int, default=0,
                    help='L27: 無作為を N 層だけ足す(**枠は L25 と同じ**)')
    a = ap.parse_args()
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
    if a.extend_random:
        shown = already_shown()
        cand = [(r, kk) for r, kk, _ in sorted(
            pool_rows, key=lambda t: (t[0]['page_id'], int(t[0]['layer_id'])))
            if (r['page_id'], r['layer_id']) not in shown]
        print(f'L27: 枠 {len(pool_rows)} 層 - 既出 {len(shown)} = 候補 {len(cand)}')
        rng = np.random.default_rng(SEED2)
        idx = rng.permutation(len(cand))[:a.extend_random]
        picked = []
        for i in idx:
            r, _kk = cand[int(i)]
            w = 'ako6' if '_ako6_' in r['page_id'] else 'ako7'
            picked.append((r['page_id'], w, r['layer_id'], 'E_random2'))
        print(f'  無作為 {len(picked)} 層を引いた (seed {SEED2})', flush=True)
        rows = []
        with Pool(4) as p:
            for i, d in enumerate(p.imap_unordered(render, picked), 1):
                if d is not None:
                    rows.append(d)
                if i % 20 == 0:
                    print(f'  {i}/{len(picked)}', flush=True)
        rows.sort(key=lambda r: (r['work'], r['pid'], r['lid']))
        rng2 = np.random.default_rng(SEED2)
        rep = [dict(rows[int(i)], why='E_random2_repeat')
               for i in rng2.permutation(len(rows))[:REPEAT]]
        rows2 = rows + rep
        q2 = [rows2[int(i)] for i in rng2.permutation(len(rows2))]
        old = json.load(open(f'{OUT}/queue.json'))
        for i, r in enumerate(q2):
            r.update(order=i, cands=[], word_share={}, top_word='', purity=0.0,
                     n_lay=0, n_lay90=0, auto=0, comp=0, region=r['lid'])
        old.update(n=len(q2), n_ask=len(q2), n_all=len(q2), queue=q2)
        json.dump(old, open(f'{OUT}/queue_l27.json', 'w'), ensure_ascii=False)
        import collections
        print(f'-> {OUT}/queue_l27.json  行列 {len(q2)}  '
              f'階級 {dict(collections.Counter(r["why"] for r in q2))}')
        return
    rng = np.random.default_rng(SEED)
    picked, taken = [], set()
    allr = sorted(pool_rows, key=lambda t: (t[0]['page_id'], int(t[0]['layer_id'])))
    by['E_random'] = [(r, kk) for r, kk, _ in allr]      # **母集団そのまま**から引く
    for sid, want in STRATA:
        cand = sorted(by.get(sid, []), key=lambda t: (t[0]['page_id'], int(t[0]['layer_id'])))
        if not cand:
            print(f'  {sid}: 候補 0。この階級は空', flush=True)
            continue
        got = 0
        for i in rng.permutation(len(cand)):
            if got >= want:
                break
            r, kk = cand[int(i)]
            key = (r['page_id'], r['layer_id'])
            if key in taken:        # **同じ層を2つの階級に入れない**
                continue
            taken.add(key); got += 1
            w = 'ako6' if '_ako6_' in r['page_id'] else 'ako7'
            picked.append((r['page_id'], w, r['layer_id'], sid))
        print(f'  {sid}: 候補 {len(cand)} -> 抽出 {got}')
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
    # 既知セル: REPEAT 枚を**もう一度出す**。同じ答が返るかで人の答の再現性を測る
    rep = [dict(rows[int(i)], why=rows[int(i)]['why'] + '_repeat')
           for i in rng2.permutation(len(rows))[:REPEAT]]
    rows2 = rows + rep
    order = rng2.permutation(len(rows2))
    q = [rows2[int(i)] for i in order]
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
                        '迷ったら M で保留、当てはまる語が無ければ T で入力。',
                   # **自由入力を残す** —— L19 では「黒消し」がここから出て語に昇格した
                   others=[['free', 'その他(語を入力)', 'T'],
                           ['mixed', '保留(後で見直す)', 'M']],
                   queue=q),
              open(f'{OUT}/queue.json', 'w'), ensure_ascii=False)
    print(f'-> {OUT}/queue.json  行列 {len(q)}  階級 {dict(cnt)}')
    print('**階級は UI に送らない**(`why` はサーバが落とす)')


if __name__ == '__main__':
    main()
