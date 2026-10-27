# -*- coding: utf-8 -*-
"""L19: ベタ面から単語を手で決める UI の待ち行列と切り出しを作る。

L17/L18 の測定が出した設計値をそのまま使う:
  候補 = ∂R から 4px 以内の**門が通した線の形の層**の墨を、z_global で最上位の層に割り当てて数えた上位
  要判断 = 層が2枚以上触り purity < 0.90 の面 (1頁あたり 6面、その8割が2択)
  **自動で決まる面(単層 or purity>=0.90)も 10% を混ぜて出す** —— 自動側の誤り率は
  こちらでは原理的に測れない(purity は曖昧でないことの測度)。作者の答だけが作れる

出力:
  results/l19_ui/crops/<work>_<pid>_c<comp>_r<region>.png   原寸の切り出し(長辺 1400 で頭打ち)
  results/l19_ui/queue.json                                  面ごとの候補と出自
"""
import json, os, sys, csv
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, os.path.join(HERE, '..', 'evaluation'))
from l15_known_cells import boundary
from l17_known_cells import (arms, D1, adj_sets, assign_top, jobs,
                             panels_and_regions, place, SEEDS, zmap)

INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
OUT = '/home/sh1/deepl/lineart-fill-region/results/l19_ui'
MAXSIDE, PAD, NCAND = 1400, 60, 5
AUTO_SAMPLE, SEED = 0.10, 20261025
# 解答欄は**語**。レイヤ名は色のヒント側に回す(2026-10-25 作者の指摘)。
# 対応は作者の層名から引いたもので、粒度は作者が決めた(規律6)
WORDS = [('髪', ('線画 hair', '線画 HEAD hair')),
         ('肌', ('線画 skin', '線画 HEAD skin')),
         ('顔(目鼻口)', ('線画 face', '線画 HEAD face')),
         ('服', ('線画 cloth', '線画 BODY cloth')),
         ('影', ('shadow',)),
         ('背景・3D', ('rendering_lineart_front0001', 'rendering_lineart_middle0001',
                       'rendering0001')),
         ('効果', ('effect',)),
         ('描き文字', ('描き文字　　白フチ',)),
         # 作者の語彙。対応する層が無い語も解答欄に出す(2026-10-25 作者の指摘)。
         # **縁にその語の層が接していなくても押せる**ことが、この道で一番知りたい事例
         ('まつ毛', ()), ('眉', ()), ('フキダシ', ('フキダシ',)), ('黒消し', ())]
W_OF = {n: w for w, ns in WORDS for n in ns}
# 候補の色 (BGR)。UI の凡例と同じ順
PAL = [(60, 60, 255), (60, 200, 60), (255, 140, 0), (200, 60, 220), (0, 200, 220)]


def one_page(job):
    out = []
    k = 0                 # **コマ番号は枠ファイルごとに振り直されるので、頁内の通し番号で名前を一意にする**
    for pid, work, ci, P, glab, gids, gstat, sub in panels_and_regions(job):
        if not sub:
            continue
        lab, keys = assign_top(sub, P.shape)
        for gi in gids:
            R = glab == gi
            b = boundary(R)
            vals, cnt = adj_sets(R, lab, P, ds=(D1,), bnd=b)[D1]
            if cnt.sum() == 0:
                continue
            order = np.argsort(-cnt)
            tot = int(cnt.sum())
            cands = [dict(lid=int(keys[int(vals[o]) - 1]), label=int(vals[o]),
                          px=int(cnt[o]), share=round(float(cnt[o]) / tot, 4))
                     for o in order[:NCAND]]
            pur = cands[0]['share']
            s = np.sort(cnt)[::-1]
            nl90 = int(np.searchsorted(np.cumsum(s), 0.9 * s.sum()) + 1)
            placeable = place(glab, gi, gstat, P, SEEDS[0], pid, ci) is not None
            # ---- 切り出し
            ys, xs = np.nonzero(R)
            y0, y1 = max(0, ys.min() - PAD), min(P.shape[0], ys.max() + 1 + PAD)
            x0, x1 = max(0, xs.min() - PAD), min(P.shape[1], xs.max() + 1 + PAD)
            # **ベタ面は原稿と同じくほぼ黒。他の層の墨は薄い灰で退く。縁 4px だけ色**
            inkP = (lab > 0) & P
            img = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
            img[inkP[y0:y1, x0:x1]] = (205, 205, 205)
            img[R[y0:y1, x0:x1]] = (26, 26, 26)
            dt = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2,
                                       cv2.DIST_MASK_PRECISE)
            near = (dt <= D1) & P
            for i, c in enumerate(cands):
                m = near & (lab == c['label'])
                img[m[y0:y1, x0:x1]] = PAL[i % len(PAL)]
            h, w = img.shape[:2]
            sc = min(1.0, MAXSIDE / max(h, w))
            if sc < 1.0:
                img = cv2.resize(img, (int(w * sc), int(h * sc)),
                                 interpolation=cv2.INTER_AREA)
            k += 1
            name = f'{work}_{pid}_c{ci}_r{gi}_{k:03d}.png'
            cv2.imwrite(f'{OUT}/crops/{name}', img)
            # ---- 文脈: コマ全体。面を赤枠で囲む (キー C で切り替え)
            ctx = np.full(P.shape + (3,), 255, np.uint8)
            ctx[inkP] = (205, 205, 205)
            ctx[R] = (26, 26, 26)
            for i, c in enumerate(cands):
                ctx[near & (lab == c['label'])] = PAL[i % len(PAL)]
            ctx[~P] = (245, 245, 245)
            th = max(2, int(round(max(P.shape) / 400)))
            cv2.rectangle(ctx, (max(0, x0 - 4), max(0, y0 - 4)),
                          (min(P.shape[1] - 1, x1 + 3), min(P.shape[0] - 1, y1 + 3)),
                          (0, 0, 255), th)
            ch, cw = ctx.shape[:2]
            csc = min(1.0, MAXSIDE / max(ch, cw))
            if csc < 1.0:
                ctx = cv2.resize(ctx, (int(cw * csc), int(ch * csc)),
                                 interpolation=cv2.INTER_AREA)
            cname = f'ctx_{name}'
            cv2.imwrite(f'{OUT}/crops/{cname}', ctx)
            out.append(dict(work=work, pid=pid, comp=ci, region=gi, crop=name,
                            ctx=cname, ctx_scale=round(csc, 4),
                            area=int(R.sum()), adj=tot, n_lay=int(len(vals)),
                            n_lay90=nl90, purity=pur, scale=round(sc, 4),
                            placeable=int(placeable), cands=cands,
                            auto=int(len(vals) <= 1 or pur >= 0.90)))
    return out


def main():
    os.makedirs(f'{OUT}/crops', exist_ok=True)
    names = {(r['page_id'], r['layer_id']): r['name']
             for r in csv.DictReader(open(INV, encoding='utf-8-sig'))}
    zz = zmap(); fill, line = arms('gated')
    js = jobs(zz, fill, line)
    print(f'頁 {len(js)}')
    rows = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(one_page, js), 1):
            rows += r
            if i % 10 == 0:
                print(f'  {i}/{len(js)} 頁  面 {len(rows)}', flush=True)
    for r in rows:
        for c in r['cands']:
            c['name'] = names.get((r['pid'], str(c['lid'])), f'id={c["lid"]}')
            c['word'] = W_OF.get(c['name'], '')      # 空 = どの語にも対応しない層
        # 語ごとの割合(同じ語の層が複数触っていれば足す)。UI のボタンはこれを出す
        wm = {}
        for c in r['cands']:
            if c['word']:
                wm[c['word']] = wm.get(c['word'], 0.0) + c['share']
        r['word_share'] = {k: round(v, 4) for k, v in wm.items()}
        r['top_word'] = max(wm, key=wm.get) if wm else ''
    # **並びを決定的にする**: Pool の戻り順は不定なので、先に鍵で並べ替えてから混ぜる
    # (そうしないと作り直すたびに order がずれ、保存済みの答と対応しなくなる)
    rows.sort(key=lambda r: (r['work'], r['pid'], r['comp'], r['region']))
    rng = np.random.default_rng(SEED)
    ask = [r for r in rows if not r['auto']]
    auto = [r for r in rows if r['auto']]
    idx = rng.permutation(len(auto))[:int(round(AUTO_SAMPLE * len(auto)))]
    check = [auto[int(i)] for i in idx]
    for r in ask:
        r['why'] = 'ask'
    for r in check:
        r['why'] = 'check'          # **UI には出さない。見た目は同じ**
    q = ask + check
    ordr = rng.permutation(len(q))
    q = [q[int(i)] for i in ordr]
    for i, r in enumerate(q):
        r['order'] = i
    json.dump(dict(n=len(q), n_ask=len(ask), n_check=len(check),
                   n_auto=len(auto), n_all=len(rows),
                   words=[w for w, _ in WORDS], queue=q),
              open(f'{OUT}/queue.json', 'w'), ensure_ascii=False)
    print(f'-> {OUT}/queue.json  行列 {len(q)}  (要判断 {len(ask)} + '
          f'自動の確認 {len(check)} / 自動 {len(auto)}、全面 {len(rows)})')
    import collections
    for w in ('ako6', 'ako7'):
        s = [r for r in q if r['work'] == w]
        c2 = collections.Counter(min(r['n_lay90'], 4) for r in s)
        print(f'  {w}: 行列 {len(s)}  n_lay90 ごと {dict(sorted(c2.items()))}')


if __name__ == '__main__':
    main()
