# -*- coding: utf-8 -*-
"""L39 の採点(登録 `doc/l39_line_to_fill_registration_20261108.md` §3〜§6)。

腕0 / 腕1 / 腕C / 腕M / 天井 と、IC2(照合器)/ IC4(ずらしの器具)/ NC1(予測のずらし)を
**同じ面・同じ照合**で採点し、棒を**登録のとおりに**読む。**棒はここで決めない。登録の写しである。**

    python tools/evaluation/l39_score_20261108.py [run ディレクトリ ...]
    (引数なし = 器具だけ: IC2 / IC4 と自明な腕・天井)
"""
import collections
import glob
import json
import os
import pickle
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from l39_prepare_20261108 import CACHE, unpack                      # noqa: E402

FLOOR = 6000
THRS = (0.25, 0.5, 0.75)
MAIN = 0.5
BOOT, BOOT_SEED = 1000, 20261108
SIZE_BINS = ((0, .05, '<5%'), (.05, .25, '5-25%'), (.25, .75, '25-75%'), (.75, 1.01, '>=75%'))


def comps_ge(m, floor=FLOOR):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    big = [lab == i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= floor]
    small = sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < floor)
    return big, small


def match(preds, gts, thr):
    """IoU >= thr で貪欲に1対1。返り値: (当たった面の添字の集合, 当たった予測の数)。"""
    cand = []
    for i, p in enumerate(preds):
        pa = int(p.sum())
        for j, g in enumerate(gts):
            it = int((p & g).sum())
            if it:
                u = pa + int(g.sum()) - it
                if it / u >= thr:
                    cand.append((it / u, i, j))
    cand.sort(reverse=True)
    up, ug = set(), set()
    for _, i, j in cand:
        if i not in up and j not in ug:
            up.add(i); ug.add(j)
    return ug, len(up)


def shift(m):
    """NC1 / IC4: 矩形の中で幅の 1/3・高さの 1/3 だけ巡回させる。"""
    h, w = m.shape
    return np.roll(np.roll(m, h // 3, 0), w // 3, 1)


def load_test(work):
    return [d for d in (pickle.load(open(p, 'rb')) for p in sorted(glob.glob(f'{CACHE}/[0-9]*.pkl')))
            if d['work'] == work]


def prob_full(run, pid, shape):
    pr = cv2.imread(f'{run}/prob/{pid}.png', cv2.IMREAD_GRAYSCALE)
    if pr is None:
        raise FileNotFoundError(f'{run}/prob/{pid}.png')
    return cv2.resize(pr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) >= 128


def score_work(pages, run=None):
    """コマごとの記録を返す。arm -> list of dict(n, hit{thr}, npred, small, inter, union, ...)"""
    recs = collections.defaultdict(list)
    for pg in pages:
        pred_page = prob_full(run, pg['pid'], pg['shape']) if run else None
        for pan in pg['panels']:
            y0, y1, x0, x1 = pan['box']
            P = unpack(pan['P'])
            gts = [unpack(t) for _, t in pan['faces']]
            gidx = [i for i, _ in pan['faces']]
            orc = unpack(pan['oracle'])
            arms = {}
            if run is None:
                arms['腕0 塗らない'] = np.zeros_like(P)
                arms['腕1 全部塗る'] = P
                arms['腕C 写す'] = unpack(pan['copy'])
                arms['天井'] = orc
                arms['IC4 天井をずらす'] = shift(orc) & P
            else:
                pm = pred_page[y0:y1, x0:x1] & P
                arms['腕M 学習'] = pm
                arms['NC1 腕Mをずらす'] = shift(pm) & P
            for name, m in arms.items():
                big, small = comps_ge(m)
                hit = {t: match(big, gts, t) for t in THRS}
                it = int((m & orc).sum())
                recs[name].append(dict(n=len(gts), gidx=gidx, P=int(P.sum()),
                                       areas=[int(g.sum()) for g in gts],
                                       hit={t: sorted(h[0]) for t, h in hit.items()},
                                       tp={t: h[1] for t, h in hit.items()},
                                       npred=len(big), small=small, inter=it,
                                       union=int(m.sum()) + int(orc.sum()) - it))
            if run is None:
                # IC2 照合器: 面のマスクをそのまま予測に(連結成分にしない)
                h, k = match(gts, gts, MAIN)
                recs['IC2 面そのもの'].append(dict(n=len(gts), hit={MAIN: sorted(h)},
                                                 tp={MAIN: k}, npred=len(gts)))
    return recs


def R(rs, thr=MAIN):
    n = sum(r['n'] for r in rs)
    return sum(len(r['hit'][thr]) for r in rs) / n if n else float('nan')


def Pn(rs, thr=MAIN, face_only=False):
    rs = [r for r in rs if r['n'] or not face_only]
    p = sum(r['npred'] for r in rs)
    return sum(r['tp'][thr] for r in rs) / p if p else float('nan')


def main():
    runs = sys.argv[1:]
    idx = json.load(open(f'{CACHE}/_index.json'))
    print(f'L39 採点。IC1(材料の器具)= {"PASS" if idx["ic1"] else "FAIL"}')
    rebuilt = idx['rebuilt']
    out = {}
    for work in ('ako6', 'ako7'):
        pages = load_test(work)
        bp = f'{CACHE}/_base_{work}.pkl'          # 自明な腕と器具の記録は走行に依らないので使い回す
        if os.path.exists(bp):
            base = pickle.load(open(bp, 'rb'))
        else:
            base = score_work(pages)
            pickle.dump(dict(base), open(bp, 'wb'))
        print(f'\n===== 試験 {work}: 頁 {len(pages)} / コマ {len(base["腕1 全部塗る"])} '
              f'(面の在るコマ {sum(1 for r in base["腕1 全部塗る"] if r["n"])}) / '
              f'面 {sum(r["n"] for r in base["腕1 全部塗る"])} =====')
        ic2 = base['IC2 面そのもの']
        ic2_ok = abs(R(ic2) - 1) < 1e-9 and abs(Pn(ic2) - 1) < 1e-9
        print(f'IC2 照合器: R {R(ic2):.3f} / Pn {Pn(ic2):.3f} -> {"PASS" if ic2_ok else "FAIL"}')
        r1 = R(base['腕1 全部塗る'])
        ic4 = R(base['IC4 天井をずらす'])
        ic4_ok = ic4 <= 0.2 * R(base['天井'])
        print(f'IC4 ずらしの器具: 天井 {R(base["天井"]):.4f} -> ずらすと {ic4:.4f}'
              f'(登録: 天井の 0.2 倍 = {0.2 * R(base["天井"]):.4f} 以下)-> {"PASS" if ic4_ok else "FAIL"}'
              f'  [参考: 腕1 + 0.05 = {r1 + 0.05:.4f}]')
        allrec = dict(base)
        for run in runs:
            if not os.path.basename(run.rstrip('/')).startswith(f'test_{work}_'):
                continue                       # 走行は自分の試験作品だけで採点する
            allrec.update({f'{k} [{os.path.basename(run.rstrip("/"))}]': v
                           for k, v in score_work(pages, run).items()})
        print(f'\n{"腕":42s} {"R@.25":>7} {"R@.5":>7} {"R@.75":>7} {"Pn@.5":>7} {"Pn面のコマ":>9}'
              f' {"予測":>5} {"小片":>5} {"microIoU":>8}')
        for name, rs in allrec.items():
            if name.startswith('IC2'):
                continue
            I = sum(r['inter'] for r in rs); Uu = sum(r['union'] for r in rs)
            print(f'{name:42s} {R(rs, .25):7.4f} {R(rs):7.4f} {R(rs, .75):7.4f} {Pn(rs):7.4f}'
                  f' {Pn(rs, face_only=True):9.4f} {sum(r["npred"] for r in rs):5d}'
                  f' {sum(r["small"] for r in rs):5d} {I / Uu if Uu else float("nan"):8.4f}')
        triv = ['腕0 塗らない', '腕1 全部塗る', '腕C 写す']
        T = max(R(base[t]) for t in triv)
        print(f'\n自明な腕の最良 R = {T:.4f}(腕0 {R(base[triv[0]]):.4f} / 腕1 {r1:.4f} / '
              f'腕C {R(base[triv[2]]):.4f})')
        res = {}
        for run in runs:
            tag = os.path.basename(run.rstrip('/'))
            if not tag.startswith(f'test_{work}_'):
                continue
            M = allrec[f'腕M 学習 [{tag}]']
            S = allrec[f'NC1 腕Mをずらす [{tag}]']
            rm, pm, rs_ = R(M), Pn(M), R(S)
            G = rm - T
            rng = np.random.default_rng(BOOT_SEED)
            npan = len(M)
            gs = []
            for _ in range(BOOT):
                ix = rng.integers(0, npan, npan)
                n = sum(M[i]['n'] for i in ix)
                if not n:
                    continue
                rmb = sum(len(M[i]['hit'][MAIN]) for i in ix) / n
                tb = max(sum(len(base[t][i]['hit'][MAIN]) for i in ix) / n for t in triv)
                gs.append(rmb - tb)
            lo, hi = np.percentile(gs, [2.5, 97.5])
            nc1_bad = rs_ > r1 + 0.05
            ok_ic = idx['ic1'] and ic2_ok and ic4_ok
            if not ok_ic:
                v = 'C5 IC が外れた -> 何も読まない'
            elif nc1_bad:
                v = 'C4 NC1 が落ちた -> 何も読まない'
            elif rm >= .5 and pm >= .5 and G >= .2:
                v = 'C1 取り出せる'
            elif G >= .10 and lo > 0:
                v = 'C2 運ぶが届かない'
            elif hi < .10:
                v = 'C3 この定式化では運ばない'
            else:
                v = '判定保留'
            print(f'  {tag}: R {rm:.4f} / Pn {pm:.4f} / G {G:+.4f} (95% {lo:+.4f}〜{hi:+.4f}) / '
                  f'NC1 ずらし R {rs_:.4f}(棒 <= {r1 + .05:.4f})-> **{v}**')
            # 併記: 大きさ別 / kept と piece
            by = collections.defaultdict(lambda: [0, 0])
            for r in M:
                hs = set(r['hit'][MAIN])
                for j, (a, gi) in enumerate(zip(r['areas'], r['gidx'])):
                    fr = a / r['P']
                    lab = next(l for lo_, hi_, l in SIZE_BINS if lo_ <= fr < hi_)
                    org = rebuilt[gi][5]
                    for k in (lab, org):
                        by[k][0] += j in hs; by[k][1] += 1
            print('     ' + ' / '.join(f'{k} {a}/{b}' for k, (a, b) in by.items()))
            res[tag] = dict(R=rm, Pn=pm, G=G, lo=float(lo), hi=float(hi), nc1=rs_, verdict=v)
        out[work] = dict(T=T, r1=r1, rC=R(base['腕C 写す']), ceiling=R(base['天井']),
                         ic2=ic2_ok, ic4=ic4_ok, runs=res)
    if runs:
        print('\n===== C6 種の一致 =====')
        for work, o in out.items():
            vs = [r['verdict'] for r in o['runs'].values()]
            head = vs[0] if vs and len(set(vs)) == 1 and len(vs) == 3 else '判定保留(種が割れた / 3種そろわない)'
            print(f'  試験 {work}: {vs} -> 見出し **{head}**')
            o['headline'] = head
        json.dump(out, open('/home/sh1/deepl/lineart-fill-region/results/l39_score.json', 'w'),
                  ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
