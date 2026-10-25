# -*- coding: utf-8 -*-
"""L18 本体: ベタに接した語は、墨を伝って線の束ごと動くか。

登録は `doc/work_log.md § 2026-10-25 L18 事前登録` (1fb1128)。器具と既知セルは
`l18_known_cells.py`(KC21〜KC24 が PASS していなければ回さない)。
**棒の定数は基盤が決める。** この走りが出すのは縮退・帰無値・引数だけ。

  --arms degen     縮退の点検だけ (成分の単位。棒の前に出す)
  --arms control   対照(帰無値)だけ      --arms real  本来の腕   --arms both
"""
import argparse, csv, os, statistics as st, sys
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import A
from l15_known_cells import boundary
from l17_known_cells import (D1, assign_top, gated_layers, jobs, panels_and_regions,
                             place, SEEDS, TRIES, zmap)
from l18_known_cells import (GS, comp_pur, geo_spread, ink_components, prop_full,
                             seed_of)

OUT = '/home/sh1/deepl/lineart-fill-region/results/l18_propagate'
INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
ARMS = 'both'


def n_lay90_of(C, lab):
    v = lab[C]; v = v[v > 0]
    if v.size == 0:
        return 0, 0
    vals, cnt = np.unique(v, return_counts=True)
    s = np.sort(cnt)[::-1]
    return int(np.searchsorted(np.cumsum(s), 0.9 * s.sum()) + 1), int(len(vals))


def one_page(job):
    comps, regs, pans = [], [], []
    for pid, work, ci, P, glab, gids, gstat, sub in panels_and_regions(job):
        if not sub:
            continue
        lab, keys = assign_top(sub, P.shape)
        ink, cl, st_, n = ink_components(lab, P)
        tot = int(ink.sum())
        if tot == 0:
            continue
        # ---- 縮退 (成分の単位。登録 §3)
        areas = sorted((int(st_[i, A]) for i in range(1, n)), reverse=True)
        pans.append(dict(pid=pid, work=work, comp=ci, ink=tot, n_comp=n - 1,
                         top_share=round(areas[0] / tot, 4) if areas else 0,
                         med_comp=areas[len(areas) // 2] if areas else 0))
        if ARMS == 'degen':
            for i in range(1, n):
                C = cl == i
                nl, nraw = n_lay90_of(C, lab)
                comps.append(dict(pid=pid, work=work, comp=ci, cid=i,
                                  area=int(st_[i, A]), share=round(st_[i, A] / tot, 5),
                                  n_lay90=nl, n_lay=nraw, arm='degen'))
            continue
        # ---- 腕
        for arm in (('real',) if ARMS in ('both', 'real') else ()) + \
                   (('shuffled',) if ARMS in ('both', 'control') else ()):
            seeds = SEEDS if arm == 'shuffled' else (0,)
            for sd in seeds:
                acc = np.zeros(P.shape, bool)
                seedlab = np.zeros(P.shape, np.uint16)
                owner = {}
                for gi in gids:
                    if arm == 'real':
                        R = glab == gi
                        if place(glab, gi, gstat, P, SEEDS[0], pid, ci) is None:
                            continue          # 両腕で同じ面集合 (L17 §5)
                    else:
                        R = place(glab, gi, gstat, P, sd, pid, ci)
                        if R is None:
                            continue
                    s, L, adj = seed_of(R, lab, P)
                    if adj == 0 or L == 0:
                        continue
                    C, ids = prop_full(s, cl)
                    acc |= C
                    seedlab[s & (seedlab == 0)] = gi
                    for x in ids:
                        owner.setdefault(int(x), []).append((gi, L, adj))
                    regs.append(dict(pid=pid, work=work, comp=ci, region=gi, arm=arm,
                                     seed=sd, adj=adj, prop=int(C.sum()),
                                     n_comp=int(len(ids))))
                # 成分ごとの正直さ -> cov_honest (基盤 band_form_l18 §1)
                honest = 0; conf_px = 0; lab_of = {}
                for cid, lst in owner.items():
                    C = cl == cid
                    lst.sort(key=lambda t: -t[2])
                    L = lst[0][1]
                    lab_of[cid] = L
                    ar = int(st_[cid, A])
                    cf = int(len({t[1] for t in lst}) > 1)
                    pu = comp_pur(C, lab, L)
                    nl, nraw = n_lay90_of(C, lab)
                    # **競合した成分は cov1 に入れ cov_honest には入れない**
                    if cf:
                        conf_px += ar
                    elif pu >= 0.90:
                        honest += ar
                    comps.append(dict(pid=pid, work=work, comp=ci, cid=cid, arm=arm,
                                      seed=sd, area=ar, share=round(ar / tot, 5),
                                      pur=round(pu, 4), n_lay90=nl, n_lay=nraw,
                                      conflict=cf, n_owner=len(lst)))
                rec = dict(pid=pid, work=work, comp=ci, arm=arm, seed=sd, ink=tot,
                           cov=int(acc.sum()), honest=honest, conflict_px=conf_px,
                           n_comp_owned=len(owner),
                           top_share=round(max((int(st_[i, A]) for i in range(1, n)),
                                               default=0) / tot, 4))
                # 感度: 測地距離の打ち切り (判定に使わない。単位は成分のまま)
                if seedlab.any():
                    sn = geo_spread(seedlab, ink, GS)
                    for g in GS:
                        reach = sn[g] > 0
                        rec[f'cov{g}'] = int(reach.sum())
                        h = 0
                        for cid, L in lab_of.items():
                            Cg = (cl == cid) & reach
                            k = int(Cg.sum())
                            if k == 0 or len({t[1] for t in owner[cid]}) > 1:
                                continue
                            if int((Cg & (lab == L)).sum()) / k >= 0.90:
                                h += k
                        rec[f'honest{g}'] = h
                else:
                    for g in GS:
                        rec[f'cov{g}'] = 0; rec[f'honest{g}'] = 0
                pans.append(rec)
    return comps, regs, pans


def med(v):
    return st.median(v) if v else float('nan')


def main():
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='both',
                    choices=('both', 'real', 'control', 'degen'))
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    ARMS = a.arms
    os.makedirs(OUT, exist_ok=True)
    zz = zmap(); fill, line = gated_layers()
    js = jobs(zz, fill, line)
    if a.limit:
        js = js[:a.limit]
    print(f'腕 {ARMS}  頁 {len(js)}  (既知セル KC21-24 は別プロセスで PASS 済みであること)')
    comps, regs, pans = [], [], []
    with Pool(4) as pool:
        for i, (c, r, p) in enumerate(pool.imap_unordered(one_page, js), 1):
            comps += c; regs += r; pans += p
            if i % 10 == 0:
                print(f'  {i}/{len(js)} 頁  成分 {len(comps)}', flush=True)
    for nm, rows in (('comps', comps), ('regions', regs), ('panels', pans)):
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(f'{OUT}/{nm}_{ARMS}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'-> {OUT}/{nm}_{ARMS}.csv  行 {len(rows)}')

    if ARMS == 'degen':
        print('\n' + '=' * 92)
        print('縮退の点検 (登録 §3。**棒の前に出す**。単位は成分)')
        for w in ('ako6', 'ako7'):
            pp = [r for r in pans if r['work'] == w]
            cc = [r for r in comps if r['work'] == w]
            if not cc:
                continue
            print(f'  {w}: コマ {len(pp)}  成分 {len(cc)}')
            print(f'    最大成分がコマの墨に占める割合: 中央 '
                  f'{med([r["top_share"] for r in pp]):.4f}  '
                  f'90分位 {sorted(r["top_share"] for r in pp)[int(.9*len(pp))]:.4f}  '
                  f'>0.5 のコマ {sum(1 for r in pp if r["top_share"]>0.5)/len(pp):.3f}')
            print(f'    1コマの成分の数: 中央 {med([r["n_comp"] for r in pp]):.0f}')
            ar = sorted(r['area'] for r in cc)
            print(f'    成分の画素: 中央 {med(ar):.0f}  90分位 {ar[int(.9*len(ar))]}  '
                  f'最大 {ar[-1]}')
            for lab2, lo, hi in (('全成分', 0, 10**12), ('>=1000px', 1000, 10**12),
                                 ('>=10000px', 10000, 10**12)):
                s = [r for r in cc if lo <= r['area'] < hi]
                if not s:
                    continue
                nl = [r['n_lay90'] for r in s]
                print(f'    {lab2:10} {len(s):6d}  1成分が跨ぐ層 n_lay90 中央 '
                      f'{med(nl):.1f}  <=1 {sum(1 for x in nl if x<=1)/len(nl):.3f}  '
                      f'>=3 {sum(1 for x in nl if x>=3)/len(nl):.3f}  '
                      f'生の層数 中央 {med([r["n_lay"] for r in s]):.1f}')
            # **門は画素重み中央** (基盤 band_form_l18 §4。成分を数えると巨大成分1個が映らない)
            tw = sum(r['area'] for r in cc)
            ss = sorted(cc, key=lambda r: r['n_lay90'])
            acc2 = 0; pwm = ss[-1]['n_lay90']
            for r in ss:
                acc2 += r['area']
                if acc2 >= tw / 2:
                    pwm = r['n_lay90']; break
            print(f'    **門: 成分が跨ぐ層数の画素重み中央 {pwm}** '
                  f'(>=3 なら「縮退のため測定にならない」で閉じる)  '
                  f'成分数での中央 {med([r["n_lay90"] for r in cc]):.1f}')
            for k in (1, 2, 3):
                print(f'    墨で重み付け: n_lay90 <= {k} の成分が墨に占める割合 '
                      f'{sum(r["area"] for r in cc if r["n_lay90"]<=k)/tw:.4f}')
        return

    print('\n' + '=' * 92)
    print('基盤 band_form_l18_20261025.md の引数 (一次は **画素重み**。分母は P36 と同じ)')
    res = {}
    for arm in ('real', 'shuffled'):
        pp = [r for r in pans if r.get('arm') == arm]
        if not pp:
            continue
        print(f'\n{"本来の腕" if arm=="real" else "対照 = 帰無値"} ({arm})')
        for w in ('ako6', 'ako7'):
            s0 = [r for r in pp if r['work'] == w]
            if not s0:
                continue
            per = []
            for sd in sorted({r['seed'] for r in s0}):
                t = [r for r in s0 if r['seed'] == sd]
                ink = sum(r['ink'] for r in t)
                c1 = sum(r['cov'] for r in t) / ink
                ch = sum(r['honest'] for r in t) / ink
                per.append(dict(seed=sd, cov1=c1, ch=ch,
                                rate=ch / c1 if c1 else float('nan'),
                                cf=sum(r['conflict_px'] for r in t) / ink))
                line_ = (f'  {w} seed {sd}: cov1 {c1:.4f}  **cov_honest {ch:.4f}**  '
                         f'honest_rate {per[-1]["rate"]:.4f}  競合 {per[-1]["cf"]:.4f}')
                print(line_)
            for k in ('cov1', 'ch', 'rate'):
                v = [q[k] for q in per]
                res[(arm, w, k)] = (med(v), max(v) - min(v))
            if len(per) > 1:
                print(f'    seed間: cov_honest 中央 {res[(arm,w,"ch")][0]:.4f}  '
                      f'**r0 = 最大−最小 {res[(arm,w,"ch")][1]:.4f}**  '
                      f'honest_rate 中央 {res[(arm,w,"rate")][0]:.4f}')
            # コマの2階級 (基盤 §4)
            for lab2, lo, hi in (('最大成分<0.5', 0, 0.5), ('最大成分>=0.5', 0.5, 1.01)):
                t = [r for r in s0 if lo <= r['top_share'] < hi]
                if not t:
                    continue
                ink = sum(r['ink'] for r in t)
                c1 = sum(r['cov'] for r in t) / ink
                ch = sum(r['honest'] for r in t) / ink
                print(f'    [{lab2:12}] コマ行 {len(t):5d}  墨 {ink/sum(r["ink"] for r in s0):.3f}  '
                      f'cov1 {c1:.4f}  cov_honest {ch:.4f}  '
                      f'honest_rate {ch/c1 if c1 else float("nan"):.4f}')
            # 測地距離の格子 (開示。判定に使わない)
            ink = sum(r['ink'] for r in s0)
            print('    感度 g (判定に使わない。g=∞ が一次)')
            for g in list(GS) + ['inf']:
                if g == 'inf':
                    c1 = sum(r['cov'] for r in s0) / ink
                    ch = sum(r['honest'] for r in s0) / ink
                else:
                    c1 = sum(r[f'cov{g}'] for r in s0) / ink
                    ch = sum(r[f'honest{g}'] for r in s0) / ink
                print(f'      g={str(g):>4}  cov1 {c1:.4f}  cov_honest {ch:.4f}  '
                      f'honest_rate {ch/c1 if c1 else float("nan"):.4f}')
            cc = [r for r in comps if r['work'] == w and r['arm'] == arm]
            if cc:
                pu = [r['pur'] for r in cc]
                print(f'    (併記) 語を受けた成分 {len(cc)}  '
                      f'成分数で share(comp_pur>=0.90) '
                      f'{sum(1 for x in pu if x>=0.90)/len(pu):.4f}  中央 {med(pu):.4f}')
                for lab2, lo, hi in (('<1000', 0, 1000), ('1000-9999', 1000, 10000),
                                     ('>=10000', 10000, 10**12)):
                    t = [r for r in cc if lo <= r['area'] < hi]
                    if not t:
                        continue
                    pu2 = [r['pur'] for r in t]
                    aw = sum(r['area'] for r in t)
                    print(f'      成分 {lab2:10} {len(t):6d}  share>=0.90 '
                          f'{sum(1 for x in pu2 if x>=0.90)/len(pu2):.4f}  '
                          f'画素重み {sum(r["pur"]*r["area"] for r in t)/aw:.4f}')
    if ('real', 'ako6', 'ch') in res and ('shuffled', 'ako6', 'ch') in res:
        print('\n' + '=' * 92)
        print('A / B / C (基盤 band_form_l18_20261025.md §2。式は動かさない)')
        for w in ('ako6', 'ako7'):
            if ('real', w, 'ch') not in res:
                continue
            ch1 = res[('real', w, 'ch')][0]; ch0 = res[('shuffled', w, 'ch')][0]
            r0 = res[('shuffled', w, 'ch')][1]
            rate = res[('real', w, 'rate')][0]
            print(f'  {w}: A cov_honest {ch1:.4f} >= 0.30 -> '
                  f'{"PASS" if ch1 >= 0.30 else "FAIL"}')
            print(f'      B 本来−対照 {ch1-ch0:+.4f} 対 3r0 {3*r0:.4f} -> '
                  f'{"床の上" if ch1-ch0 >= 3*r0 else "床の下"}   '
                  f'比 {ch1/ch0 if ch0 else float("nan"):.2f}x')
            print(f'      C honest_rate {rate:.4f} >= 0.50 -> '
                  f'{"PASS" if rate >= 0.50 else "FAIL"}')


if __name__ == '__main__':
    main()
