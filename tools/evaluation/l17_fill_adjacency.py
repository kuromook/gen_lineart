# -*- coding: utf-8 -*-
"""L17 本体: ベタ面に隣接する線は、作者が1枚にまとめた線の束と一致するか。

登録は `doc/work_log.md § 2026-10-24 L17 事前登録` (926c280)。器具と既知セルは
`l17_known_cells.py`(KC16〜KC20 が PASS していなければ回さない)。
帯は基盤の `inbox/band_check_l17_20261025.md` §6 に**関数として**書かれている。
この走りが報告するのは**その引数だけ**。

基盤の指摘で直した点:
  §5 対照の面集合が腕と違っていた(面 17/19 枚少なく、落ちていたのは大きい面)。
     **置けない面は両方の腕から外す**。乱数は面ごとに種を切るので、どの腕を走らせても
     同じ面集合になる。落とした面は面積つきで印字(規律9-1)
  §3 対照は **seed 5本**。報告は seed 間の中央と 最大−最小
  §6 一次は「中央値」ではなく **share(purity >= 0.90)**。`adj` に 2枚以上触る面に限る

  --arms control   対照(帰無値)だけ。本来の腕の purity は計算しない
  --arms real      本来の腕だけ        --arms both  両方
"""
import argparse, csv, os, statistics as st, sys
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l15_known_cells import boundary
from l17_known_cells import (DS, D1, adj_sets, assign_top, gated_layers, jobs,
                             n_lay90, panels_and_regions, place, purity, zmap,
                             SEEDS, TRIES)

INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
OUT = '/home/sh1/deepl/lineart-fill-region/results/l17_fill_adjacency'
ARMS = 'both'
# 基盤 `inbox/band_confirm_l17_20261025.md` §1 で確定した棒。式は §6 のまま
# **撤回**(2026-10-25): 0.5672 / 0.5779 は**前の走行の対照**から固定した値で、
# `place()` の種が過程依存だったため**再現しない引き**の上に乗っていた(罠 T97)。
# **式は基盤 `band_check_l17_20261025.md` §6 のまま。値は同じ走行の対照から計算する。**
WITHDRAWN_BAR_Q = {'ako6': 0.5672, 'ako7': 0.5779}
BAR_K = 2


CTL = {}       # 同じ走行の対照。棒はここから計算する


def bar_q(q0):
    """P34'(b) の棒。**帰無値の関数**(基盤 §6)。

    `q1 >= 0.50` かつ `q1 >= q0 + 0.5 * (1 - q0)` の厳しい方。
    **腕と同じ走行の対照から計算する** —— 走行をまたいで凍結すると、
    **凍結した引きの運が棒に入る**(2026-10-25 の新しい規律)。
    """
    return max(0.50, q0 + 0.5 * (1.0 - q0))
TSTEP = [round(x, 2) for x in np.arange(0.0, 1.001, 0.1)]


def measure(R, lab, P, bnd, rec, keys):
    got = adj_sets(R, lab, P, ds=DS, bnd=bnd)
    for d in DS:
        vals, cnt = got[d]
        tot = int(cnt.sum())
        rec[f'adj{d}'] = tot; rec[f'lay{d}'] = int(len(vals))
        if tot == 0:
            rec[f'pur{d}'] = ''; rec[f'nl{d}'] = ''; rec[f'top{d}'] = ''
            continue
        rec[f'pur{d}'] = round(purity(cnt), 4)
        rec[f'nl{d}'] = n_lay90(cnt)
        # **`vals` はラベル番号(z で並べた通し番号)なので、層 id に直してから入れる。**
        # ここを `int(vals[...])` のまま CSV に書いて名前を引くと別の層の名前が出る(2026-10-25 の修正)
        rec[f'top{d}'] = keys[int(vals[int(np.argmax(cnt))]) - 1]
    return got


def one_page(job):
    rows, pages, sel = [], [], []
    for pid, work, ci, P, glab, gids, gstat, sub in panels_and_regions(job):
        if not sub:
            continue
        lab, keys = assign_top(sub, P.shape)
        ph, pw = P.shape
        big = sum(1 for lid, ink, z in sub if int(ink.sum()) >= 500)
        inkP = (lab > 0) & P
        cov_hit = {d: np.zeros(P.shape, bool) for d in DS}
        dom = {d: {} for d in DS}
        for gi in gids:
            R = glab == gi
            b = boundary(R)
            npx = int(R.sum()); nb = max(int(b.sum()), 1); pa = int(P.sum())
            base = dict(pid=pid, work=work, comp=ci, region=gi, area=npx, bnd=nb,
                        panel_area=pa, frac=round(npx / max(pa, 1), 4),
                        n_lay_panel=big)
            # --- 先に置けるかを決める。腕に依らない(面ごとに種を切る)
            pos = {}
            for sd in SEEDS:
                pos[sd] = place(glab, gi, gstat, P, sd, pid, ci, tries=TRIES)
            placed = sum(1 for v in pos.values() if v is not None)
            ok = (placed == len(SEEDS))
            base['placeable'] = int(ok)
            sel.append(dict(base, placed=placed))
            # 帯の母集団は placeable=1 の面。**本来の腕は置けない面も測って印字する**
            # (基盤 §4-3 の開示。対照が無いので棒は立たない)
            if ARMS in ('both', 'real'):
                rec = dict(base, arm='real', seed=0)
                got = measure(R, lab, P, b, rec, keys)
                dt = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2,
                                           cv2.DIST_MASK_PRECISE)
                for d in DS:
                    if rec[f'pur{d}'] != '':
                        dom[d].setdefault(rec[f'top{d}'], []).append(gi)
                        cov_hit[d] |= (dt <= d)
                rows.append(rec)
            if ok and ARMS in ('both', 'control'):
                for sd in SEEDS:
                    Rs = pos[sd]
                    bs = boundary(Rs)
                    rec = dict(base, arm='shuffled', seed=sd,
                               bnd=max(int(bs.sum()), 1))
                    measure(Rs, lab, P, bs, rec, keys)
                    rows.append(rec)
        if ARMS in ('both', 'real'):
            rec = dict(pid=pid, work=work, comp=ci, n_lay_panel=big,
                       ink=int(inkP.sum()), n_region=len(gids))
            for d in DS:
                rec[f'cov{d}'] = int((cov_hit[d] & inkP).sum())
                sp = [len(v) for v in dom[d].values()]
                rec[f'dom_lay{d}'] = len(sp)
                rec[f'dom_split{d}'] = round(sum(sp) / len(sp), 3) if sp else ''
            pages.append(rec)
    return rows, pages, sel


def med(v):
    return st.median(v) if v else float('nan')


def seed_table(sub, label):
    """seed ごとの表と seed 間の中央・振れ。本来の腕は seed 0 の1行だけ。"""
    if not sub:
        return None
    seeds = sorted({r['seed'] for r in sub})
    per = []
    for sd in seeds:
        t = [r for r in sub if r['seed'] == sd]
        pu = [float(r[f'pur{D1}']) for r in t]
        nl = [int(r[f'nl{D1}']) for r in t]
        wt = sum(float(r[f'pur{D1}']) * r[f'adj{D1}'] for r in t) / \
            sum(r[f'adj{D1}'] for r in t)
        per.append(dict(seed=sd, n=len(t),
                        q90=sum(1 for x in pu if x >= 0.90) / len(pu),
                        q98=sum(1 for x in pu if x >= 0.98) / len(pu),
                        mpur=med(pu), wpur=wt, mnl=med(nl)))
    print(f'    {label}')
    print(f'    {"seed":>9} {"面":>6} {"share>=0.90":>12} {"share>=0.98":>12} '
          f'{"median":>8} {"画素重み":>9} {"median n_lay90":>15}')
    for q in per:
        print(f'    {q["seed"]:>9} {q["n"]:6d} {q["q90"]:12.4f} {q["q98"]:12.4f} '
              f'{q["mpur"]:8.4f} {q["wpur"]:9.4f} {q["mnl"]:15.1f}')
    out = {}
    for k in ('q90', 'q98', 'mpur', 'wpur', 'mnl'):
        v = [q[k] for q in per]
        out[k] = med(v); out[k + '_rng'] = max(v) - min(v)
    if len(seeds) > 1:
        print('    ' + '  '.join(f'{k} 中央 {out[k]:.4f} 振れ {out[k+"_rng"]:.4f}'
                                 for k in ('q90', 'mpur', 'mnl')))
    return out


def dist(sub, label):
    pu = [float(r[f'pur{D1}']) for r in sub]
    if not pu:
        return
    print(f'    {label} の累積 (0.1 刻み)')
    print('      ' + ' '.join(f'>={t:.1f}:{sum(1 for x in pu if x>=t)/len(pu):.3f}'
                              for t in TSTEP[1:]))


def strata(sub):
    print('    |adj| の層別')
    for lab2, lo, hi in (('<200', 0, 200), ('200-999', 200, 1000),
                         ('>=1000', 1000, 10 ** 12)):
        t = [r for r in sub if lo <= r[f'adj{D1}'] < hi]
        if not t:
            continue
        pu2 = [float(r[f'pur{D1}']) for r in t]
        print(f'      {lab2:8} 面 {len(t):5d}  share>=0.90 '
              f'{sum(1 for x in pu2 if x>=0.90)/len(pu2):.4f}  中央 {med(pu2):.4f}  '
              f'median n_lay90 {med([int(r[f"nl{D1}"]) for r in t]):.1f}')
    print('    感度 d (判定に使わない)')
    for d in DS:
        t = [r for r in sub if r[f'pur{d}'] != '' and r[f'lay{d}'] >= 2]
        if not t:
            continue
        pu2 = [float(r[f'pur{d}']) for r in t]
        print(f'      d={d:2d}  面 {len(t):5d}  share>=0.90 '
              f'{sum(1 for x in pu2 if x>=0.90)/len(pu2):.4f}  中央 {med(pu2):.4f}  '
              f'|adj| 中央 {med([r[f"adj{d}"] for r in t]):.0f}')


def report(rows, arm, pool):
    """基盤 §6 の引数 + §4 の開示4点。帯の母集団は placeable=1 かつ adj に2枚以上。"""
    allr = [r for r in rows if r['arm'] == arm and r['work'] == pool
            and r['placeable'] == 1]
    if not allr:
        return None
    s = [r for r in allr if r[f'pur{D1}'] != '']
    zero = len(allr) - len(s)
    one = [r for r in s if r[f'lay{D1}'] <= 1]
    s2 = [r for r in s if r[f'lay{D1}'] >= 2]
    print(f'  {pool} {arm}: 行 {len(allr)}  うち adj が空(その墨に触らない) {zero} '
          f'({zero/len(allr):.4f})  -> 測った面 {len(s)}')
    print(f'    adj に層が1枚 {len(one)} ({len(one)/len(s):.4f})  '
          f'-> 2枚以上 {len(s2)} (帯の母集団)')
    out = seed_table(s2, '【帯の母集団】adj に 2枚以上の層が触る面') if s2 else None
    if s2:
        dist(s2, 'purity')
        strata(s2)
    # 開示2: 全面(絞りなし)
    print('    --- 開示2 (棒なし): 全面(絞りなし。単層の面を含む) ---')
    seed_table(s, '絞りなし')
    # 開示1: 単層の面
    if one:
        ad = sorted(r[f'adj{D1}'] for r in one)
        print(f'    --- 開示1 (棒なし): 層が1枚の面 {len(one)} ({len(one)/len(s):.4f}) '
              f'|adj| 中央 {med(ad):.0f} 最大 {ad[-1]} ---')
        print('      purity は定義上 1.0、n_lay90 は 1 なので分布は出さない。'
              f'面積 中央 {med(sorted(r["area"] for r in one)):.0f}px')
    return out


def report_unplaced(rows, pool):
    """開示3 (棒なし): 対照が置けなかった大きい面を、本来の腕でだけ測った値。"""
    s = [r for r in rows if r['arm'] == 'real' and r['work'] == pool
         and r['placeable'] == 0 and r[f'pur{D1}'] != '']
    if not s:
        return
    s2 = [r for r in s if r[f'lay{D1}'] >= 2]
    ar = sorted(r['area'] for r in s)
    print(f'  {pool}: 面 {len(s)}  面積 中央 {med(ar):.0f}px 最大 {ar[-1]}px  '
          f'面/コマ 中央 {med(sorted(r["frac"] for r in s)):.4f}')
    print(f'    adj に層が1枚 {len(s)-len(s2)}  -> 2枚以上 {len(s2)}')
    if s2:
        pu = [float(r[f'pur{D1}']) for r in s2]
        print(f'    share>=0.90 {sum(1 for x in pu if x>=0.90)/len(pu):.4f}  '
              f'中央 {med(pu):.4f}  median n_lay90 '
              f'{med([int(r[f"nl{D1}"]) for r in s2]):.1f}  '
              f'|adj| 中央 {med([r[f"adj{D1}"] for r in s2]):.0f}')
    print('    **対照が無いので棒は立たない。落としたものを黙って落とさないための数**')


def main():
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='both', choices=('both', 'real', 'control'))
    ap.add_argument('--tag', default='')
    ap.add_argument('--limit', type=int, default=0, help='smoke test 用に頁を絞る')
    a = ap.parse_args()
    ARMS = a.arms
    os.makedirs(OUT, exist_ok=True)
    names = {(r['page_id'], r['layer_id']): r['name']
             for r in csv.DictReader(open(INV, encoding='utf-8-sig'))}
    zz = zmap(); fill, line = gated_layers()
    js = jobs(zz, fill, line)
    if a.limit:
        js = js[:a.limit]
    print(f'腕 {ARMS}  頁 {len(js)}  seed {list(SEEDS)}  '
          f'(既知セル KC16-20 は別プロセスで PASS 済みであること)')
    rows, pages, sel = [], [], []
    with Pool(4) as pool:
        for i, (r, p, s) in enumerate(pool.imap_unordered(one_page, js), 1):
            rows += r; pages += p; sel += s
            if i % 10 == 0:
                print(f'  {i}/{len(js)} 頁  行 {len(rows)}', flush=True)
    tag = a.tag or ARMS
    keys = ['pid', 'work', 'comp', 'region', 'area', 'bnd', 'panel_area', 'frac',
            'n_lay_panel', 'placeable', 'arm', 'seed'] + \
           [f'{k}{d}' for d in DS for k in ('adj', 'lay', 'pur', 'nl', 'top')]
    for r in rows:
        for d in DS:
            t = r.get(f'top{d}')
            if t not in ('', None):
                r[f'top{d}'] = names.get((r['pid'], str(t)), f'id={t}')
    with open(f'{OUT}/per_region_{tag}.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(f'{OUT}/selection_{tag}.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['pid', 'work', 'comp', 'region', 'area',
                                           'panel_area', 'frac', 'n_lay_panel',
                                           'placed', 'placeable'],
                           extrasaction='ignore')
        w.writeheader()
        for r in sel:
            w.writerow(r)
    if pages:
        pk = ['pid', 'work', 'comp', 'n_lay_panel', 'ink', 'n_region'] + \
             [f'{k}{d}' for d in DS for k in ('cov', 'dom_lay', 'dom_split')]
        with open(f'{OUT}/per_panel_{tag}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=pk, extrasaction='ignore')
            w.writeheader()
            for r in pages:
                w.writerow(r)
    print(f'-> {OUT}/per_region_{tag}.csv  行 {len(rows)}')
    print(f'-> {OUT}/selection_{tag}.csv  面 {len(sel)}')

    print('\n' + '=' * 92)
    print('選別 (規律9-1。両方の腕から同じ面を外している。基盤 §5 の指摘)')
    for w in ('ako6', 'ako7'):
        s = [r for r in sel if r['work'] == w]
        if not s:
            continue
        drop = [r for r in s if not r['placeable']]
        print(f'  {w}: 列挙した面 {len(s)}  -> 測る面 {len(s)-len(drop)}  '
              f'外した面 {len(drop)} ({len(drop)/len(s):.4f})')
        if drop:
            ar = sorted(r['area'] for r in drop)
            fr = sorted(r['frac'] for r in drop)
            print(f'    外した面の面積 中央 {med(ar):.0f}px 最大 {ar[-1]}px  '
                  f'面/コマ 中央 {med(fr):.4f} 最大 {fr[-1]:.4f}')
            print(f'    (内訳) 5 seed 全部で置けた {sum(1 for r in s if r["placed"]==5)}  '
                  f'一部だけ {sum(1 for r in s if 0 < r["placed"] < 5)}  '
                  f'1本も置けなかった {sum(1 for r in s if r["placed"]==0)}')
        kept = [r['frac'] for r in s if r['placeable']]
        print(f'    測る面の 面/コマ: 中央 {med(kept):.4f}  >0.2 '
              f'{sum(1 for x in kept if x>0.2)}  >0.5 {sum(1 for x in kept if x>0.5)}  '
              f'最大 {max(kept):.4f}')

    print('\n' + '=' * 92)
    if ARMS in ('both', 'control'):
        print('対照 = 帰無値 (面をコマ内のランダムな位置へ平行移動、本物の線層で測る)')
        print('**基盤 §6 の引数。q0 / r0 / m0 / k0 はここから読む**')
        for w in ('ako6', 'ako7'):
            CTL[w] = report(rows, 'shuffled', w)
    if ARMS in ('both', 'real'):
        print('\n本来の腕')
        got = {}
        for w in ('ako6', 'ako7'):
            got[w] = report(rows, 'real', w)
        print('\n開示3 (棒なし): 対照が置けなかった大きい面 (本来の腕だけで測った)')
        for w in ('ako6', 'ako7'):
            report_unplaced(rows, w)
        print('\n' + '=' * 92)
        print('P34\' / P35\' (式は基盤 band_check_l17 §6。'
              '**値は同じ走行の対照から計算する**)')
        print(f'  **撤回済みの棒(前の走行の引きの上): {WITHDRAWN_BAR_Q}**')
        v34, v35 = [], []
        ctl = CTL
        for w in ('ako6', 'ako7'):
            if not got.get(w):
                continue
            q1 = got[w]['q90']; k1 = got[w]['mnl']
            q0 = ctl[w]['q90'] if ctl.get(w) else float('nan')
            r0 = ctl[w].get('q90_rng', float('nan')) if ctl.get(w) else float('nan')
            bq = bar_q(q0)
            print(f'  {w}: 対照 q0 {q0:.4f}  ->  **棒 = max(0.50, q0+0.5(1-q0)) '
                  f'= {bq:.4f}**  (r0 {r0:.4f})')
            if q1 - q0 < 3 * r0:
                print(f'       **(a) 床: q1-q0 {q1-q0:+.4f} < 3r0 {3*r0:.4f} '
                      f'-> PASS でも FAIL でもなく「読めない」**')
            a = q1 >= bq; b = k1 <= BAR_K
            v34.append(a); v35.append(b)
            print(f'  {w}: q1 {q1:.4f} 対 棒 {bq:.4f} -> '
                  f'{"PASS" if a else "FAIL"}   k1 {k1:.1f} 対 {BAR_K} -> '
                  f'{"PASS" if b else "FAIL"}')
        print(f'  **P34\' {"PASS" if v34 and all(v34) else "FAIL"}'
              f'(両プールで成立すること) / P35\' '
              f'{"PASS" if v35 and all(v35) else "FAIL"}**')
    if pages:
        print('\nP36 開示 (棒なし): 語彙の被覆 —— **門が通した層の墨**のうちベタ面に隣接する割合')
        for w in ('ako6', 'ako7'):
            s = [r for r in pages if r['work'] == w]
            if not s:
                continue
            tot = sum(r['ink'] for r in s)
            for d in DS:
                print(f'  {w} d={d:2d}: {sum(r[f"cov{d}"] for r in s)/max(tot,1):.4f}  '
                      f'(門が通した層の墨 {tot} px, コマ {len(s)})')
        print('\nP37 開示 (棒なし): 逆向きの粒度 —— 1つの線層が何個の面に割れるか')
        print('  **粒度の正否は作者の判断なので棒を立てない (規律6)**')
        for w in ('ako6', 'ako7'):
            s = [r for r in pages if r['work'] == w and r[f'dom_split{D1}'] != '']
            if not s:
                continue
            print(f'  {w} d={D1}: 面を代表する線層 中央 '
                  f'{med([r[f"dom_lay{D1}"] for r in s]):.1f} 層/コマ  '
                  f'1層あたりの面 中央 {med([float(r[f"dom_split{D1}"]) for r in s]):.2f}')


if __name__ == '__main__':
    main()
