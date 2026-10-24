# -*- coding: utf-8 -*-
"""L17 本体: ベタ面に隣接する線は、作者が1枚にまとめた線の束と一致するか。

登録は `doc/work_log.md § 2026-10-24 L17 事前登録` (commit 926c280)。
器具と既知セルは `l17_known_cells.py`。**既知セルが PASS していなければ回さない。**

  --arms control   対照(帰無値)だけを出す。本来の腕の値は計算しない
  --arms real      本来の腕だけ
  --arms both      両方 (既定)

一次は d=4。d ∈ {2,4,8,16} は毎回印字するが判定には使わない。
プールは ako6/ako7 を行ごとに分ける。
"""
import argparse, csv, os, statistics as st, sys
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import A
from l15_known_cells import boundary
from l17_known_cells import (DS, D1, adj_sets, assign_top, gated_layers, jobs,
                             n_lay90, panels_and_regions, purity, zmap)

INV  = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l17_fill_adjacency'
SEED = 20261026
ARMS = 'both'


def one_page(job):
    pid, work, fz, lz, sel = job
    rng = np.random.default_rng(SEED + abs(hash(pid)) % 100000)
    rows, pages = [], []
    for pid_, w_, ci, P, glab, gids, gstat, sub in panels_and_regions(job):
        if not sub:
            continue
        lab, keys = assign_top(sub, P.shape)
        ph, pw = P.shape
        big = [1 for lid, ink, z in sub if int(ink.sum()) >= 500]
        inkP = (lab > 0) & P
        cov_hit = {d: np.zeros(P.shape, bool) for d in DS}
        dom = {d: {} for d in DS}
        for gi in gids:
            R = glab == gi
            b = boundary(R)
            npx = int(R.sum()); nb = max(int(b.sum()), 1)
            rec = dict(pid=pid_, work=w_, comp=ci, region=gi, area=npx, bnd=nb,
                       panel_area=int(P.sum()), n_lay_panel=len(big), arm='real')
            if ARMS in ('both', 'real'):
                got = adj_sets(R, lab, P, ds=DS, bnd=b)
                dt = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2,
                                           cv2.DIST_MASK_PRECISE)
                for d in DS:
                    vals, cnt = got[d]
                    tot = int(cnt.sum())
                    rec[f'adj{d}'] = tot
                    if tot == 0:
                        rec[f'pur{d}'] = ''; rec[f'nl{d}'] = ''; rec[f'top{d}'] = ''
                        continue
                    k = int(vals[int(np.argmax(cnt))])
                    rec[f'pur{d}'] = round(purity(cnt), 4)
                    rec[f'nl{d}'] = n_lay90(cnt)
                    rec[f'top{d}'] = keys[k - 1]
                    dom[d].setdefault(keys[k - 1], []).append(gi)
                    cov_hit[d] |= (dt <= d)
                rows.append(rec)
            # --- 対照: 同じ面をコマ内のランダムな位置へ平行移動し、本物の線層で測る
            if ARMS not in ('both', 'control'):
                continue
            yy, xx = gstat[gi, cv2.CC_STAT_TOP], gstat[gi, cv2.CC_STAT_LEFT]
            hh, ww = gstat[gi, cv2.CC_STAT_HEIGHT], gstat[gi, cv2.CC_STAT_WIDTH]
            if ww + 2 >= pw or hh + 2 >= ph:
                continue
            patch = (glab[yy:yy + hh, xx:xx + ww] == gi); pn = int(patch.sum())
            for _ in range(24):
                nx = int(rng.integers(1, pw - ww - 1)); ny = int(rng.integers(1, ph - hh - 1))
                if abs(nx - xx) + abs(ny - yy) < 0.10 * (pw + ph):
                    continue
                Rs = np.zeros(P.shape, bool); Rs[ny:ny + hh, nx:nx + ww] = patch
                if int((Rs & P).sum()) < 0.9 * pn:
                    continue
                bs = boundary(Rs)
                r2 = dict(pid=pid_, work=w_, comp=ci, region=gi, area=npx,
                          bnd=max(int(bs.sum()), 1), panel_area=int(P.sum()),
                          n_lay_panel=len(big), arm='shuffled')
                got = adj_sets(Rs, lab, P, ds=DS, bnd=bs)
                for d in DS:
                    vals, cnt = got[d]
                    tot = int(cnt.sum()); r2[f'adj{d}'] = tot
                    if tot == 0:
                        r2[f'pur{d}'] = ''; r2[f'nl{d}'] = ''; r2[f'top{d}'] = ''
                        continue
                    r2[f'pur{d}'] = round(purity(cnt), 4)
                    r2[f'nl{d}'] = n_lay90(cnt)
                    r2[f'top{d}'] = keys[int(vals[int(np.argmax(cnt))]) - 1]
                rows.append(r2)
                break
        if ARMS in ('both', 'real'):
            rec = dict(pid=pid_, work=w_, comp=ci, n_lay_panel=len(big),
                       ink=int(inkP.sum()), n_region=len(gids))
            for d in DS:
                rec[f'cov{d}'] = int((cov_hit[d] & inkP).sum())
                sp = [len(v) for v in dom[d].values()]
                rec[f'dom_lay{d}'] = len(sp)
                rec[f'dom_split{d}'] = round(sum(sp) / len(sp), 3) if sp else ''
            pages.append(rec)
    return rows, pages


def med(v):
    return st.median(v) if v else float('nan')


def main():
    global ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='both', choices=('both', 'real', 'control'))
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    ARMS = a.arms
    os.makedirs(OUT, exist_ok=True)
    names = {(r['page_id'], r['layer_id']): r['name']
             for r in csv.DictReader(open(INV, encoding='utf-8-sig'))}
    zz = zmap(); fill, line = gated_layers()
    js = jobs(zz, fill, line)
    print(f'腕 {ARMS}  頁 {len(js)}  (既知セル KC16-19 は別プロセスで PASS 済みであること)')
    rows, pages = [], []
    with Pool(4) as pool:
        for i, (r, p) in enumerate(pool.imap_unordered(one_page, js), 1):
            rows += r; pages += p
            if i % 10 == 0:
                print(f'  {i}/{len(js)} 頁  行 {len(rows)}', flush=True)
    for r in rows:
        for d in DS:
            t = r.get(f'top{d}')
            if t not in ('', None):
                r[f'top{d}'] = names.get((r['pid'], str(t)), f'id={t}')
    tag = a.tag or ARMS
    keys = ['pid', 'work', 'comp', 'region', 'area', 'bnd', 'panel_area',
            'n_lay_panel', 'arm'] + [f'{k}{d}' for d in DS
                                     for k in ('adj', 'pur', 'nl', 'top')]
    with open(f'{OUT}/per_region_{tag}.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    pk = ['pid', 'work', 'comp', 'n_lay_panel', 'ink', 'n_region'] + \
         [f'{k}{d}' for d in DS for k in ('cov', 'dom_lay', 'dom_split')]
    if pages:
        with open(f'{OUT}/per_panel_{tag}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=pk, extrasaction='ignore')
            w.writeheader()
            for r in pages:
                w.writerow(r)
    print(f'-> {OUT}/per_region_{tag}.csv  行 {len(rows)}')

    def rep(sel, label):
        print(f'\n{label}')
        print(f'{"作品":5} {"d":>3} {"面":>5} {"purity中央":>11} {"purity画素重み":>14} '
              f'{"n_lay90中央":>12} {"n_lay90<=2":>11} {"adj中央":>9}')
        for w in ('ako6', 'ako7'):
            s = [r for r in sel if r['work'] == w]
            for d in DS:
                v = [r for r in s if r.get(f'pur{d}') not in ('', None)]
                if not v:
                    continue
                pu = [float(r[f'pur{d}']) for r in v]
                wt = sum(float(r[f'pur{d}']) * r[f'adj{d}'] for r in v) / \
                    sum(r[f'adj{d}'] for r in v)
                nl = [int(r[f'nl{d}']) for r in v]
                print(f'{w:5} {d:3d} {len(v):5d} {med(pu):11.4f} {wt:14.4f} '
                      f'{med(nl):12.1f} {sum(1 for x in nl if x<=2)/len(nl):11.4f} '
                      f'{med([r[f"adj{d}"] for r in v]):9.0f}')

    if ARMS in ('both', 'real'):
        rep([r for r in rows if r['arm'] == 'real'], '本来の腕 (面ごと)')
    if ARMS in ('both', 'control'):
        rep([r for r in rows if r['arm'] == 'shuffled'],
            '対照 = 帰無値 (面をコマ内のランダムな位置へ平行移動、本物の線層で測る)')
    if ARMS == 'both':
        print('\nP34/P35 (登録 §4。帯は基盤の確認を受けたもの)')
        for w in ('ako6', 'ako7'):
            rr = [float(r[f'pur{D1}']) for r in rows
                  if r['work'] == w and r['arm'] == 'real' and r.get(f'pur{D1}') != '']
            cc = [float(r[f'pur{D1}']) for r in rows
                  if r['work'] == w and r['arm'] == 'shuffled' and r.get(f'pur{D1}') != '']
            nl = [int(r[f'nl{D1}']) for r in rows
                  if r['work'] == w and r['arm'] == 'real' and r.get(f'nl{D1}') != '']
            print(f'  {w}: purity中央 {med(rr):.4f}  対照 {med(cc):.4f}  '
                  f'差 {med(rr)-med(cc):+.4f}  n_lay90中央 {med(nl):.1f}')
    if pages:
        print('\nP36 開示 (棒なし): 語彙の被覆 —— 線墨のうちベタ面に隣接する割合')
        for w in ('ako6', 'ako7'):
            s = [r for r in pages if r['work'] == w]
            if not s:
                continue
            tot = sum(r['ink'] for r in s)
            for d in DS:
                print(f'  {w} d={d:2d}: {sum(r[f"cov{d}"] for r in s)/max(tot,1):.4f}  '
                      f'(線墨 {tot} px, コマ {len(s)})')
        print('\nP37 開示 (棒なし): 逆向きの粒度 —— 1つの線層が何個の面に割れるか')
        print('  **粒度の正否は作者の判断なので棒を立てない (規律6)**')
        for w in ('ako6', 'ako7'):
            s = [r for r in pages if r['work'] == w and r[f'dom_split{D1}'] != '']
            if not s:
                continue
            print(f'  {w} d={D1}: 面を代表する線層 中央 {med([r[f"dom_lay{D1}"] for r in s]):.1f} 層/コマ  '
                  f'1層あたりの面 中央 {med([float(r[f"dom_split{D1}"]) for r in s]):.2f}')


if __name__ == '__main__':
    main()
