# -*- coding: utf-8 -*-
"""L15 追加診断の帰無値つき版 (棒なし)。

l15_occlusion2.py に**対照**を足したもの。対照は L15 の P27 と同じ規則:
面をコマ内のランダムな位置へ平行移動し、**本物の障壁・本物の遮蔽物**に対して
同じ器具で測る (seed 20261026、変位 >= 0.10*(W+H)、16回試行)。

遮蔽物を 32px 膨らませれば無関係な輪郭でも当たるので、帰無値なしにこの表は読めない
(規律: 棒を引く前に帰無値を出す)。per-region の距離変換 (f_fill) は前回ほぼ 0 だったので外した。
"""
import csv, collections, os, sys, statistics as st
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import comps, gated_layers, A
from l14_arms import acc_add
from l15_known_cells import boundary, SUP_PX
from l15_occlusion import excluded_layers

KOMA = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
L12  = '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions/per_panel.csv'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l15_open_runs'
SEED = 20261026
TOLS = (2, 8, 16, 32)


def measure(R, d_b, d_o, edge, arm, nm, work, gi, out):
    bn = boundary(R)
    op = bn & (d_b > SUP_PX)
    if not op.any():
        return
    n, lab = cv2.connectedComponents(op.astype(np.uint8), 8)
    d_any = np.minimum(d_o, d_b)
    for k in range(1, n):
        u = lab == k
        L = int(u.sum())
        rec = dict(name=nm, work=work, region=gi, arm=arm, length=L,
                   f_edge=round(int((u & edge).sum()) / L, 4))
        for t in TOLS:
            rec[f'f_occ{t}'] = round(int((u & (d_o <= t)).sum()) / L, 4)
            rec[f'f_any{t}'] = round(int((u & (d_any <= t)).sum()) / L, 4)
        rec['d_occ_med'] = round(float(np.median(d_o[u])), 1)
        rec['d_any_med'] = round(float(np.median(d_any[u])), 1)
        out.append(rec)


def one_page(job):
    pid, work, fl, ll, ex, panels = job
    F = B = O = None
    for lid, path in fl:
        ink = cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128
        if F is None:
            F = np.zeros(ink.shape, bool); B = np.zeros(ink.shape, bool)
            O = np.zeros(ink.shape, bool)
        acc_add(F, None, ink, 'solid')
    for lid, path in ll:
        acc_add(B, None, cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128, 'cover')
    for lid, path, name, reason in ex:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if F is None:
            F = np.zeros(im.shape, bool); B = np.zeros(im.shape, bool)
            O = np.zeros(im.shape, bool)
        acc_add(O, None, im < 128, 'cover')
    rng = np.random.default_rng(SEED + abs(hash(pid)) % 100000)
    rows = []
    for nm, x0, y0, x1, y1 in panels:
        f, b, o = F[y0:y1, x0:x1], B[y0:y1, x0:x1], O[y0:y1, x0:x1]
        H, W = f.shape
        glab, gids, gstat = comps(f, 8)
        if not gids:
            continue
        d_b = (cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
               if b.any() else np.full(f.shape, 1e9, np.float32))
        d_o = (cv2.distanceTransform((~o).astype(np.uint8), cv2.DIST_L2, 3)
               if o.any() else np.full(f.shape, 1e9, np.float32))
        edge = np.zeros(f.shape, bool)
        edge[:3, :] = edge[-3:, :] = edge[:, :3] = edge[:, -3:] = True
        for gi in gids:
            measure(glab == gi, d_b, d_o, edge, 'real', nm, work, gi, rows)
            # 対照: L15 の P27 と同じ規則
            yy, xx = gstat[gi, cv2.CC_STAT_TOP], gstat[gi, cv2.CC_STAT_LEFT]
            hh, ww = gstat[gi, cv2.CC_STAT_HEIGHT], gstat[gi, cv2.CC_STAT_WIDTH]
            if ww + 2 >= W or hh + 2 >= H:
                continue
            patch = (glab[yy:yy + hh, xx:xx + ww] == gi)
            for _ in range(16):
                nx = int(rng.integers(1, W - ww - 1)); ny = int(rng.integers(1, H - hh - 1))
                if abs(nx - xx) + abs(ny - yy) < 0.10 * (W + H):
                    continue
                Rs = np.zeros(f.shape, bool)
                Rs[ny:ny + hh, nx:nx + ww] = patch
                measure(Rs, d_b, d_o, edge, 'shuffled', nm, work, gi, rows)
                break
    return rows


def main():
    fill, line = gated_layers()
    ex, why = excluded_layers()
    print('障壁に入れなかった層 (= 遮蔽物の候補)')
    for k, v in why.most_common():
        print(f'  {v:5d}  {k}')
    man = {r['name']: r for r in csv.DictReader(open(f'{KOMA}/manifest.csv'))}
    panels = collections.defaultdict(list)
    for r in csv.DictReader(open(L12)):
        m = man[r['name']]
        panels[m['pid'].replace('__', '_')].append(
            (r['name'], int(m['x0']), int(m['y0']), int(m['x1']), int(m['y1'])))
    jobs = [(pid, w, sorted(fl), sorted(line.get((pid, w), [])), ex.get((pid, w), []),
             panels[pid]) for (pid, w), fl in sorted(fill.items()) if pid in panels]
    rows = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(one_page, jobs), 1):
            rows += r
            if i % 15 == 0:
                print(f'  {i}/{len(jobs)} 頁  区間 {len(rows)}', flush=True)
    with open(f'{OUT}/occlusion3.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'-> {OUT}/occlusion3.csv  {len(rows)} 区間\n' + '=' * 96)
    print('遮蔽物の説明力と、その帰無値 (開いた輪郭の画素で重み付け。棒なし)')
    print(f'{"作品":5} {"長さ帯":9} {"腕":6} {"開px":>10} | '
          f'{"遮蔽物 2/8/16/32px":22} | {"何でも 2/8/16/32px":22} | 何かまで中央')
    for w in ('ako6', 'ako7'):
        for lab, lo, hi in (('<50px', 0, 50), ('50-200px', 50, 200),
                            ('>=200px', 200, 10 ** 9), ('全部', 0, 10 ** 9)):
            vals = {}
            for arm in ('real', 'shuffled'):
                s = [r for r in rows if r['work'] == w and r['arm'] == arm
                     and lo <= r['length'] < hi]
                if not s:
                    continue
                tot = sum(r['length'] for r in s)
                def P(k):
                    return sum(r['length'] * r[k] for r in s) / tot
                vals[arm] = {k: P(k) for k in
                             [f'f_occ{t}' for t in TOLS] + [f'f_any{t}' for t in TOLS]}
                oc = ' '.join(f'{P(f"f_occ{t}"):.3f}' for t in TOLS)
                an = ' '.join(f'{P(f"f_any{t}"):.3f}' for t in TOLS)
                print(f'{w:5} {lab:9} {("本来" if arm=="real" else "対照"):6} '
                      f'{tot:10,.0f} | {oc:22} | {an:22} | '
                      f'{st.median([r["d_any_med"] for r in s]):6.1f}')
            if len(vals) == 2:
                rt = ' '.join(f'{vals["real"][f"f_occ{t}"]/max(vals["shuffled"][f"f_occ{t}"],1e-9):.2f}'
                              for t in TOLS)
                ra = ' '.join(f'{vals["real"][f"f_any{t}"]/max(vals["shuffled"][f"f_any{t}"],1e-9):.2f}'
                              for t in TOLS)
                print(f'{"":5} {"":9} {"比":6} {"":>10} | {rt:22} | {ra:22} |')


if __name__ == '__main__':
    main()
