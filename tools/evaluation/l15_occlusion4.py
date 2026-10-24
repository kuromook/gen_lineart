# -*- coding: utf-8 -*-
"""L15 追加診断の帰無値つき版 (棒なし)。遮蔽物 + コマ枠 + 線、全部に対照をつける。

作者の指摘2つを同じ1パスで測る:
  1 「手前に吹き出しや3Dオブジェクトが来る所は、隠れるので描いていない」 -> occ
  2 「線画レイヤはコマレイヤと分離しているなら、コマで区切ってる部分は途切れている。
      これはフキダシ以上に線の隙間をつくる」                              -> edge

対照は L15 の P27 と同じ規則 (seed 20261026、変位 >= 0.10*(W+H)、16回試行)。
遮蔽物を 32px 膨らませれば無関係な輪郭でも当たるので、帰無値なしにこの表は読めない。
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
ARMS = ('occ', 'edge', 'any', 'all')


def measure(R, d_b, d_o, d_e, arm, nm, work, gi, out):
    bn = boundary(R)
    op = bn & (d_b > SUP_PX)
    if not op.any():
        return
    n, lab = cv2.connectedComponents(op.astype(np.uint8), 8)
    d_any = np.minimum(d_o, d_b)          # 線でも遮蔽物でもよい
    d_all = np.minimum(d_any, d_e)        # コマ枠も入れる
    for k in range(1, n):
        u = lab == k
        L = int(u.sum())
        rec = dict(name=nm, work=work, region=gi, arm=arm, length=L)
        for t in TOLS:
            rec[f'f_occ{t}'] = round(int((u & (d_o <= t)).sum()) / L, 4)
            rec[f'f_edge{t}'] = round(int((u & (d_e <= t)).sum()) / L, 4)
            rec[f'f_any{t}'] = round(int((u & (d_any <= t)).sum()) / L, 4)
            rec[f'f_all{t}'] = round(int((u & (d_all <= t)).sum()) / L, 4)
        rec['d_occ_med'] = round(float(np.median(d_o[u])), 1)
        rec['d_edge_med'] = round(float(np.median(d_e[u])), 1)
        rec['d_any_med'] = round(float(np.median(d_any[u])), 1)
        rec['d_all_med'] = round(float(np.median(d_all[u])), 1)
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
        yv = np.arange(H, dtype=np.float32)[:, None]
        xv = np.arange(W, dtype=np.float32)[None, :]
        d_e = np.minimum(np.minimum(yv, H - 1 - yv),
                         np.minimum(xv, W - 1 - xv)) + np.zeros(f.shape, np.float32)
        for gi in gids:
            measure(glab == gi, d_b, d_o, d_e, 'real', nm, work, gi, rows)
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
                measure(Rs, d_b, d_o, d_e, 'shuffled', nm, work, gi, rows)
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
    with open(f'{OUT}/occlusion4.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'-> {OUT}/occlusion4.csv  {len(rows)} 区間\n' + '=' * 110)
    print('遮蔽物・コマ枠の説明力と、その帰無値 (開いた輪郭の画素で重み付け。棒なし)')
    print(f'{"作品":4} {"長さ帯":9} {"腕":5} {"開px":>10} | {"遮蔽物 2/8/16/32":19} | '
          f'{"コマ枠 2/8/16/32":19} | {"線+遮 2/8/16/32":19} | {"全部 2/8/16/32":19} | 全部まで中央')
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

                def P(k, s=s, tot=tot):
                    return sum(r['length'] * r[k] for r in s) / tot
                vals[arm] = {f'f_{a}{t}': P(f'f_{a}{t}') for a in ARMS for t in TOLS}
                cols = [' '.join(f'{P(f"f_{a}{t}"):.3f}' for t in TOLS) for a in ARMS]
                print(f'{w:4} {lab:9} {("本来" if arm == "real" else "対照"):5} '
                      f'{tot:10,.0f} | ' + ' | '.join(f'{c:19}' for c in cols) +
                      f' | {st.median([r["d_all_med"] for r in s]):6.1f}')
            if len(vals) == 2:
                cols = [' '.join(
                    f'{vals["real"][f"f_{a}{t}"] / max(vals["shuffled"][f"f_{a}{t}"], 1e-9):.2f}'
                    for t in TOLS) for a in ARMS]
                print(f'{"":4} {"":9} {"比":5} {"":>10} | ' +
                      ' | '.join(f'{c:19}' for c in cols) + ' |')


if __name__ == '__main__':
    main()
