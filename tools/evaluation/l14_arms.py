# -*- coding: utf-8 -*-
"""L14 本体: 作者のベタ作成工程(塗りつぶし -> 隙間埋め -> 1px太らせ)を、作者の
ベタレイヤの面で試す。登録は doc/work_log.md「L14 事前登録」(commit 6e2ba9e)、
追補 A5-A7 は同節 §8 (d05079e)。既知セルが PASS しなければ起動しない。

  FILL_SOLID := ∪_{keep4>=0.95} fill_mask_4(cover)     作者のベタ面 = 母集団
  GAPFILL    := ∪_{keep4>=0.95} cover \ fill_mask_4    工程2の印 + 工程3の縁
  BARRIER    := ∪_{keep4<0.95}  cover                  線画側

  腕1 = 工程1 (障壁 BARRIER) / 腕2 = 工程1+2 (BARRIER ∪ GAPFILL)
  腕3 = 腕2 の照合面を1px膨張 / 対照 = 同じ頁の別の位置から障壁を取る
"""
import csv, os, subprocess, sys, collections, statistics as st
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import fill_mask, ink_of
sys.path.insert(0, HERE)
from l14_known_cells import (comps, match_lab, gated_layers, MINREG, A, PAD,
                             KEEP4_FILL)

KOMA = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
L12  = '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions/per_panel.csv'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l14_arms'
SEED = 20261025
N_GT_L12 = 7400          # KC10: 母集団を狭めたのだからこれより小さいはず
K3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)   # 1px の膨張(4近傍)


def is_lettering(anc, name):
    """A5: `描き文字` を階層のどこかに持つ層 (A2 は第1階層の /text しか落とさなかった)。"""
    return '描き文字' in anc or '描き文字' in name


def iou_of(a, b):
    i = int((a & b).sum())
    u = int(a.sum()) + int(b.sum()) - i
    return i / u if u else 0.0


def acc_add(acc, shape, ink, which):
    """層の被覆を頁の累積に足す。which='solid'/'thin'/'cover'。"""
    if not ink.any():
        return
    ys, xs = np.nonzero(ink)
    y0, y1 = max(0, ys.min() - PAD), min(ink.shape[0], ys.max() + 1 + PAD)
    x0, x1 = max(0, xs.min() - PAD), min(ink.shape[1], xs.max() + 1 + PAD)
    sub = ink[y0:y1, x0:x1]
    if which == 'cover':
        acc[y0:y1, x0:x1] |= sub
        return
    fm = fill_mask(sub, radius=4)
    acc[y0:y1, x0:x1] |= fm if which == 'solid' else (sub & ~fm)


def run_arms(barrier, glab, gids, gstat, want_dil=False):
    plab, pids, pstat = comps(~barrier, 4)
    m = match_lab(plab, pids, pstat, glab, gids, gstat)
    dil = {}
    if want_dil:
        for gi, (pi, iou) in m.items():
            y, x = gstat[gi, cv2.CC_STAT_TOP], gstat[gi, cv2.CC_STAT_LEFT]
            h, w = gstat[gi, cv2.CC_STAT_HEIGHT], gstat[gi, cv2.CC_STAT_WIDTH]
            py, px = pstat[pi, cv2.CC_STAT_TOP], pstat[pi, cv2.CC_STAT_LEFT]
            ph, pw = pstat[pi, cv2.CC_STAT_HEIGHT], pstat[pi, cv2.CC_STAT_WIDTH]
            y0, x0 = max(0, min(y, py) - 3), max(0, min(x, px) - 3)
            y1 = min(glab.shape[0], max(y + h, py + ph) + 3)
            x1 = min(glab.shape[1], max(x + w, px + pw) + 3)
            G = glab[y0:y1, x0:x1] == gi
            P = plab[y0:y1, x0:x1] == pi
            Pd = cv2.dilate(P.astype(np.uint8), K3, iterations=1).astype(bool)
            dil[gi] = (iou, iou_of(Pd, G))
    return len(pids), m, dil


def one_page(job):
    pid, work, fl, ll, panels = job
    shape = None
    first = cv2.imread(fl[0][1] if fl else ll[0][1], cv2.IMREAD_GRAYSCALE)
    if first is None:
        return dict(error=f'read {pid}')
    shape = first.shape
    Fc = np.zeros(shape, bool); Fl = np.zeros(shape, bool)
    Gc = np.zeros(shape, bool); Gl = np.zeros(shape, bool)
    B  = np.zeros(shape, bool)
    # A6: コマごとのラスタ(膨張済み)を先に読む
    ras = {}
    for nm, x0, y0, x1, y1 in panels:
        g = cv2.imread(f'{KOMA}/line/{nm}', cv2.IMREAD_GRAYSCALE)
        ras[nm] = (cv2.dilate(ink_of(g).astype(np.uint8), np.ones((5, 5), np.uint8))
                   .astype(bool)) if g is not None else None
    lay_rows = []
    for lid, path, name, anc, kp, isfill in fl + ll:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return dict(error=f'read {pid} L{lid}')
        ink = im < 128
        if isfill:
            let = is_lettering(anc, name)
            acc_add(Fl if let else Fc, shape, ink, 'solid')
            acc_add(Gl if let else Gc, shape, ink, 'thin')
        else:
            acc_add(B, shape, ink, 'cover')
        for nm, x0, y0, x1, y1 in panels:
            sub = ink[y0:y1, x0:x1]
            n = int(sub.sum())
            if n < MINREG or ras[nm] is None:
                continue
            lay_rows.append(dict(name=nm, layer_id=lid, layer_name=name, ancestors=anc,
                                 keep4=round(kp, 4), is_fill=int(isfill),
                                 px_in_panel=n,
                                 in_raster=round(int((sub & ras[nm]).sum()) / n, 4)))
    F = Fc | Fl; G = Gc | Gl
    rng = np.random.default_rng(SEED + abs(hash(pid)) % 100000)
    H, W = shape
    prows, rrows = [], []
    for nm, x0, y0, x1, y1 in panels:
        f, g, b = F[y0:y1, x0:x1], G[y0:y1, x0:x1], B[y0:y1, x0:x1]
        fc, gc = Fc[y0:y1, x0:x1], Gc[y0:y1, x0:x1]
        ph, pw = y1 - y0, x1 - x0
        glab, gids, gstat = comps(f, 8)
        n1, m1, _ = run_arms(b, glab, gids, gstat)
        n2, m2, d2 = run_arms(b | g, glab, gids, gstat, want_dil=True)
        # 対照: 同じ頁の別の位置から障壁を取る。R は元の相対位置のまま
        cm1 = cm2 = cn = None
        if pw + 1 < W and ph + 1 < H:
            for _ in range(16):
                nx = int(rng.integers(0, W - pw)); ny = int(rng.integers(0, H - ph))
                if abs(nx - x0) + abs(ny - y0) < 0.10 * (W + H):
                    continue
                cb = B[ny:ny + ph, nx:nx + pw]; cg = G[ny:ny + ph, nx:nx + pw]
                _, c1, _ = run_arms(cb, glab, gids, gstat)
                cn, c2, _ = run_arms(cb | cg, glab, gids, gstat)
                cm1, cm2 = len(c1), len(c2)
                break
        # A5 二次の母集団
        gl2, gi2, gs2 = comps(fc, 8)
        _, s1, _ = run_arms(b, gl2, gi2, gs2)
        _, s2, _ = run_arms(b | gc, gl2, gi2, gs2)
        _, gap_ids, _ = comps(g, 8, 1)
        prows.append(dict(
            name=nm, work=work, pid=pid, panel_area=ph * pw,
            fill_px=int(f.sum()), gap_px=int(g.sum()), barrier_px=int(b.sum()),
            gap_ratio=round(int(g.sum()) / max(int(f.sum()), 1), 5),
            gap_blobs=len(gap_ids), n_gt=len(gids), n_pred1=n1, n_pred2=n2,
            m1=len(m1), m2=len(m2),
            ctl_m1=('' if cm1 is None else cm1), ctl_m2=('' if cm2 is None else cm2),
            ctl_n2=('' if cn is None else cn),
            n_gt2=len(gi2), s1=len(s1), s2=len(s2),
            raster_frac=round(sum(r['px_in_panel'] * r['in_raster']
                                  for r in lay_rows if r['name'] == nm)
                              / max(sum(r['px_in_panel'] for r in lay_rows
                                        if r['name'] == nm), 1), 4)))
        for gi in gids:
            i1 = m1[gi][1] if gi in m1 else 0.0
            i2, i3 = (d2[gi] if gi in d2 else (0.0, 0.0))
            rrows.append(dict(name=nm, work=work, region=gi,
                              area=int(gstat[gi, A]), panel_area=ph * pw,
                              iou1=round(i1, 4), iou2=round(i2, 4), iou2_dil=round(i3, 4),
                              m1=int(gi in m1), m2=int(gi in m2)))
    return dict(error='', pid=pid, panels=prows, regions=rrows, layers=lay_rows)


def gate():
    r = subprocess.run([sys.executable, os.path.join(HERE, 'l14_known_cells.py')],
                       env={**os.environ}, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip()[-2000:])
    if r.returncode != 0:
        print('既知セルが FAIL。本体を起動しない(登録 §5)。'); sys.exit(1)


def main():
    gate()
    os.makedirs(OUT, exist_ok=True)
    fill, line = gated_layers()
    inv = {}
    for r in csv.DictReader(open('/home/sh1/deepl/lineart-fill-region/inbox/'
                                 'page_layers_priority65.csv', encoding='utf-8-sig')):
        inv[(r['page_id'], r['layer_id'])] = (r['name'], r['ancestors'])
    k4 = {(r['page_id'], r['layer_id']): float(r['keep4'])
          for r in csv.DictReader(open('/home/sh1/deepl/lineart-fill-region/results/'
                                       'l10_keep4_all_layers/keep4.csv'))}

    def decorate(pid, lst, isfill):
        out = []
        for lid, path in sorted(lst):
            nm, anc = inv.get((pid, str(lid)), ('', ''))
            out.append((lid, path, nm, anc, k4.get((pid, str(lid)), 0.0), isfill))
        return out

    man = {r['name']: r for r in csv.DictReader(open(f'{KOMA}/manifest.csv'))}
    panels = collections.defaultdict(list)
    for r in csv.DictReader(open(L12)):
        m = man[r['name']]
        pid = m['pid'].replace('__', '_')
        panels[pid].append((r['name'], int(m['x0']), int(m['y0']),
                            int(m['x1']), int(m['y1'])))
    jobs = []
    for (pid, work), fl in sorted(fill.items()):
        if pid not in panels:
            continue
        jobs.append((pid, work, decorate(pid, fl, True),
                     decorate(pid, line.get((pid, work), []), False), panels[pid]))
    lim = int(os.environ.get('L14_LIMIT', '0'))
    if lim:
        jobs = jobs[:lim]
        panels = {k: v for k, v in panels.items() if k in {j[0] for j in jobs}}
        print(f'  smoke: 先頭 {lim} 頁だけ')
    print(f'\n{len(jobs)} 頁 / {sum(len(p) for p in panels.values())} コマ'
          f' / 面層 {sum(len(j[2]) for j in jobs)} 線層 {sum(len(j[3]) for j in jobs)}')
    P, R, Lr = [], [], []
    with Pool(4) as pool:
        for i, res in enumerate(pool.imap_unordered(one_page, jobs), 1):
            if res['error']:
                print('FAIL', res['error']); sys.exit(1)
            P += res['panels']; R += res['regions']; Lr += res['layers']
            if i % 10 == 0:
                print(f'  {i}/{len(jobs)} 頁  面 {len(R)}', flush=True)
    # ---- KC10 は結果を1つも印字する前に見る(L12 の失敗の修理)
    ok = len(R) < N_GT_L12
    print(f'\nKC10 母集団: 面 {len(R)} < L12 の n_gt 合計 {N_GT_L12}  '
          f'{"PASS" if ok else "FAIL"}')
    for nm, rows, fields in (('per_panel', P, list(P[0].keys())),
                             ('per_region', R, list(R[0].keys())),
                             ('layer_in_raster', Lr, list(Lr[0].keys()))):
        with open(f'{OUT}/{nm}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'-> {OUT}/{nm}.csv  {len(rows)} 行')
    if not ok:
        print('KC10 FAIL。腕の数値は読まない(登録 §5)。'); sys.exit(1)
    report(P, R, Lr)


def rec(rows, key, gt='n_gt'):
    a = sum(r[key] for r in rows); b = sum(r[gt] for r in rows)
    return a / b if b else 0.0


def report(P, R, Lr):
    print('\n' + '=' * 78)
    print('L14 腕. プールを跨いで平均しない。面をプールした値と、コマごと中央の両方')
    print(f'{"作品":5} {"コマ":>4} {"面":>5} {"腕1 recall":>11} {"腕2 recall":>11} '
          f'{"腕1 prec":>9} {"腕2 prec":>9} {"対照2":>7}')
    for w in ('ako6', 'ako7'):
        s = [r for r in P if r['work'] == w]
        c = [r for r in s if r['ctl_m2'] != '']
        print(f'{w:5} {len(s):4d} {sum(r["n_gt"] for r in s):5d} '
              f'{rec(s,"m1"):11.4f} {rec(s,"m2"):11.4f} '
              f'{sum(r["m1"] for r in s)/max(sum(r["n_pred1"] for r in s),1):9.4f} '
              f'{sum(r["m2"] for r in s)/max(sum(r["n_pred2"] for r in s),1):9.4f} '
              f'{rec(c,"ctl_m2"):7.4f}')
    print()
    for w in ('ako6', 'ako7'):
        s = [r for r in P if r['work'] == w and r['n_gt'] > 0]
        if not s:
            print(f'  {w}: 面を持つコマが無い'); continue
        for lab, k in (('腕1', 'm1'), ('腕2', 'm2')):
            v = st.median([r[k] / r['n_gt'] for r in s])
            print(f'  {w} {lab} コマごと recall 中央 {v:.4f}  (n={len(s)} コマ)')
    print('\n--- 棒 ---')
    for w in ('ako6', 'ako7'):
        s = [r for r in P if r['work'] == w]
        if not s or not sum(r['n_gt'] for r in s):
            print(f'  {w}: 面が無い'); continue
        c = [r for r in s if r['ctl_m2'] != '']
        r1, r2 = rec(s, 'm1'), rec(s, 'm2')
        rc = rec(c, 'ctl_m2')
        print(f'  {w}  P20 腕1 recall {r1:.4f} >= 0.50 -> {"PASS" if r1>=0.50 else "FAIL"}')
        print(f'      P21 腕2 recall {r2:.4f} >= 0.80 -> {"PASS" if r2>=0.80 else "FAIL"}')
        print(f'      P22 腕2-腕1 {r2-r1:+.4f} >= 0.20 -> '
              f'{"PASS" if r2-r1>=0.20 else "FAIL"}')
        gr = st.median([x['gap_ratio'] for x in s])
        print(f'      P23 GAPFILL/FILL コマ中央 {gr:.4f} <= 0.05 -> '
              f'{"PASS" if gr<=0.05 else "FAIL"} (開示: 全体 0.0096 を先に見ている)'
              f'  成分数 中央 {st.median([x["gap_blobs"] for x in s]):.0f}/コマ'
              f'  面 中央 {st.median([x["n_gt"] for x in s]):.0f}/コマ')
        d = [x for x in R if x['work'] == w and x['m2']]
        if d:
            dm = st.median([x['iou2_dil'] - x['iou2'] for x in d])
            print(f'      P24 腕3 IoU 変化 中央 {dm:+.4f} > 0 -> '
                  f'{"PASS" if dm>0 else "FAIL"}  (IoU 中央 {st.median([x["iou2"] for x in d]):.4f}'
                  f' -> {st.median([x["iou2_dil"] for x in d]):.4f}, n={len(d)})')
        print(f'      P25 対照 recall {rc:.4f} <= 0.10 かつ 腕2-対照 {r2-rc:+.4f} >= 0.30 -> '
              f'{"PASS" if (rc<=0.10 and r2-rc>=0.30) else "FAIL"}')
    print('\n--- 面の大きさ別 (開示。棒は立てない。母集団は分けない) ---')
    for w in ('ako6', 'ako7'):
        rs = [r for r in R if r['work'] == w]
        if not rs:
            continue
        pa = {r['name']: r['panel_area'] for r in P}
        for lab, lo, hi in (('コマの<5%', 0.0, 0.05), ('5-25%', 0.05, 0.25),
                            ('25-75%', 0.25, 0.75), ('>=75%', 0.75, 1.01)):
            g = [r for r in rs if lo <= r['area'] / pa[r['name']] < hi]
            if not g:
                continue
            print(f'  {w} {lab:9} 面 {len(g):5d}  腕1 {sum(r["m1"] for r in g)/len(g):.4f}'
                  f'  腕2 {sum(r["m2"] for r in g)/len(g):.4f}'
                  f'  面積中央 {st.median([r["area"] for r in g]):,.0f}px')
    print('\n--- A5 二次の母集団 (描き文字の層を抜く)。結論は一次 ---')
    for w in ('ako6', 'ako7'):
        s = [r for r in P if r['work'] == w]
        print(f'  {w}: 面 {sum(r["n_gt"] for r in s)} -> {sum(r["n_gt2"] for r in s)}  '
              f'腕1 {rec(s,"m1"):.4f} -> {rec(s,"s1",gt="n_gt2"):.4f}  '
              f'腕2 {rec(s,"m2"):.4f} -> {rec(s,"s2",gt="n_gt2"):.4f}')
    print('\n--- IoU の分布 (腕2, 0.0->1.0 の10分位) ---')
    for w in ('ako6', 'ako7'):
        h = [0] * 10
        for r in R:
            if r['work'] == w:
                h[min(9, int(r['iou2'] * 10))] += 1
        print(f'  {w}: {h}')
    print('\n--- A6 クレンジング: 層の墨のうちデータセットのラスタに在る割合 (棒なし) ---')
    allr = [r for r in Lr]
    tot = sum(r['px_in_panel'] for r in allr)
    inr = sum(r['px_in_panel'] * r['in_raster'] for r in allr)
    print(f'  コマ内に 500px 以上ある層×コマ {len(allr)} 件、墨 {tot:,}px のうち'
          f' ラスタに在る {inr/tot:.4f}')
    h = [0] * 10
    for r in allr:
        h[min(9, int(r['in_raster'] * 10))] += 1
    print(f'  in_raster の10分位 (件数): {h}')
    low = [r for r in allr if r['in_raster'] < 0.5]
    print(f'  in_raster < 0.5 の層×コマ {len(low)} 件 ({len(low)/len(allr):.1%})、'
          f'墨 {sum(r["px_in_panel"] for r in low):,}px '
          f'({sum(r["px_in_panel"] for r in low)/tot:.1%})')
    cnt = collections.Counter()
    px = collections.Counter()
    for r in low:
        cnt[r['layer_name']] += 1; px[r['layer_name']] += r['px_in_panel']
    print('  落ちている層の名前 上位 (A7: 名前は出所の説明。母集団は切らない)')
    for nm, n in cnt.most_common(12):
        print(f'    {n:5d} 件 {px[nm]:12,}px  {nm}')
    pan = collections.defaultdict(lambda: [0, 0])
    for r in allr:
        pan[r['name']][0] += r['px_in_panel']
        pan[r['name']][1] += r['px_in_panel'] * r['in_raster']
    v = sorted(b / a for a, b in pan.values())
    print(f'  コマごとの raster_frac: 中央 {v[len(v)//2]:.4f} / 最小 {v[0]:.4f} / '
          f'最大 {v[-1]:.4f} / <0.90 のコマ {sum(1 for x in v if x<0.90)}/{len(v)}')


if __name__ == '__main__':
    main()
