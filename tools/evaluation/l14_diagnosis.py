# -*- coding: utf-8 -*-
"""L14 の FAIL の診断。棒は立てない(登録 §4 の棒は読み終えて記録済み)。

面ごとに、照合の閾値 0.5 を外した素の値を出す:
  support   := |∂R のうち B∪G から 2px 以内| / |∂R|      線はそこに在るか
  ratio     := |C| / |R|   C = R と最も重なる自由成分      漏れているか痩せているか
  iou0..3   := C を 0..3px 膨張させたときの IoU            工程3の膨張量の掃引
"""
import csv, collections, os, sys, statistics as st
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import comps, gated_layers, A
from l14_arms import acc_add

KOMA = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
L12  = '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions/per_panel.csv'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l14_arms'
K3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


def one_page(job):
    pid, work, fl, ll, panels = job
    F = G = B = None
    for lid, path in fl:
        ink = cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128
        if F is None:
            F = np.zeros(ink.shape, bool); G = np.zeros(ink.shape, bool)
            B = np.zeros(ink.shape, bool)
        acc_add(F, None, ink, 'solid'); acc_add(G, None, ink, 'thin')
    for lid, path in ll:
        acc_add(B, None, cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128, 'cover')
    rows = []
    for nm, x0, y0, x1, y1 in panels:
        f, g, b = F[y0:y1, x0:x1], G[y0:y1, x0:x1], B[y0:y1, x0:x1]
        bar = b | g
        glab, gids, gstat = comps(f, 8)
        if not gids:
            continue
        plab, pids, pstat = comps(~bar, 4)
        d = cv2.distanceTransform((~bar).astype(np.uint8), cv2.DIST_L2, 3)
        for gi in gids:
            yy, xx = gstat[gi, cv2.CC_STAT_TOP], gstat[gi, cv2.CC_STAT_LEFT]
            hh, ww = gstat[gi, cv2.CC_STAT_HEIGHT], gstat[gi, cv2.CC_STAT_WIDTH]
            a0, a1 = max(0, yy - 6), min(f.shape[0], yy + hh + 6)
            c0, c1 = max(0, xx - 6), min(f.shape[1], xx + ww + 6)
            R = glab[a0:a1, c0:c1] == gi
            n = int(R.sum())
            bnd = R & ~cv2.erode(R.astype(np.uint8), K3, 1).astype(bool)
            nb = int(bnd.sum())
            sup = int((bnd & (d[a0:a1, c0:c1] <= 2)).sum()) / max(nb, 1)
            nr, _ = cv2.connectedComponents((bnd & (d[a0:a1, c0:c1] > 2)).astype(np.uint8), 8)
            sl = plab[a0:a1, c0:c1]
            lv, lc = np.unique(sl[R & (sl > 0)], return_counts=True)
            rec = dict(name=nm, work=work, region=gi, area=n, bnd_px=nb,
                       support=round(sup, 4), open_runs=max(nr - 1, 0))
            if len(lv) == 0:
                rec.update(ratio=0.0, iou0=0.0, iou1=0.0, iou2=0.0, iou3=0.0,
                           cov_b=round(int((R & b[a0:a1, c0:c1]).sum()) / n, 4))
                rows.append(rec); continue
            k = int(lv[int(np.argmax(lc))])
            # 自由成分は bbox をはみ出しうるので、成分全体の面積は stats から取る
            C = sl == k
            ious = []
            cur = C
            for t in range(4):
                if t:
                    cur = cv2.dilate(cur.astype(np.uint8), K3, 1).astype(bool)
                i = int((cur & R).sum()); u = int(cur.sum()) + n - i
                ious.append(round(i / u, 4) if u else 0.0)
            rec.update(ratio=round(pstat[k, A] / n, 3), iou0=ious[0], iou1=ious[1],
                       iou2=ious[2], iou3=ious[3],
                       cov_b=round(int((R & b[a0:a1, c0:c1]).sum()) / n, 4))
            rows.append(rec)
    return rows


def main():
    fill, line = gated_layers()
    man = {r['name']: r for r in csv.DictReader(open(f'{KOMA}/manifest.csv'))}
    panels = collections.defaultdict(list)
    for r in csv.DictReader(open(L12)):
        m = man[r['name']]
        panels[m['pid'].replace('__', '_')].append(
            (r['name'], int(m['x0']), int(m['y0']), int(m['x1']), int(m['y1'])))
    jobs = [(pid, w, sorted(fl), sorted(line.get((pid, w), [])), panels[pid])
            for (pid, w), fl in sorted(fill.items()) if pid in panels]
    rows = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(one_page, jobs), 1):
            rows += r
            if i % 10 == 0:
                print(f'  {i}/{len(jobs)} 頁  面 {len(rows)}', flush=True)
    with open(f'{OUT}/diagnosis.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'-> {OUT}/diagnosis.csv  {len(rows)} 面\n')
    rep(rows)


def rep(rows):
    print('=' * 78)
    print('L14 診断 (棒なし). 線はそこに在るか / 漏れか痩せか / 工程3の膨張量')
    for w in ('ako6', 'ako7'):
        s = [r for r in rows if r['work'] == w]
        print(f'\n{w}  面 {len(s)}')
        print(f'  support 中央 {st.median([r["support"] for r in s]):.4f}  '
              f'(L13 の同じ器具・違う母集団: ako6 0.2139 / ako7 0.2443)')
        h = [0] * 10
        for r in s:
            h[min(9, int(r['support'] * 10))] += 1
        print(f'  support の10分位 {h}')
        print(f'  開いた区間 中央 {st.median([r["open_runs"] for r in s]):.1f}')
        leak = [r for r in s if r['ratio'] > 2]; thin = [r for r in s if 0 < r['ratio'] < 0.5]
        mid = [r for r in s if 0.5 <= r['ratio'] <= 2]
        print(f'  自由成分/面: 漏れ(>2) {len(leak)} / 近い(0.5-2) {len(mid)} / '
              f'痩せ(<0.5) {len(thin)} / 成分なし {sum(1 for r in s if r["ratio"]==0)}')
        for lab, k in (('膨張なし', 'iou0'), ('+1px', 'iou1'), ('+2px', 'iou2'),
                       ('+3px', 'iou3')):
            v = [r[k] for r in s]
            print(f'  IoU {lab:8} 中央 {st.median(v):.4f}  >=0.5 の面 '
                  f'{sum(1 for x in v if x >= 0.5):5d} ({sum(1 for x in v if x>=0.5)/len(v):.3f})'
                  f'  >=0.75 {sum(1 for x in v if x >= 0.75):5d}')
        good = [r for r in s if r['support'] >= 0.95]
        print(f'  support>=0.95 の面 {len(good)} ({len(good)/len(s):.3f}):  '
              f'IoU+1px 中央 {st.median([r["iou1"] for r in good]):.4f} / '
              f'>=0.5 {sum(1 for r in good if r["iou1"]>=0.5)/max(len(good),1):.3f}')
        bad = [r for r in s if r['support'] < 0.5]
        if bad:
            print(f'  support<0.50 の面 {len(bad)} ({len(bad)/len(s):.3f}):  '
                  f'IoU+1px 中央 {st.median([r["iou1"] for r in bad]):.4f}  '
                  f'面積中央 {st.median([r["area"] for r in bad]):,.0f}px')
    print('\n面積帯ごと (両作品まとめず作品ごと)')
    for w in ('ako6', 'ako7'):
        for lab, lo, hi in (('<2k', 0, 2000), ('2k-20k', 2000, 20000),
                            ('20k-200k', 20000, 200000), ('>=200k', 200000, 10**12)):
            s = [r for r in rows if r['work'] == w and lo <= r['area'] < hi]
            if not s:
                continue
            print(f'  {w} {lab:9} 面 {len(s):5d}  support中央 '
                  f'{st.median([r["support"] for r in s]):.4f}  IoU+1px 中央 '
                  f'{st.median([r["iou1"] for r in s]):.4f}  >=0.5 '
                  f'{sum(1 for r in s if r["iou1"]>=0.5)/len(s):.3f}')


if __name__ == '__main__':
    main()
