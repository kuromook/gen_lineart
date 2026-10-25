# -*- coding: utf-8 -*-
"""障壁の作り方を変えて測り直す (診断。棒なし)。

作者の指摘「コマで区切ってる部分は線が途切れている」への対応。独立したコマ枠レイヤは
存在せず、コマの区切りは `線画` という名前の**面形**レイヤのベタとして描かれている
(cov 0.10-0.35 / keep4 1.000 / 塊1-3個)。形の試験では面層 FILL に行くので、
いまの障壁には入っていない。

そこで作者のツール設定 (「線画レイヤとベタレイヤの両方を判定」) に合わせる:

  V0 = 線層                            いまの L14/L15。非循環
  V1 = 線層 + ベタ層の細い印            L14 の腕2。非循環
  V2 = V1 + 作者のベタのうち R 以外      枠はここに自動的に入る。**半循環**
  V3 = V2 + 遮蔽物(フキダシ・3Dの白い面) **半循環**

**V2/V3 は「線からベタを予測できる」とは読めない。** 読めるのは「作者のキャンバスの
状態を与えたとき、ツールは決定論的に R を再現するか」= こちらの工程理解の点検。
対照 (平行移動、seed 20261026) を V3 につける。
"""
import csv, collections, os, sys, statistics as st
import zlib
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import comps, gated_layers, A, MINREG
from l14_arms import acc_add
from l15_known_cells import boundary
from l15_occlusion import excluded_layers

KOMA = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
L12  = '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions/per_panel.csv'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l15_open_runs'
SEED = 20261026


def best_iou(R, lab, stats):
    """R と最も重なる自由成分の IoU と 大きさの比。"""
    n = int(R.sum())
    sl = lab[R]
    v = sl[sl > 0]
    if v.size == 0:
        return 0.0, 0.0
    vals, cnt = np.unique(v, return_counts=True)
    k = int(vals[int(np.argmax(cnt))])
    inter = int(cnt.max()); area = int(stats[k, A])
    return inter / (area + n - inter), area / n


def flood(barrier):
    n, lab, stats, _ = cv2.connectedComponents((~barrier).astype(np.uint8), 4), None, None, None
    return n


def one_page(job):
    pid, work, fl, ll, ex, panels = job
    F = B = G = O = None
    for lid, path in fl:
        ink = cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128
        if F is None:
            F = np.zeros(ink.shape, bool); G = np.zeros(ink.shape, bool)
            B = np.zeros(ink.shape, bool); O = np.zeros(ink.shape, bool)
        acc_add(F, None, ink, 'solid'); acc_add(G, None, ink, 'thin')
    for lid, path in ll:
        acc_add(B, None, cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128, 'cover')
    for lid, path, name, reason in ex:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        acc_add(O, None, im < 128, 'cover')
    rng = np.random.default_rng(SEED + zlib.crc32(pid.encode()) % 100000)
    # **`hash(str)` は過程ごとに変わる**ので種に使えない(2026-10-25、罠 T97)。
    # crc32 は過程に依らない。直す前の走行の帰無値は再現しない
    rows = []
    for nm, x0, y0, x1, y1 in panels:
        f, b, g, o = (F[y0:y1, x0:x1], B[y0:y1, x0:x1],
                      G[y0:y1, x0:x1], O[y0:y1, x0:x1])
        H, W = f.shape
        glab, gids, gstat = comps(f, 8)
        if not gids:
            continue
        glob = {}
        for v, bar in (('V0', b), ('V1', b | g)):
            n, lab, stats, _ = cv2.connectedComponentsWithStats(
                (~bar).astype(np.uint8), 4)
            glob[v] = (lab, stats, bar)
        for gi in gids:
            R = glab == gi
            n = int(R.sum())
            bn = boundary(R); nb = max(int(bn.sum()), 1)
            rec = dict(name=nm, work=work, region=gi, area=n, arm='real')
            for v in ('V0', 'V1'):
                lab, stats, bar = glob[v]
                iou, ratio = best_iou(R, lab, stats)
                d = cv2.distanceTransform((~bar).astype(np.uint8), cv2.DIST_L2, 3) \
                    if bar.any() else None
                rec[f'{v}_iou'] = round(iou, 4); rec[f'{v}_ratio'] = round(ratio, 3)
            others = f & ~R
            for v, bar in (('V2', b | g | others), ('V3', b | g | others | o)):
                nn, lab, stats, _ = cv2.connectedComponentsWithStats(
                    (~bar).astype(np.uint8), 4)
                iou, ratio = best_iou(R, lab, stats)
                rec[f'{v}_iou'] = round(iou, 4); rec[f'{v}_ratio'] = round(ratio, 3)
                rec[f'{v}_sup'] = round(int((bn & cv2.dilate(
                    bar.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).sum())
                    / nb, 4)
            rows.append(rec)
            # 対照: R をコマ内のランダムな位置へ。障壁は本物のまま (V3)
            yy, xx = gstat[gi, cv2.CC_STAT_TOP], gstat[gi, cv2.CC_STAT_LEFT]
            hh, ww = gstat[gi, cv2.CC_STAT_HEIGHT], gstat[gi, cv2.CC_STAT_WIDTH]
            if ww + 2 >= W or hh + 2 >= H:
                continue
            patch = (glab[yy:yy + hh, xx:xx + ww] == gi)
            for _ in range(16):
                nx = int(rng.integers(1, W - ww - 1)); ny = int(rng.integers(1, H - hh - 1))
                if abs(nx - xx) + abs(ny - yy) < 0.10 * (W + H):
                    continue
                Rs = np.zeros(f.shape, bool); Rs[ny:ny + hh, nx:nx + ww] = patch
                bar = b | g | (f & ~Rs) | o
                nn, lab, stats, _ = cv2.connectedComponentsWithStats(
                    (~bar).astype(np.uint8), 4)
                iou, ratio = best_iou(Rs, lab, stats)
                rows.append(dict(name=nm, work=work, region=gi, area=n, arm='shuffled',
                                 V3_iou=round(iou, 4), V3_ratio=round(ratio, 3)))
                break
    return rows


def main():
    fill, line = gated_layers()
    ex, _ = excluded_layers()
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
            if i % 10 == 0:
                print(f'  {i}/{len(jobs)} 頁  面 {len(rows)}', flush=True)
    keys = ['name', 'work', 'region', 'area', 'arm'] + \
           [f'{v}_{k}' for v in ('V0', 'V1', 'V2', 'V3') for k in ('iou', 'ratio', 'sup')]
    with open(f'{OUT}/barrier_variants.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore'); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'-> {OUT}/barrier_variants.csv  {len(rows)} 行\n' + '=' * 92)
    print('障壁の作り方を変えたときの再現 (面ごと。棒なし)')
    print(f'{"作品":5} {"障壁":4} {"面":>5} {"IoU中央":>8} {"IoU>=0.5":>9} {"IoU>=0.75":>10} '
          f'{"漏れ(比>2)":>10} {"支持中央":>9}')
    for w in ('ako6', 'ako7'):
        s = [r for r in rows if r['work'] == w and r['arm'] == 'real']
        for v in ('V0', 'V1', 'V2', 'V3'):
            iou = [r[f'{v}_iou'] for r in s]
            sup = [r[f'{v}_sup'] for r in s if f'{v}_sup' in r]
            print(f'{w:5} {v:4} {len(s):5d} {st.median(iou):8.4f} '
                  f'{sum(1 for x in iou if x >= 0.5)/len(iou):9.4f} '
                  f'{sum(1 for x in iou if x >= 0.75)/len(iou):10.4f} '
                  f'{sum(1 for r in s if r[f"{v}_ratio"] > 2)/len(s):10.4f} '
                  f'{(st.median(sup) if sup else float("nan")):9.4f}')
        c = [r for r in rows if r['work'] == w and r['arm'] == 'shuffled']
        if c:
            iou = [r['V3_iou'] for r in c]
            print(f'{w:5} {"対照V3":4} {len(c):5d} {st.median(iou):8.4f} '
                  f'{sum(1 for x in iou if x >= 0.5)/len(iou):9.4f} '
                  f'{sum(1 for x in iou if x >= 0.75)/len(iou):10.4f} '
                  f'{sum(1 for r in c if r["V3_ratio"] > 2)/len(c):10.4f}')
    print('\n面積帯ごと (V3)')
    for w in ('ako6', 'ako7'):
        for lab, lo, hi in (('<2k', 0, 2000), ('2k-20k', 2000, 20000),
                            ('20k-200k', 20000, 200000), ('>=200k', 200000, 10**12)):
            s = [r for r in rows if r['work'] == w and r['arm'] == 'real'
                 and lo <= r['area'] < hi]
            if not s:
                continue
            print(f'  {w} {lab:9} 面 {len(s):5d}  V0 {sum(1 for r in s if r["V0_iou"]>=0.5)/len(s):.3f}'
                  f'  V1 {sum(1 for r in s if r["V1_iou"]>=0.5)/len(s):.3f}'
                  f'  V2 {sum(1 for r in s if r["V2_iou"]>=0.5)/len(s):.3f}'
                  f'  V3 {sum(1 for r in s if r["V3_iou"]>=0.5)/len(s):.3f}')


if __name__ == '__main__':
    main()
