# -*- coding: utf-8 -*-
"""z順で「前にある遮蔽物」を決めて測る (診断。棒なし)。

前回の代用 `OCC ∧ ¬F` は**循環**だった(重なりが無い面では寄与 +0.000、大きく重なる面で
+0.47/+0.55・支持ちょうど 1.000 = 面の輪郭を引き写していた)。今回は納品された
`layer_z_order_priority65.csv` の `z_global`(0 が一番下)で決める。**F を参照しない。**

  遮蔽物の候補 = 障壁にも母集団にも入らなかった層のうち visibility=1 で /sketch set でないもの
  OCC_front(Lf) = ∪ cover(L)  over  z_global(L) > z_global(Lf)
  面 R はその出所の面層 Lf(重なりが最大の層)の z を使う

**先に作者の言明を試験する**: |R ∧ OCC_front| / |R| がほぼ 0 か。
「手前に来るものの下は描いていない」が正しければ 0 に近いはず。これは腕ではなく観察。

  W6 = W2 + OCC_front       W7 = W3 + OCC_front       対照は W7 で取る
"""
import csv, collections, json, os, sys, statistics as st
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import comps, gated_layers, A, topf
from l14_arms import acc_add
from l15_known_cells import boundary
from l15_occlusion import excluded_layers

DEL  = '/home/sh1/deepl/lineart/data/incoming_manuscript'
INV  = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
ZCSV = '/home/sh1/deepl/lineart-fill-region/inbox/layer_z_order_priority65.csv'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l15_zorder_occ'
SEED = 20261026
MIN_FRAC, MAX_FRAC = 0.002, 0.50


def best(R, lab, stats):
    n = int(R.sum()); v = lab[R]; v = v[v > 0]
    if v.size == 0:
        return 0.0, 0.0
    vals, cnt = np.unique(v, return_counts=True)
    k = int(vals[int(np.argmax(cnt))]); inter = int(cnt.max()); ar = int(stats[k, A])
    return inter / (ar + n - inter), ar / n


def one_page(job):
    pid, work, fl, ll, occ, frames = job
    solids = []          # (z, mask) 面層ごと
    F = G = B = None
    for lid, path, z in fl:
        ink = cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128
        if F is None:
            F = np.zeros(ink.shape, bool); G = np.zeros(ink.shape, bool)
            B = np.zeros(ink.shape, bool)
        s = np.zeros(ink.shape, bool)
        acc_add(s, None, ink, 'solid'); acc_add(G, None, ink, 'thin')
        F |= s; solids.append((z, s))
    if F is None:
        return []
    for lid, path in ll:
        acc_add(B, None, cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128, 'cover')
    H, W = F.shape
    # z の高い遮蔽物から積み上げ、面層の z ごとの OCC_front を作る
    occ = sorted(occ, key=lambda t: -t[2])
    cum = np.zeros((H, W), bool); front = {}
    zs = sorted({z for z, _ in solids}, reverse=True)
    oi = 0
    for z in zs:
        while oi < len(occ) and occ[oi][2] > z:
            im = cv2.imread(occ[oi][1], cv2.IMREAD_GRAYSCALE)
            if im is not None and im.shape == (H, W):
                acc_add(cum, None, im < 128, 'cover')
            oi += 1
        front[z] = cum.copy()
    kl = cv2.imread(f'{DEL}/{work}/koma/{pid}_koma_lines.png', cv2.IMREAD_GRAYSCALE)
    KL = (kl > 127) if (kl is not None and kl.shape == (H, W)) else np.zeros((H, W), bool)
    rng = np.random.default_rng(SEED + abs(hash(pid)) % 100000)
    rows = []
    for fr in frames:
        m = cv2.imread(f'{DEL}/{work}/koma/{fr["file"]}', cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != (H, W):
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), 8)
        for ci in range(1, n):
            a = int(stats[ci, A])
            if a < MIN_FRAC * H * W or a > MAX_FRAC * H * W:
                continue
            y0 = max(0, stats[ci, cv2.CC_STAT_TOP] - 2); x0 = max(0, stats[ci, cv2.CC_STAT_LEFT] - 2)
            y1 = min(H, y0 + stats[ci, cv2.CC_STAT_HEIGHT] + 4)
            x1 = min(W, x0 + stats[ci, cv2.CC_STAT_WIDTH] + 4)
            P = (lab[y0:y1, x0:x1] == ci)
            f = F[y0:y1, x0:x1] & P
            b = B[y0:y1, x0:x1]; g = G[y0:y1, x0:x1]; kll = KL[y0:y1, x0:x1]
            glab, gids, gstat = comps(f, 8)
            if not gids:
                continue
            ph, pw = P.shape
            glob = {}
            for v, bar in (('W0', b), ('W1', b | g), ('W2', b | g | kll)):
                nn, lb, stt, _ = cv2.connectedComponentsWithStats((P & ~bar).astype(np.uint8), 4)
                glob[v] = (lb, stt, bar)
            for gi in gids:
                R = glab == gi; npx = int(R.sum())
                bn = boundary(R); nb = max(int(bn.sum()), 1)
                # 出所の面層 = 重なりが最大
                zbest, cbest = zs[-1], -1
                for z, s in solids:
                    c = int((R & s[y0:y1, x0:x1]).sum())
                    if c > cbest:
                        cbest, zbest = c, z
                OF = front[zbest][y0:y1, x0:x1]
                rec = dict(pid=pid, work=work, comp=ci, region=gi, area=npx,
                           panel_area=int(P.sum()), arm='real', src_z=zbest,
                           src_share=round(cbest / max(npx, 1), 4),
                           occf_cover=round(int((R & OF).sum()) / max(npx, 1), 4))
                for v in ('W0', 'W1', 'W2'):
                    lb, stt, bar = glob[v]
                    iou, ratio = best(R, lb, stt)
                    rec[f'{v}_iou'] = round(iou, 4); rec[f'{v}_ratio'] = round(ratio, 3)
                    rec[f'{v}_sup'] = round(int((bn & cv2.dilate(
                        (bar | ~P).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    ).sum()) / nb, 4)
                others = f & ~R
                for v, bar in (('W6', b | g | kll | OF),
                               ('W3', b | g | kll | others),
                               ('W7', b | g | kll | others | OF)):
                    nn, lb, stt, _ = cv2.connectedComponentsWithStats(
                        (P & ~bar).astype(np.uint8), 4)
                    iou, ratio = best(R, lb, stt)
                    rec[f'{v}_iou'] = round(iou, 4); rec[f'{v}_ratio'] = round(ratio, 3)
                    rec[f'{v}_sup'] = round(int((bn & cv2.dilate(
                        (bar | ~P).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    ).sum()) / nb, 4)
                rows.append(rec)
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
                    bar = b | g | kll | (f & ~Rs) | OF
                    nn, lb, stt, _ = cv2.connectedComponentsWithStats(
                        (P & ~bar).astype(np.uint8), 4)
                    iou, ratio = best(Rs, lb, stt)
                    rows.append(dict(pid=pid, work=work, comp=ci, region=gi, area=npx,
                                     panel_area=int(P.sum()), arm='shuffled',
                                     occf_cover=round(int((Rs & OF).sum()) / max(pn, 1), 4),
                                     W7_iou=round(iou, 4), W7_ratio=round(ratio, 3)))
                    break
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    zz = {(r['page_id'], r['layer_id']): int(r['z_global'])
          for r in csv.DictReader(open(ZCSV, encoding='utf-8-sig'))}
    vis = {(r['page_id'], r['layer_id']): (r['visibility'], r['ancestors'])
           for r in csv.DictReader(open(INV, encoding='utf-8-sig'))}
    fill, line = gated_layers()
    ex, _ = excluded_layers()
    cut = collections.Counter()
    occs = {}
    for k, v in ex.items():
        keep = []
        for lid, path, name, reason in v:
            vi, anc = vis.get((k[0], str(lid)), ('', ''))
            if vi != '1':
                cut[f'visibility={vi or "空"} (可視でない)'] += 1; continue
            if topf(anc) == 'sketch set':
                cut['/sketch set (下絵は入力画像。L12 の修正A2)'] += 1; continue
            z = zz.get((k[0], str(lid)))
            if z is None:
                cut['z順に無い'] += 1; continue
            keep.append((lid, path, z))
        occs[k] = keep
    print('遮蔽物の選別 (z順で前後を決める。名前では切らない)')
    print(f'  通した {sum(len(v) for v in occs.values())}')
    for k, v in cut.most_common():
        print(f'  落とした {v:5d}  {k}')
    jobs = []
    for work in ('ako6', 'ako7'):
        d = json.load(open(f'{DEL}/{work}/koma/koma_manifest.json'))
        for p in d['pages']:
            pid = p['page']
            if (pid, work) not in fill:
                continue
            sel = [fr for fr in p['frames']
                   if int((fr.get('ink') or {}).get('layers_with_png') or 0) > 0]
            if not sel:
                continue
            fz = [(lid, path, zz.get((pid, str(lid)), 0))
                  for lid, path in sorted(fill[(pid, work)])]
            jobs.append((pid, work, fz, sorted(line.get((pid, work), [])),
                         occs.get((pid, work), []), sel))
    print(f'  頁 {len(jobs)}')
    rows = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(one_page, jobs), 1):
            rows += r
            if i % 10 == 0:
                print(f'  {i}/{len(jobs)} 頁  行 {len(rows)}', flush=True)
    keys = ['pid', 'work', 'comp', 'region', 'area', 'panel_area', 'arm', 'src_z',
            'src_share', 'occf_cover'] + \
           [f'{v}_{k}' for v in ('W0', 'W1', 'W2', 'W6', 'W3', 'W7')
            for k in ('iou', 'ratio', 'sup')]
    with open(f'{OUT}/per_region.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore'); w.writeheader()
        for r in rows:
            w.writerow(r)
    real = [r for r in rows if r['arm'] == 'real']
    print(f'-> {OUT}/per_region.csv  面 {len(real)}\n' + '=' * 88)
    print('【先に観察】作者の言明「手前に来るものの下は描いていない」の試験')
    print('  |R ∧ OCC_front| / |R|  —— 0 に近ければ言明どおり。**F を参照していない**')
    for w in ('ako6', 'ako7'):
        s = [r for r in real if r['work'] == w]
        v = sorted(r['occf_cover'] for r in s)
        print(f'  {w}: 面 {len(s)}  中央 {st.median(v):.4f}  平均 {sum(v)/len(v):.4f}  '
              f'90分位 {v[int(.9*len(v))]:.4f}  >0.5 の面 {sum(1 for x in v if x>0.5)} '
              f'({sum(1 for x in v if x>0.5)/len(v):.3f})  >0.1 {sum(1 for x in v if x>0.1)/len(v):.3f}')
    print('\n' + '=' * 88)
    print('腕 (面ごと。棒なし)')
    print(f'{"作品":5} {"障壁":5} {"面":>5} {"IoU中央":>8} {"IoU>=0.5":>9} {"IoU>=0.75":>10} '
          f'{"漏れ":>8} {"支持中央":>9}')
    for w in ('ako6', 'ako7'):
        s = [r for r in real if r['work'] == w]
        if not s:
            continue
        for v in ('W0', 'W1', 'W2', 'W6', 'W3', 'W7'):
            iou = [r[f'{v}_iou'] for r in s]
            print(f'{w:5} {v:5} {len(s):5d} {st.median(iou):8.4f} '
                  f'{sum(1 for x in iou if x >= 0.5)/len(iou):9.4f} '
                  f'{sum(1 for x in iou if x >= 0.75)/len(iou):10.4f} '
                  f'{sum(1 for r in s if r[f"{v}_ratio"] > 2)/len(s):8.4f} '
                  f'{st.median([r[f"{v}_sup"] for r in s]):9.4f}')
        c = [r for r in rows if r['work'] == w and r['arm'] == 'shuffled']
        if c:
            iou = [r['W7_iou'] for r in c]
            print(f'{w:5} {"対照W7":5} {len(c):5d} {st.median(iou):8.4f} '
                  f'{sum(1 for x in iou if x >= 0.5)/len(iou):9.4f} '
                  f'{sum(1 for x in iou if x >= 0.75)/len(iou):10.4f} '
                  f'{sum(1 for r in c if r["W7_ratio"] > 2)/len(c):8.4f}')
    print('\n循環の点検: 遮蔽物が面に重なる量で層別 (前回はここで +0.000 と +0.47 に割れた)')
    for w in ('ako6', 'ako7'):
        s0 = [r for r in real if r['work'] == w]
        for lab, lo, hi in (('<0.01', 0, 0.01), ('0.01-0.2', 0.01, 0.2),
                            ('0.2-0.5', 0.2, 0.5), ('>=0.5', 0.5, 1.01)):
            s = [r for r in s0 if lo <= r['occf_cover'] < hi]
            if not s:
                continue
            f2 = sum(1 for r in s if r['W2_iou'] >= 0.5) / len(s)
            f6 = sum(1 for r in s if r['W6_iou'] >= 0.5) / len(s)
            print(f'  {w} occf {lab:9} 面 {len(s):5d}  W2 {f2:.3f}  W6 {f6:.3f}  '
                  f'差 {f6-f2:+.3f}  W6支持中央 {st.median([r["W6_sup"] for r in s]):.3f}')


if __name__ == '__main__':
    main()
