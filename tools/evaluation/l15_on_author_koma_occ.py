# -*- coding: utf-8 -*-
"""作者のコマの上で、障壁にコマ枠線を入れて測り直す (診断。棒なし)。

変わるのは2つだけ:
  母集団  データセットの矩形 -> **作者のコマ**(納品された枠マスクの8連結成分)
  障壁    コマ枠線 `koma_lines` を足せる(本物の線。いまの BARRIER には無い)

自由空間は `コマのマスク ∧ ¬障壁`。**枠が絵を切り抜く**という CSP の挙動に合わせる
(前はデータセットの矩形で「切って」いただけで、縁は障壁ではなかった)。

  W0 = 線層
  W1 = W0 + ベタ層の細い印
  W2 = W1 + コマ枠線                        ← 新しい
  W3 = W2 + 作者のベタのうち R 以外           **半循環**
対照 = R をコマの中でランダムに平行移動 (seed 20261026、コマに9割以上入る位置)。

棒は立てない。前の V0-V3 表 (`results/l15_open_runs/barrier_variants.csv`) と
同じ量を出して並べるためのもの。
"""
import csv, collections, json, os, sys, statistics as st
import zlib
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import comps, A, MINREG
from l14_arms import acc_add
from l15_known_cells import boundary
from l15_occlusion import excluded_layers
from l17_known_cells import arms  # T131: 腕と標準は1か所(KC57)

DEL  = '/home/sh1/deepl/lineart/data/incoming_manuscript'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l15_author_koma_occ'
SEED = 20261026
MIN_FRAC, MAX_FRAC = 0.002, 0.50      # 成分の床 / 頁全体の覆い(モザイク)を外す上限
K4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


def best(R, lab, stats):
    n = int(R.sum())
    v = lab[R]; v = v[v > 0]
    if v.size == 0:
        return 0.0, 0.0
    vals, cnt = np.unique(v, return_counts=True)
    k = int(vals[int(np.argmax(cnt))]); inter = int(cnt.max()); ar = int(stats[k, A])
    return inter / (ar + n - inter), ar / n


def one_page(job):
    pid, work, fl, ll, ex, frames = job
    F = G = B = None
    for lid, path in fl:
        ink = cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128
        if F is None:
            F = np.zeros(ink.shape, bool); G = np.zeros(ink.shape, bool)
            B = np.zeros(ink.shape, bool)
        acc_add(F, None, ink, 'solid'); acc_add(G, None, ink, 'thin')
    for lid, path in ll:
        acc_add(B, None, cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128, 'cover')
    if F is None:
        return []
    H, W = F.shape
    # 前にある遮蔽物の代用: 作者はベタの下に遮蔽物を描かない -> ベタと重ならない部分
    # **半循環** (F を使って遮蔽物を定義する)。構成上 OCCF ∧ F = ∅ なので面を飲まない
    OCC = np.zeros((H, W), bool)
    for lid, path, name, reason in ex:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None or im.shape != (H, W):
            continue
        acc_add(OCC, None, im < 128, 'cover')
    OCCF = OCC & ~F
    kl = cv2.imread(f'{DEL}/{work}/koma/{pid}_koma_lines.png', cv2.IMREAD_GRAYSCALE)
    KL = (kl > 127) if (kl is not None and kl.shape == (H, W)) else np.zeros((H, W), bool)
    rng = np.random.default_rng(SEED + zlib.crc32(pid.encode()) % 100000)
    # **`hash(str)` は過程ごとに変わる**ので種に使えない(2026-10-25、罠 T97)。
    # crc32 は過程に依らない。直す前の走行の帰無値は再現しない
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
            y0 = max(0, stats[ci, cv2.CC_STAT_TOP] - 2)
            x0 = max(0, stats[ci, cv2.CC_STAT_LEFT] - 2)
            y1 = min(H, y0 + stats[ci, cv2.CC_STAT_HEIGHT] + 4)
            x1 = min(W, x0 + stats[ci, cv2.CC_STAT_WIDTH] + 4)
            P = (lab[y0:y1, x0:x1] == ci)
            f = F[y0:y1, x0:x1] & P
            b = B[y0:y1, x0:x1]; g = G[y0:y1, x0:x1]; kll = KL[y0:y1, x0:x1]
            of = OCCF[y0:y1, x0:x1]; oa = OCC[y0:y1, x0:x1]
            glab, gids, gstat = comps(f, 8)
            if not gids:
                continue
            ph, pw = P.shape
            glob = {}
            for v, bar in (('W0', b), ('W1', b | g), ('W2', b | g | kll),
                           ('W4', b | g | kll | of)):
                nn, lb, stt, _ = cv2.connectedComponentsWithStats(
                    (P & ~bar).astype(np.uint8), 4)
                glob[v] = (lb, stt, bar)
            for gi in gids:
                R = glab == gi
                bn = boundary(R); nb = max(int(bn.sum()), 1)
                rec = dict(pid=pid, work=work, layer_id=fr['layer_id'], comp=ci,
                           region=gi, area=int(gstat[gi, A]), panel_area=int(P.sum()),
                           arm='real')
                for v in ('W0', 'W1', 'W2', 'W4'):
                    lb, stt, bar = glob[v]
                    iou, ratio = best(R, lb, stt)
                    rec[f'{v}_iou'] = round(iou, 4); rec[f'{v}_ratio'] = round(ratio, 3)
                    rec[f'{v}_sup'] = round(int((bn & cv2.dilate(
                        (bar | ~P).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    ).sum()) / nb, 4)
                rec['occ_cover'] = round(int((R & oa).sum()) / max(int(R.sum()), 1), 4)
                rec['occf_cover'] = round(int((R & of).sum()) / max(int(R.sum()), 1), 4)
                others = f & ~R
                bar = b | g | kll | others
                nn, lb, stt, _ = cv2.connectedComponentsWithStats(
                    (P & ~bar).astype(np.uint8), 4)
                iou, ratio = best(R, lb, stt)
                rec['W3_iou'] = round(iou, 4); rec['W3_ratio'] = round(ratio, 3)
                rec['W3_sup'] = round(int((bn & cv2.dilate(
                    (bar | ~P).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                ).sum()) / nb, 4)
                bar5 = bar | of
                nn, lb, stt, _ = cv2.connectedComponentsWithStats(
                    (P & ~bar5).astype(np.uint8), 4)
                iou, ratio = best(R, lb, stt)
                rec['W5_iou'] = round(iou, 4); rec['W5_ratio'] = round(ratio, 3)
                rec['W5_sup'] = round(int((bn & cv2.dilate(
                    (bar5 | ~P).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                ).sum()) / nb, 4)
                rows.append(rec)
                # 対照: コマの中でランダムに平行移動 (コマに9割以上入る位置)
                yy, xx = gstat[gi, cv2.CC_STAT_TOP], gstat[gi, cv2.CC_STAT_LEFT]
                hh, ww = gstat[gi, cv2.CC_STAT_HEIGHT], gstat[gi, cv2.CC_STAT_WIDTH]
                if ww + 2 >= pw or hh + 2 >= ph:
                    continue
                patch = (glab[yy:yy + hh, xx:xx + ww] == gi)
                npx = int(patch.sum())
                for _ in range(24):
                    nx = int(rng.integers(1, pw - ww - 1)); ny = int(rng.integers(1, ph - hh - 1))
                    if abs(nx - xx) + abs(ny - yy) < 0.10 * (pw + ph):
                        continue
                    Rs = np.zeros(P.shape, bool); Rs[ny:ny + hh, nx:nx + ww] = patch
                    if int((Rs & P).sum()) < 0.9 * npx:
                        continue
                    bar2 = b | g | kll | (f & ~Rs)
                    nn, lb, stt, _ = cv2.connectedComponentsWithStats(
                        (P & ~bar2).astype(np.uint8), 4)
                    iou3, ratio3 = best(Rs, lb, stt)
                    nn, lb, stt, _ = cv2.connectedComponentsWithStats(
                        (P & ~(bar2 | of)).astype(np.uint8), 4)
                    iou5, ratio5 = best(Rs, lb, stt)
                    rows.append(dict(pid=pid, work=work, layer_id=fr['layer_id'],
                                     comp=ci, region=gi, area=int(gstat[gi, A]),
                                     panel_area=int(P.sum()), arm='shuffled',
                                     W3_iou=round(iou3, 4), W3_ratio=round(ratio3, 3),
                                     W5_iou=round(iou5, 4), W5_ratio=round(ratio5, 3)))
                    break
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    fill, line = arms('gated')
    ex, _ = excluded_layers()
    jobs = []
    kept = 0
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
            kept += len(sel)
            jobs.append((pid, work, sorted(fill[(pid, work)]),
                         sorted(line.get((pid, work), [])), ex.get((pid, work), []), sel))
    print(f'頁 {len(jobs)}  枠 {kept} (配下に描画層がある枠だけ。名前では切らない)')
    rows = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(one_page, jobs), 1):
            rows += r
            if i % 10 == 0:
                print(f'  {i}/{len(jobs)} 頁  行 {len(rows)}', flush=True)
    keys = ['pid', 'work', 'layer_id', 'comp', 'region', 'area', 'panel_area', 'arm',
            'occ_cover', 'occf_cover'] + \
           [f'{v}_{k}' for v in ('W0', 'W1', 'W2', 'W3', 'W4', 'W5')
            for k in ('iou', 'ratio', 'sup')]
    with open(f'{OUT}/per_region.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore'); w.writeheader()
        for r in rows:
            w.writerow(r)
    real = [r for r in rows if r['arm'] == 'real']
    print(f'-> {OUT}/per_region.csv  面 {len(real)} (対照 {len(rows)-len(real)})\n' + '=' * 86)
    print('作者のコマの上で、障壁にコマ枠線を入れた (面ごと。棒なし)')
    print(f'{"作品":5} {"障壁":5} {"面":>5} {"IoU中央":>8} {"IoU>=0.5":>9} {"IoU>=0.75":>10} '
          f'{"漏れ(比>2)":>10} {"支持中央":>9}')
    for w in ('ako6', 'ako7'):
        s = [r for r in real if r['work'] == w]
        if not s:
            continue
        for v in ('W0', 'W1', 'W2', 'W4', 'W3', 'W5'):
            iou = [r[f'{v}_iou'] for r in s]
            sup = [r[f'{v}_sup'] for r in s]
            print(f'{w:5} {v:5} {len(s):5d} {st.median(iou):8.4f} '
                  f'{sum(1 for x in iou if x >= 0.5)/len(iou):9.4f} '
                  f'{sum(1 for x in iou if x >= 0.75)/len(iou):10.4f} '
                  f'{sum(1 for r in s if r[f"{v}_ratio"] > 2)/len(s):10.4f} '
                  f'{st.median(sup):9.4f}')
        c = [r for r in rows if r['work'] == w and r['arm'] == 'shuffled']
        for v in ('W3', 'W5'):
            if not c:
                continue
            iou = [r[f'{v}_iou'] for r in c]
            print(f'{w:5} {"対照"+v:5} {len(c):5d} {st.median(iou):8.4f} '
                  f'{sum(1 for x in iou if x >= 0.5)/len(iou):9.4f} '
                  f'{sum(1 for x in iou if x >= 0.75)/len(iou):10.4f} '
                  f'{sum(1 for r in c if r[f"{v}_ratio"] > 2)/len(c):10.4f}')
        print(f'      遮蔽物が面を覆う割合: 生 OCC 中央 '
              f'{st.median([r["occ_cover"] for r in s]):.4f} (>0.5 の面 '
              f'{sum(1 for r in s if r["occ_cover"]>0.5)/len(s):.3f}) / '
              f'差し引き後 OCCF 中央 {st.median([r["occf_cover"] for r in s]):.4f} '
              f'(>0.5 {sum(1 for r in s if r["occf_cover"]>0.5)/len(s):.3f})')
    print('\n面積帯ごと (W2 = 線+細い印+コマ枠線。非循環の最良)')
    for w in ('ako6', 'ako7'):
        for lab, lo, hi in (('<2k', 0, 2000), ('2k-20k', 2000, 20000),
                            ('20k-200k', 20000, 200000), ('>=200k', 200000, 10**12)):
            s = [r for r in real if r['work'] == w and lo <= r['area'] < hi]
            if not s:
                continue
            print(f'  {w} {lab:9} 面 {len(s):5d}  '
                  f'W0 {sum(1 for r in s if r["W0_iou"]>=0.5)/len(s):.3f}  '
                  f'W1 {sum(1 for r in s if r["W1_iou"]>=0.5)/len(s):.3f}  '
                  f'W2 {sum(1 for r in s if r["W2_iou"]>=0.5)/len(s):.3f}  '
                  f'W4 {sum(1 for r in s if r["W4_iou"]>=0.5)/len(s):.3f}  '
                  f'W3 {sum(1 for r in s if r["W3_iou"]>=0.5)/len(s):.3f}  '
                  f'W5 {sum(1 for r in s if r["W5_iou"]>=0.5)/len(s):.3f}')


if __name__ == '__main__':
    main()
