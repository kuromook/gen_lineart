# -*- coding: utf-8 -*-
"""L15 追加診断 (棒なし): 長い開いた区間は何に接しているのか。
作者の仮説「手前に吹き出しや3Dオブジェクトが来る所は、隠れるので描いていない」を測る。

競合する3つの説明を同時に測り、どれでもない残りも出す:
  occ  := 障壁に入れなかった層の被覆 (preview_dark<0.5 の白い吹き出し等、/text, /sketch set)
  edge := コマの枠から 3px 以内
  fill := 自分以外の作者のベタ面 (髪の束が髪全体のベタに開く類)
"""
import csv, collections, os, sys, statistics as st
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import comps, gated_layers, A, DARK_MIN, COV_MAX, topf, DROP_TOP
from l14_arms import acc_add
from l15_known_cells import boundary, endpoints, SUP_PX, END_PX, K8
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from pid_work import work_of  # **既定値に落とさない**(罠 T104 の族)

KOMA = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
L12  = '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions/per_panel.csv'
INV  = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
KEEP4= '/home/sh1/deepl/lineart-fill-region/results/l10_keep4_all_layers/keep4.csv'
BASE = '/home/sh1/deepl/lineart/data/incoming_manuscript'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l15_open_runs'


def excluded_layers():
    """障壁にも母集団にも入らなかった層 = 遮蔽物の候補。落とした理由も持って帰る。"""
    k4 = {(r['page_id'], r['layer_id']): r for r in csv.DictReader(open(KEEP4))}
    out = collections.defaultdict(list)
    why = collections.Counter()
    for r in csv.DictReader(open(INV, encoding='utf-8-sig')):
        if r['folder'] != 'False' or r['rendered'] != 'True':
            continue
        try:
            if float(r['drawn_ratio']) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        kk = k4.get((r['page_id'], r['layer_id']))
        if kk is None or float(kk['cov']) >= COV_MAX:
            continue                       # 用紙・全面トーンは遮蔽物にしない
        pd = r['preview_dark'].strip()
        tf = topf(r['ancestors'])
        reason = None
        if pd == '':
            reason = 'preview_dark が空'
        elif float(pd) < DARK_MIN:
            reason = 'preview_dark < 0.5 (白い・淡い)'
        elif tf in DROP_TOP:
            reason = f'第1階層 /{tf}'
        if reason is None:
            continue
        pid = r['page_id']
        w = work_of(pid, r.get('page_path'))
        out[(pid, w)].append((int(r['layer_id']),
                              f'{BASE}/{w}/all_layers/{pid}_L{r["layer_id"]}.png',
                              r['name'], reason))
        why[reason] += 1
    return out, why


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
    occ_names = collections.Counter()
    for lid, path, name, reason in ex:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        ink = im < 128
        if F is None:
            F = np.zeros(ink.shape, bool); B = np.zeros(ink.shape, bool)
            O = np.zeros(ink.shape, bool)
        acc_add(O, None, ink, 'cover')
        occ_names[(name, reason)] += int(ink.sum())
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
            R = glab == gi
            other = (glab > 0) & ~R
            d_f = (cv2.distanceTransform((~other).astype(np.uint8), cv2.DIST_L2, 3)
                   if other.any() else np.full(f.shape, 1e9, np.float32))
            bn = boundary(R)
            op = bn & (d_b > SUP_PX)
            n, lab = cv2.connectedComponents(op.astype(np.uint8), 8)
            for k in range(1, n):
                u = lab == k
                L = int(u.sum())
                if L == 0:
                    continue
                rec = dict(name=nm, work=work, region=gi, length=L,
                           f_edge=round(int((u & edge).sum()) / L, 4),
                           f_fill=round(int((u & (d_f <= 2)).sum()) / L, 4))
                for t in (2, 8, 16, 32):
                    rec[f'f_occ{t}'] = round(int((u & (d_o <= t)).sum()) / L, 4)
                    rec[f'f_any{t}'] = round(int((u & ((np.minimum(d_o, d_b)) <= t))
                                                 .sum()) / L, 4)
                rec['d_occ_med'] = round(float(np.median(d_o[u])), 1)
                rec['d_any_med'] = round(float(np.median(np.minimum(d_o, d_b)[u])), 1)
                rec['f_none'] = round(int((u & (d_o > 2) & ~edge & (d_f > 2)).sum()) / L, 4)
                rows.append(rec)
    return rows, occ_names


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
    names = collections.Counter()
    with Pool(4) as pool:
        for i, (r, nn) in enumerate(pool.imap_unordered(one_page, jobs), 1):
            rows += r; names.update(nn)
            if i % 15 == 0:
                print(f'  {i}/{len(jobs)} 頁  区間 {len(rows)}', flush=True)
    with open(f'{OUT}/occlusion.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'-> {OUT}/occlusion.csv  {len(rows)} 区間\n' + '=' * 84)
    print('開いた区間は何に接しているか (開いた輪郭の画素で重み付け。棒なし)')
    print(f'{"作品":5} {"長さ帯":9} {"開px":>10} | 遮蔽物 2/8/16/32px | 何でも 2/8/16/32px | '
          f'遮蔽物まで中央 / 何かまで中央')
    for w in ('ako6', 'ako7'):
        s0 = [r for r in rows if r['work'] == w]
        for lab, lo, hi in (('<50px', 0, 50), ('50-200px', 50, 200),
                            ('>=200px', 200, 10 ** 9), ('全部', 0, 10 ** 9)):
            s = [r for r in s0 if lo <= r['length'] < hi]
            if not s:
                continue
            tot = sum(r['length'] for r in s)
            def W(k):
                return sum(r['length'] * r[k] for r in s) / tot
            oc = ' '.join(f'{W(f"f_occ{t}"):.3f}' for t in (2, 8, 16, 32))
            an = ' '.join(f'{W(f"f_any{t}"):.3f}' for t in (2, 8, 16, 32))
            print(f'{w:5} {lab:9} {tot:10,.0f} | {oc} | {an} | '
                  f'{st.median([r["d_occ_med"] for r in s]):7.1f} / '
                  f'{st.median([r["d_any_med"] for r in s]):7.1f}')
    print('\n遮蔽物になった層の名前 上位 (面積)')
    for (nm, reason), v in names.most_common(14):
        print(f'  {v:13,}px  {nm}   [{reason}]')


if __name__ == '__main__':
    main()
