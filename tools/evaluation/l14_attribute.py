# -*- coding: utf-8 -*-
"""L14 の面を、それを出した面層に帰属させる。棒なし。
「その面は何なのか」を名前で答える。名前で母集団は切らない(A7)。"""
import csv, collections, os, sys, statistics as st
from multiprocessing import Pool
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_known_cells import comps, A
from l14_arms import acc_add
from l17_known_cells import arms  # T131: 腕と標準は1か所(KC57)
KOMA = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
L12  = '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions/per_panel.csv'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l14_arms'
INV  = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'

def one_page(job):
    pid, work, fl, panels, meta = job
    per = []
    F = None
    for lid, path in fl:
        ink = cv2.imread(path, cv2.IMREAD_GRAYSCALE) < 128
        m = np.zeros(ink.shape, bool); acc_add(m, None, ink, 'solid')
        per.append((lid, m))
        if F is None:
            F = np.zeros(ink.shape, bool)
        F |= m
    rows = []
    for nm, x0, y0, x1, y1 in panels:
        f = F[y0:y1, x0:x1]
        glab, gids, gstat = comps(f, 8)
        for gi in gids:
            R = glab == gi; n = int(R.sum())
            best, bn = None, 0
            for lid, m in per:
                c = int((R & m[y0:y1, x0:x1]).sum())
                if c > bn:
                    best, bn = lid, c
            name, anc, kp, hm = meta.get(best, ('', '', 0.0, ''))
            rows.append(dict(name=nm, work=work, region=gi, area=n,
                             layer_id=best, share=round(bn / n, 3),
                             layer_name=name, ancestors=anc, keep4=kp, has_mask=hm))
    return rows

def main():
    fill, _ = arms('gated')
    meta = {}
    for r in csv.DictReader(open(INV, encoding='utf-8-sig')):
        meta[(r['page_id'], int(r['layer_id']))] = (r['name'], r['ancestors'],
                                                    r['has_mask'])
    k4 = {(r['page_id'], int(r['layer_id'])): round(float(r['keep4']), 3)
          for r in csv.DictReader(open('/home/sh1/deepl/lineart-fill-region/results/'
                                       'l10_keep4_all_layers/keep4.csv'))}
    man = {r['name']: r for r in csv.DictReader(open(f'{KOMA}/manifest.csv'))}
    panels = collections.defaultdict(list)
    for r in csv.DictReader(open(L12)):
        m = man[r['name']]
        panels[m['pid'].replace('__', '_')].append(
            (r['name'], int(m['x0']), int(m['y0']), int(m['x1']), int(m['y1'])))
    jobs = []
    for (pid, w), fl in sorted(fill.items()):
        if pid not in panels:
            continue
        mm = {lid: (meta[(pid, lid)][0], meta[(pid, lid)][1], k4.get((pid, lid), 0.0),
                    meta[(pid, lid)][2]) for lid, _ in fl}
        jobs.append((pid, w, sorted(fl), panels[pid], mm))
    rows = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(one_page, jobs), 1):
            rows += r
            if i % 15 == 0:
                print(f'  {i}/{len(jobs)} 頁', flush=True)
    d = {(r['name'], r['region']): r for r in
         csv.DictReader(open(f'{OUT}/diagnosis.csv'))}
    for r in rows:
        k = (r['name'], str(r['region']))
        r['support'] = float(d[k]['support']) if k in d else ''
        r['iou1'] = float(d[k]['iou1']) if k in d else ''
        r['ratio'] = float(d[k]['ratio']) if k in d else ''
    with open(f'{OUT}/attribution.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'-> {OUT}/attribution.csv  {len(rows)} 面\n' + '=' * 78)
    print('面を出した層の名前ごと (A7: 名前は説明。母集団は切らない)')
    g = collections.defaultdict(list)
    for r in rows:
        if r['support'] != '':
            g[r['layer_name']].append(r)
    print(f'{"面":>5} {"面積中央":>9} {"support中央":>11} {"IoU+1px中央":>11} '
          f'{">=0.5":>6} {"漏れ":>6} {"mask":>5}  名前')
    for nm, s in sorted(g.items(), key=lambda kv: -len(kv[1]))[:18]:
        print(f'{len(s):5d} {st.median([r["area"] for r in s]):9,.0f} '
              f'{st.median([r["support"] for r in s]):11.4f} '
              f'{st.median([r["iou1"] for r in s]):11.4f} '
              f'{sum(1 for r in s if r["iou1"]>=0.5)/len(s):6.3f} '
              f'{sum(1 for r in s if r["ratio"]>2)/len(s):6.3f} '
              f'{sum(1 for r in s if r["has_mask"]=="True")/len(s):5.2f}  {nm}')
    print('\nクリップマスクの有無で割る (L11 の論点)')
    for lab, v in (('has_mask=True', 'True'), ('has_mask=False', 'False')):
        s = [r for r in rows if r['support'] != '' and r['has_mask'] == v]
        if not s:
            continue
        print(f'  {lab}: 面 {len(s)}  support中央 '
              f'{st.median([r["support"] for r in s]):.4f}  IoU+1px中央 '
              f'{st.median([r["iou1"] for r in s]):.4f}  >=0.5 '
              f'{sum(1 for r in s if r["iou1"]>=0.5)/len(s):.3f}')
    print('\n1つの層が面の何割を出しているか (share) の中央',
          f'{st.median([r["share"] for r in rows]):.3f}',
          f'/ share<0.9 の面 {sum(1 for r in rows if r["share"]<0.9)}/{len(rows)}')

if __name__ == '__main__':
    main()
