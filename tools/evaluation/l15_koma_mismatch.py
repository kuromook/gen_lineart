# -*- coding: utf-8 -*-
"""作者のコマ(フォルダー)と、データセットの矩形コマの食い違いを数で出す (診断。棒なし)。

枠の図形は納品に来ていないので、作者のコマの範囲は**そのフォルダー配下の層の墨**で
代用する(描画層の72.3%・墨の68.2%がフォルダー配下。頁単位の層は配下に無い)。

測るのは2方向:
  分断 split : 作者のコマ1つの墨が、いくつの矩形に散っているか
  併合 merge : 矩形1つの中に、いくつの作者のコマの墨が入っているか
全面層(被覆>=0.95)は除く。`用紙` を入れると全部が全部に重なる。
"""
import csv, collections, os, re, sys, statistics as st
from multiprocessing import Pool
import numpy as np, cv2
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from pid_work import work_of  # **既定値に落とさない**(罠 T104 の族)

HERE = os.path.dirname(os.path.abspath(__file__))
INV  = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
KEEP4= '/home/sh1/deepl/lineart-fill-region/results/l10_keep4_all_layers/keep4.csv'
BASE = '/home/sh1/deepl/lineart/data/incoming_manuscript'
KOMA = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
L12  = '/home/sh1/deepi/lineart-fill-region/results/l12_panel_regions/per_panel.csv' \
       if False else '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions/per_panel.csv'
OUT  = '/home/sh1/deepl/lineart-fill-region/results/l15_koma_mismatch'
COV_MAX = 0.95


def koma_of(anc):
    m = re.search(r'/?(コマ\s*\d+)(/|$)', anc)
    return m.group(1) if m else None


def one_page(job):
    pid, work, layers, boxes = job
    acc = collections.defaultdict(lambda: dict(
        total=0, per=collections.Counter(), outside=0,
        y0=10**9, y1=-1, x0=10**9, x1=-1, n_layer=0))
    for lid, path, km in layers:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        ink = im < 128
        n = int(ink.sum())
        if n == 0:
            continue
        a = acc[km]
        a['total'] += n; a['n_layer'] += 1
        ys, xs = np.nonzero(ink)
        a['y0'] = min(a['y0'], int(ys.min())); a['y1'] = max(a['y1'], int(ys.max()) + 1)
        a['x0'] = min(a['x0'], int(xs.min())); a['x1'] = max(a['x1'], int(xs.max()) + 1)
        inside = 0
        for nm, bx0, by0, bx1, by1 in boxes:
            c = int(ink[by0:by1, bx0:bx1].sum())
            if c:
                a['per'][nm] += c; inside += c
        a['outside'] += n - inside
    rows = []
    for km, a in acc.items():
        if a['total'] == 0:
            continue
        srt = a['per'].most_common()
        tot = a['total']
        cum = 0; n90 = 0
        for _, c in srt:
            cum += c; n90 += 1
            if cum >= 0.9 * (tot - a['outside']) and (tot - a['outside']) > 0:
                break
        rows.append(dict(pid=pid, work=work, koma=km, n_layer=a['n_layer'], ink=tot,
                         bbox=f"{a['x0']},{a['y0']},{a['x1']},{a['y1']}",
                         outside=a['outside'],
                         outside_frac=round(a['outside'] / tot, 4),
                         top_rect=(srt[0][0] if srt else ''),
                         top_share=round((srt[0][1] / tot) if srt else 0.0, 4),
                         n_rect=len(srt), n_rect_90=n90,
                         per=';'.join(f'{k}:{v}' for k, v in srt[:6])))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    k4 = {(r['page_id'], r['layer_id']): float(r['cov'])
          for r in csv.DictReader(open(KEEP4))}
    man = {r['name']: r for r in csv.DictReader(open(f'{KOMA}/manifest.csv'))}
    boxes = collections.defaultdict(list)
    for r in csv.DictReader(open(L12)):
        m = man[r['name']]
        boxes[m['pid'].replace('__', '_')].append(
            (r['name'], int(m['x0']), int(m['y0']), int(m['x1']), int(m['y1'])))
    per_page = collections.defaultdict(list)
    skip = collections.Counter()
    for r in csv.DictReader(open(INV, encoding='utf-8-sig')):
        if r['folder'] != 'False' or r['rendered'] != 'True':
            continue
        pid = r['page_id']
        if pid not in boxes:
            continue
        cov = k4.get((pid, r['layer_id']))
        if cov is None:
            skip['keep4.csv に無い(0画素)'] += 1; continue
        if cov >= COV_MAX:
            skip[f'被覆>={COV_MAX:.0%}(用紙・全面)'] += 1; continue
        km = koma_of(r['ancestors'])
        if km is None:
            skip['コマフォルダー配下でない'] += 1; continue
        w = work_of(pid, r.get('page_path'))
        per_page[(pid, w)].append((int(r['layer_id']),
                                   f'{BASE}/{w}/all_layers/{pid}_L{r["layer_id"]}.png', km))
    print('層の選別 (落としたものを理由別に印字。規律 D-1)')
    print(f'  通した層 {sum(len(v) for v in per_page.values())}')
    for k, v in skip.most_common():
        print(f'  落とした層 {v:5d}  {k}')
    jobs = [(pid, w, v, boxes[pid]) for (pid, w), v in sorted(per_page.items())]
    print(f'  頁 {len(jobs)}')
    rows = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(one_page, jobs), 1):
            rows += r
            if i % 10 == 0:
                print(f'  {i}/{len(jobs)} 頁  作者のコマ {len(rows)}', flush=True)
    with open(f'{OUT}/per_koma.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'-> {OUT}/per_koma.csv  {len(rows)} 作者のコマ\n' + '=' * 80)
    print('分断: 作者のコマ1つの墨が、1つの矩形にどれだけ収まるか')
    for w in ('ako6', 'ako7'):
        s = [r for r in rows if r['work'] == w]
        if not s:
            continue
        ts = [r['top_share'] for r in s]
        print(f'  {w}: 作者のコマ {len(s)}  top_share 中央 {st.median(ts):.4f}  '
              f'>=0.9 の割合 {sum(1 for x in ts if x >= 0.9)/len(s):.3f}  '
              f'>=0.5 {sum(1 for x in ts if x >= 0.5)/len(s):.3f}')
        print(f'      墨の9割を覆うのに要る矩形の数 中央 {st.median([r["n_rect_90"] for r in s]):.1f} / '
              f'触れている矩形の数 中央 {st.median([r["n_rect"] for r in s]):.1f}')
        print(f'      どの矩形にも入らない墨の割合 中央 {st.median([r["outside_frac"] for r in s]):.4f}')
    # 併合: 矩形ごとに作者のコマを集計
    inv = collections.defaultdict(collections.Counter)
    for r in rows:
        for kv in r['per'].split(';'):
            if not kv:
                continue
            nm, c = kv.rsplit(':', 1)
            inv[nm][(r['pid'], r['koma'])] += int(c)
    print('\n併合: 矩形1つの中に、作者のコマがいくつ入っているか')
    byw = collections.defaultdict(list)
    for nm, cnt in inv.items():
        tot = sum(cnt.values())
        srt = cnt.most_common()
        cum = 0; n90 = 0
        for _, c in srt:
            cum += c; n90 += 1
            if cum >= 0.9 * tot:
                break
        byw['ako6' if '_ako6_' in nm else 'ako7'].append((srt[0][1] / tot, len(srt), n90))
    for w in ('ako6', 'ako7'):
        s = byw[w]
        if not s:
            continue
        print(f'  {w}: 矩形 {len(s)}  最大の作者コマが占める割合 中央 '
              f'{st.median([x[0] for x in s]):.4f}  >=0.9 {sum(1 for x in s if x[0]>=0.9)/len(s):.3f}')
        print(f'      墨の9割に要る作者コマの数 中央 {st.median([x[2] for x in s]):.1f} / '
              f'触れている数 中央 {st.median([x[1] for x in s]):.1f}')
    print('\n分断の大きい作者のコマ 上位')
    for r in sorted(rows, key=lambda r: r['top_share'])[:12]:
        print(f'  top_share {r["top_share"]:.3f} 矩形{r["n_rect"]:2d}本 墨 {r["ink"]:10,} '
              f'外 {r["outside_frac"]:.2f}  {r["pid"][-22:]} {r["koma"]}')


if __name__ == '__main__':
    main()
