# -*- coding: utf-8 -*-
"""L11: 「ベース領域の上に線画クリッピング」を画素で確かめる。

事前登録は doc/work_log.md「2026-10-24 L11 事前登録」、回す前に commit 75ed33c。
  contain(line, base) := |line_ink ∧ base_ink| / |line_ink|
名前が合っているかではなく、構造が合っているかを見る。
"""
import csv, os, re, random, statistics as st
from multiprocessing import Pool
import numpy as np, cv2

INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
D = {'ako6': '/home/sh1/deepl/lineart/data/incoming_manuscript/ako6/all_layers',
     'ako7': '/home/sh1/deepl/lineart/data/incoming_manuscript/ako7/all_layers'}
OUT = '/home/sh1/deepl/lineart-fill-region/results/l11_clip_containment'

def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def path(r):
    w = 'ako6' if '_ako6_' in r['page_id'] else 'ako7'
    return os.path.join(D[w], f"{r['page_id']}_L{r['layer_id']}.png")

def ink(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE) < 128

def contain(job):
    lp, bp, tag, pid, anc, lname, bname = job
    li, bi = ink(lp), ink(bp)
    n = int(li.sum())
    if n == 0:
        return None
    return dict(tag=tag, page_id=pid, ancestors=anc, line=lname, base=bname,
                line_px=n, base_px=int(bi.sum()),
                contain=float((li & bi).sum()) / n)

rows = [r for r in csv.DictReader(open(INV, encoding='utf-8-sig'))]
live = [r for r in rows if r['folder'] == 'False' and r['rendered'] == 'True'
        and (f(r['drawn_ratio']) or 0) > 0]

byfolder = {}
for r in live:
    byfolder.setdefault((r['page_id'], r['ancestors']), []).append(r)

matched, bases_by_page = [], {}
for (pid, anc), ls in byfolder.items():
    bs = [r for r in ls if re.search(r'base', r['name'], re.I)]
    lns = [r for r in ls if re.search(r'lineart|線画', r['name'], re.I)]
    bases_by_page.setdefault(pid, []).extend((anc, b) for b in bs)
    for b in bs:
        for l in lns:
            matched.append((path(l), path(b), 'matched', pid, anc, l['name'], b['name']))

# 陰性対照: 同じページの別のフォルダーの base と組み替える
random.seed(0)
shuffled = []
for lp, bp, _, pid, anc, lname, bname in matched:
    others = [(a, b) for a, b in bases_by_page.get(pid, []) if a != anc]
    if not others:
        continue
    a2, b2 = random.choice(others)
    shuffled.append((lp, path(b2), 'shuffled', pid, anc, lname, b2['name']))

print(f'対になった組 {len(matched)}  /  組み替えた組 {len(shuffled)}')

# KC3
b0 = matched[0][1]
i0 = ink(b0)
print(f'KC3 base と自分自身の contain: {float((i0 & i0).sum())/i0.sum():.4f}  '
      f'{"PASS" if (i0 & i0).sum() == i0.sum() else "FAIL"}')

with Pool(6) as pool:
    recs = [r for r in pool.imap_unordered(contain, matched + shuffled, chunksize=2) if r]

def row(sel, lab):
    if not sel:
        print(f'  {lab:34} n=0'); return None
    v = [r['contain'] for r in sel]
    print(f'  {lab:34} n={len(v):4d}  中央 {st.median(v):.4f}  '
          f'>=0.8 {sum(1 for x in v if x >= 0.8)/len(v):.3f}  '
          f'下側10% {sorted(v)[len(v)//10]:.4f}')
    return st.median(v)

print('\nP11 / P12')
mm = [r for r in recs if r['tag'] == 'matched']
ss = [r for r in recs if r['tag'] == 'shuffled']
m_all = row(mm, '対になった組')
for w in ('ako6', 'ako7'):
    row([r for r in mm if ('_ako6_' in r['page_id']) == (w == 'ako6')], f'  {w}')
seen = set(); ded = []
for r in mm:
    k = (r['line'], r['base'], r['ancestors'], r['line_px'], r['base_px'])
    if k not in seen:
        seen.add(k); ded.append(r)
row(ded, '  別個の構成のみ')
s_all = row(ss, '組み替えた組(陰性対照)')

print(f'\nP11 中央 {m_all:.4f} >= 0.80 ?  {"PASS" if m_all >= 0.80 else "FAIL"}')
if s_all is not None:
    print(f'P12 差 {m_all - s_all:.4f} >= 0.30 ?  '
          f'{"PASS" if m_all - s_all >= 0.30 else "FAIL"}')

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'containment.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
    w.writeheader()
    for r in sorted(recs, key=lambda r: (r['tag'], r['page_id'], r['ancestors'])):
        w.writerow(r)
print(f'-> {OUT}/containment.csv')
