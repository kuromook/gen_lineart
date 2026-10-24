# -*- coding: utf-8 -*-
"""L10: 優先65ページの全2,963層に keep4 を当てる。

事前登録は doc/work_log.md「2026-10-24 L10 事前登録」。回す前に commit 1004fba。

  keep4(層) := |fill_mask(層の被覆)| / |層の被覆|

納品の値は `255 - 被覆` で色ではない。gray<128 は「墨」ではなく「その層が塗った範囲」。
keep4 が答えるのは「この層は面か線か」であって「ベタか」ではない。
"""
import csv, json, os, sys
from multiprocessing import Pool
import numpy as np, cv2

sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import fill_mask

INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
DIRS = {'ako6': '/home/sh1/deepl/lineart/data/incoming_manuscript/ako6/all_layers',
        'ako7': '/home/sh1/deepl/lineart/data/incoming_manuscript/ako7/all_layers'}
OUT = '/home/sh1/deepl/lineart-fill-region/results/l10_keep4_all_layers'
PAD = 32          # 半径4の開きは局所的なので bbox + 余白で足りる(2*4=8 より十分大きい)
MINBLOB = 500     # 連結成分の下限(J の器具と同じ)

def selfcheck():
    d = np.zeros((200, 200), bool); yy, xx = np.ogrid[:200, :200]
    d[(yy - 100) ** 2 + (xx - 100) ** 2 <= 20 ** 2] = True
    kd = fill_mask(d).sum() / d.sum()
    l = np.zeros((200, 200), bool); l[100:103, 20:180] = True
    kl = fill_mask(l).sum() / l.sum()
    print(f'KC1 器具: 円盤 r=20 keep4 {kd:.4f} (>0.95) / 3px線 keep4 {kl:.4f} (==0)  '
          f'{"PASS" if kd > 0.95 and kl == 0.0 else "FAIL"}')
    return kd > 0.95 and kl == 0.0

def one(job):
    pid, lid, path, dr_csv, h, w = job
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return dict(page_id=pid, layer_id=lid, error='read')
    ink = im < 128
    n = int(ink.sum())
    rec = dict(page_id=pid, layer_id=lid, error='', px=n,
               cov=n / ink.size, dr_csv=dr_csv, shape=f'{im.shape[0]}x{im.shape[1]}')
    if n == 0:
        rec.update(keep4=0.0, kept_px=0, blobs=0, maxblob=0)
        return rec
    ys, xs = np.nonzero(ink)
    y0, y1 = max(0, ys.min() - PAD), min(ink.shape[0], ys.max() + 1 + PAD)
    x0, x1 = max(0, xs.min() - PAD), min(ink.shape[1], xs.max() + 1 + PAD)
    sub = ink[y0:y1, x0:x1]
    fm = fill_mask(sub)
    kept = int(fm.sum())
    nb, lab, stats, _ = cv2.connectedComponentsWithStats(fm.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if nb > 1 else np.array([], int)
    big = areas[areas >= MINBLOB]
    rec.update(keep4=kept / n, kept_px=kept, blobs=int(big.size),
               maxblob=int(areas.max()) if areas.size else 0)
    return rec

def main():
    if not selfcheck():
        sys.exit(1)
    rows = [r for r in csv.DictReader(open(INV, encoding='utf-8-sig'))]
    jobs = []
    for r in rows:
        if r['folder'] != 'False' or r['rendered'] != 'True':
            continue
        try:
            dr = float(r['drawn_ratio'])
        except (TypeError, ValueError):
            continue
        if dr <= 0:
            continue
        w = 'ako6' if '_ako6_' in r['page_id'] else 'ako7'
        p = os.path.join(DIRS[w], f"{r['page_id']}_L{r['layer_id']}.png")
        if not os.path.exists(p):
            print('missing', p); sys.exit(1)
        jobs.append((r['page_id'], int(r['layer_id']), p, dr, 0, 0))
    print(f'{len(jobs)} 層')
    os.makedirs(OUT, exist_ok=True)
    recs = []
    with Pool(6) as pool:
        for i, rec in enumerate(pool.imap_unordered(one, jobs, chunksize=4), 1):
            recs.append(rec)
            if i % 250 == 0:
                print(f'  {i}/{len(jobs)}', flush=True)
    # KC2 データ経路
    bad = [r for r in recs if r['error'] == '' and abs(r['cov'] - r['dr_csv']) > 5e-5]
    print(f'\nKC2 データ経路: 被覆が一覧の drawn_ratio と一致しない層 {len(bad)} / {len(recs)}  '
          f'{"PASS" if not bad else "FAIL"}')
    for r in bad[:5]:
        print(f'   {r["page_id"]} L{r["layer_id"]} cov={r["cov"]:.6f} csv={r["dr_csv"]:.6f}')
    with open(os.path.join(OUT, 'keep4.csv'), 'w', newline='', encoding='utf-8') as fh:
        wtr = csv.DictWriter(fh, fieldnames=['page_id', 'layer_id', 'error', 'px', 'cov',
                                             'dr_csv', 'shape', 'keep4', 'kept_px',
                                             'blobs', 'maxblob'])
        wtr.writeheader()
        for r in sorted(recs, key=lambda r: (r['page_id'], r['layer_id'])):
            wtr.writerow(r)
    print(f'-> {OUT}/keep4.csv')

if __name__ == '__main__':
    main()
