# -*- coding: utf-8 -*-
"""L12 段2: コマごとに測る。登録は doc/work_log.md「L12 事前登録」(commit eb6067e)、
正解の修正は「L12 修正」(commit dd614ed)。棒は動かしていない。

  pred   = fill_mask_4(ink_of(データセットの線画ラスタ))     ... L7 と同一・無改変
  truth_r = 段1 が書いた ∪_L fill_mask_r(coverage(L))      ... r = 4(主) / 6(感度)
  画素: precision / recall / FP率        面: IoU>=0.5 の一対一対応 (J の器具)
  層: leak_all = |truth4 ∧ raster_ink| / |truth4|  で CLEAN_all / LEAKY_all
  陰性対照: 同一作品の別ページの truth4 を同じ bbox で当てる
"""
import csv, json, os, sys, collections, statistics as st
import numpy as np, cv2

sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import fill_mask, ink_of
from region_metrics import match, MIN_REGION_PX, MATCH_IOU

FORM  = 'results/l5_ako7_generation/layer_form.csv'
L6CSV = 'results/l6_clean_subset/per_panel.csv'
L7CSV = 'results/l7_mask_on_clean/per_panel.csv'
BASE  = '/home/sh1/deepl/lineart/data/incoming_manuscript'
KOMA  = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
TRUTH = os.environ['L12_TRUTH']
OUT   = 'results/l12_panel_regions'
WORKS = [('ako6', '018_2018_ako6'), ('ako7', '026_2019_ako7')]
MIN_INK_FRAC = 0.002


def load(work, pid, key):
    p = os.path.join(TRUTH, f'{work}_{pid}_{key}.png')
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return None if im is None else im > 127


def old_truth_page(work, pid, form):
    """L7 が使っていた正解: 納品ベタ層のうち keep4>=0.95。KC5 のためだけに作る。"""
    f = f'{BASE}/{work}/fill/beta_manifest.json'
    man = json.load(open(f))
    pg = [p for p in man['pages'] if p['page'] == pid]
    if not pg or not pg[0]['beta']:
        return None
    out = None
    for b in pg[0]['beta']:
        im = cv2.imread(f'{BASE}/{work}/fill/{b["file"]}', cv2.IMREAD_GRAYSCALE) < 128
        k = form.get((work, pid, b['name'], str(int(im.sum()))))
        if k is not None and k >= 0.95:
            out = im if out is None else (out | im)
    return out


def main():
    form = {}
    for r in csv.DictReader(open(FORM)):
        form[(r['work'], r['page'], r['name'], r['n_px'])] = float(r['keep4'])
    l6 = {r['name']: r for r in csv.DictReader(open(L6CSV))}
    l7 = {r['name']: r for r in csv.DictReader(open(L7CSV))}
    have = {r['pid'] for r in csv.DictReader(open(f'{OUT}/page_truth.csv')) if not r['error']}

    panels = collections.defaultdict(list); workof = {}
    for r in csv.DictReader(open(f'{KOMA}/manifest.csv')):
        for w, wid in WORKS:
            if r['work_id'] == wid:
                pid = r['pid'].replace('__', '_')
                if pid in have:
                    panels[pid].append(r); workof[pid] = w

    # 陰性対照の供与ページ: 同一作品で、並べて次のページ (決定的)
    bywork = collections.defaultdict(list)
    for pid in sorted(panels):
        bywork[workof[pid]].append(pid)
    donor = {}
    for w, ps in bywork.items():
        for i, pid in enumerate(ps):
            donor[pid] = ps[(i + 1) % len(ps)]

    rows = []; kc5 = []
    for n, pid in enumerate(sorted(panels), 1):
        w = workof[pid]
        t4, t6, ttxt = (load(w, pid, k) for k in ('t4', 't6', 'ttxt'))
        if t4 is None:
            print('FAIL truth 読めない', pid); sys.exit(1)
        dn = donor[pid]
        td = load(workof[dn], dn, 't4')
        old = old_truth_page(w, pid, form)
        print(f'  段2 {n}/{len(panels)} {pid} コマ{len(panels[pid])}', flush=True)
        for r in panels[pid]:
            x0, y0, x1, y1 = (int(r[k]) for k in ('x0', 'y0', 'x1', 'y1'))
            g = cv2.imread(f'{KOMA}/line/{r["name"]}', cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            ink = ink_of(g)
            if float(ink.mean()) < MIN_INK_FRAC:
                continue
            if (y1 - y0, x1 - x0) != ink.shape:
                print('   bbox と線画の寸法が違う', r['name'], ink.shape, (y1-y0, x1-x0))
                continue
            pred = fill_mask(ink, radius=4)
            T4, T6 = t4[y0:y1, x0:x1], t6[y0:y1, x0:x1]
            TX = T4 | ttxt[y0:y1, x0:x1]
            TD = td[y0:y1, x0:x1] if (td is not None and td.shape[0] >= y1
                                      and td.shape[1] >= x1) else None
            area = (y1 - y0) * (x1 - x0)
            leak_all = float((T4 & ink).sum()) / max(int(T4.sum()), 1)
            m = match(pred, T4)
            row = dict(name=r['name'], work=w, pid=pid, area=area,
                       pred_px=int(pred.sum()), t4_px=int(T4.sum()), t6_px=int(T6.sum()),
                       tx_px=int(TX.sum()),
                       inter4=int((pred & T4).sum()), inter6=int((pred & T6).sum()),
                       interx=int((pred & TX).sum()),
                       leak_all=round(leak_all, 6),
                       n_pred=m['n_pred'], n_gt=m['n_gt'], matched=m['matched'],
                       reg_prec=m['precision'], reg_rec=m['recall'], reg_f1=m['region_f1'],
                       l6_leak=l6.get(r['name'], {}).get('leak', ''),
                       l6_share_mixed=l6.get(r['name'], {}).get('share_mixed', ''),
                       l7_stratum=l7.get(r['name'], {}).get('stratum', ''),
                       shuf_px=(int(TD.sum()) if TD is not None else ''),
                       shuf_inter=(int((pred & TD).sum()) if TD is not None else ''),
                       donor=dn)
            rows.append(row)
            if old is not None and r['name'] in l7:
                o = old[y0:y1, x0:x1]
                if o.sum():
                    kc5.append((r['name'], int(o.sum()),
                                float((o & ~T4).sum()) / float(o.sum())))
    with open(f'{OUT}/per_panel.csv', 'w', newline='', encoding='utf-8') as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    print(f'\n-> {OUT}/per_panel.csv  コマ {len(rows)}')
    report(rows, kc5)


def pool(rs, ik, pk, tk):
    i = sum(r[ik] for r in rs); p = sum(r[pk] for r in rs); t = sum(r[tk] for r in rs)
    return (i / p if p else float('nan'), i / t if t else float('nan'), p, t)


def report(rows, kc5):
    print('\n' + '=' * 78)
    print('KC5 正解の接続: |旧truth \\ truth4| / |旧truth|  (L7 の129コマ)')
    if kc5:
        v = sorted(x[2] for x in kc5)
        med, mx = st.median(v), max(v)
        ok = med <= 0.05 and mx <= 0.20
        print(f'  n={len(v)}  中央 {med:.4f} (<=0.05)  最大 {mx:.4f} (<=0.20)  '
              f'{"PASS" if ok else "FAIL"}')
        for nm, px, f in sorted(kc5, key=lambda x: -x[2])[:5]:
            print(f'    {f:.4f}  {nm}  旧truth {px:,}px')
    else:
        print('  FAIL 旧truth が1コマも作れていない')

    for sname, sel in (('CLEAN_all (leak_all<0.10)', lambda r: r['leak_all'] < 0.10),
                       ('LEAKY_all (leak_all>=0.50)', lambda r: r['leak_all'] >= 0.50)):
        print(f'\n--- {sname}')
        print(f'{"作品":6} {"コマ":>4} {"prec":>7} {"FP率":>7} {"recall":>7} '
              f'{"prec r6":>8} {"prec +text":>11} {"mask面積":>8} {"真ベタ面積":>10}')
        for w, _ in WORKS:
            rs = [r for r in rows if sel(r) and r['work'] == w]
            if not rs:
                continue
            p4, r4, pp, tt = pool(rs, 'inter4', 'pred_px', 't4_px')
            p6, _, _, _ = pool(rs, 'inter6', 'pred_px', 't6_px')
            px, _, _, _ = pool(rs, 'interx', 'pred_px', 'tx_px')
            ar = sum(r['area'] for r in rs)
            print(f'{w:6} {len(rs):4d} {p4:7.4f} {1-p4:6.1%} {r4:7.4f} '
                  f'{p6:8.4f} {px:11.4f} {pp/ar:8.2%} {tt/ar:10.2%}')
        rs = [r for r in rows if sel(r)]
        if rs:
            rp = [r['reg_prec'] for r in rs if r['n_pred']]
            rr = [r['reg_rec'] for r in rs if r['n_gt']]
            print(f'  面: region_precision コマ平均 {st.mean(rp) if rp else float("nan"):.4f} '
                  f'(n={len(rp)}) / プール {sum(r["matched"] for r in rs)}/'
                  f'{sum(r["n_pred"] for r in rs)} = '
                  f'{sum(r["matched"] for r in rs)/max(sum(r["n_pred"] for r in rs),1):.4f}')
            print(f'      region_recall コマ平均 {st.mean(rr) if rr else float("nan"):.4f}'
                  f'  真の面/コマ 中央 {st.median([r["n_gt"] for r in rs]):.1f}'
                  f'  予測の面/コマ 中央 {st.median([r["n_pred"] for r in rs]):.1f}')
            sh = [r for r in rs if r['shuf_inter'] != '']
            if sh:
                si = sum(r['shuf_inter'] for r in sh); sp = sum(r['pred_px'] for r in sh)
                p4a, _, _, _ = pool(sh, 'inter4', 'pred_px', 't4_px')
                print(f'  P15 陰性対照: 組み替え precision {si/max(sp,1):.4f} (<=0.15) / '
                      f'本来 {p4a:.4f} / 差 {p4a-si/max(sp,1):.4f} (>=0.30)  n={len(sh)}')

    print('\n--- 旧 L6/L7 のラベルでも印字(置き換えを何と並べて読むか)')
    for lab in ('CLEAN', 'LEAKY'):
        for w, _ in WORKS:
            rs = [r for r in rows if r['l7_stratum'] == lab and r['work'] == w]
            if not rs:
                continue
            p4, r4, pp, tt = pool(rs, 'inter4', 'pred_px', 't4_px')
            print(f'  {lab:6} {w:6} {len(rs):4d}コマ  prec {p4:.4f}  FP率 {1-p4:.1%}  '
                  f'recall {r4:.4f}')

    print('\n--- プールの構成')
    for w, _ in WORKS:
        rs = [r for r in rows if r['work'] == w]
        print(f'  {w}: コマ {len(rs)} / ページ {len({r["pid"] for r in rs})}')
    print(f'  全体: コマ {len(rows)} / ページ {len({r["pid"] for r in rows})}'
          f'  (版の対は基盤の規則で実質49〜50ページ。どちらが完成版かは決めない)')


if __name__ == '__main__':
    main()
