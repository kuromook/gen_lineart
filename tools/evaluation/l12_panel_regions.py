# -*- coding: utf-8 -*-
"""L12: 撤回した FP率 71.3% の置き換え。正解を2,963層の全部から作り直す。

事前登録は doc/work_log.md「2026-10-24 L12 事前登録」。回す前に commit eb6067e。
正解の作り方の修正 A1-A3 は同 doc の「L12 修正」節。棒 (P12-P15) は一切動かしていない。

正解 (選別は全部こちら側。開示するのが利得):
    truth_r(page) := ∪_L fill_mask_r(coverage(L))
    通す層: preview_dark>=0.5 かつ 被覆<画布の95% かつ 第1階層が /text /sketch set でない
    使わない: 名前の試験・色の試験・面積の床 (L8 がこの3つが切った中身を出している)
層ごとに画素を切る。層ごと選抜しない (b556d1a)。

予測側は L7 と同一・無改変: pred = fill_mask_4(ink_of(データセットの線画ラスタ))

段1 でページごとの truth を書き出し、段2 で測る。P15 の陰性対照が「別ページの正解」を
要るので、同じ正解を2度作らないため。中間は scratchpad に置き、ファイル名で上書きする
(削除はしない。段2 は page_truth.csv に載ったページしか読まない)。
"""
import csv, json, os, sys, collections, statistics as st
from multiprocessing import Pool
import numpy as np, cv2

sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import fill_mask, ink_of
from region_metrics import match, MIN_REGION_PX, MATCH_IOU
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from pid_work import work_of  # **既定値に落とさない**(罠 T104 の族)

INV   = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
KEEP4 = '/home/sh1/deepl/lineart-fill-region/results/l10_keep4_all_layers/keep4.csv'
FORM  = '/home/sh1/deepl/lineart-fill-region/results/l5_ako7_generation/layer_form.csv'
L6CSV = '/home/sh1/deepl/lineart-fill-region/results/l6_clean_subset/per_panel.csv'
BASE  = '/home/sh1/deepl/lineart/data/incoming_manuscript'
KOMA  = '/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826'
TRUTH = os.environ.get('L12_TRUTH', '')   # 2026-10-25: KeyError で import が死ぬと、
                                          # 選別の内訳を数え直すだけでも読めなくなる。
                                          # 段1 は書く前に理由を印字して止まる
OUT   = '/home/sh1/deepl/lineart-fill-region/results/l12_panel_regions'
WORKS = [('ako6', '018_2018_ako6'), ('ako7', '026_2019_ako7')]
PAD, COV_MAX, DARK_MIN = 32, 0.95, 0.5
DROP_TOP = ('text', 'sketch set')     # A1/A2: 下絵は入力画像であって作者のベタではない / 描き文字
MIN_INK_FRAC = 0.002                  # L6 と同じコマの採用条件


def top_folder(anc):
    parts = [p for p in anc.split('/') if p]
    return parts[0] if parts else ''


def selfcheck():
    ok = True
    for r in (4, 6):
        d = np.zeros((200, 200), bool); yy, xx = np.ogrid[:200, :200]
        d[(yy - 100) ** 2 + (xx - 100) ** 2 <= 20 ** 2] = True
        kd = fill_mask(d, radius=r).sum() / d.sum()
        l = np.zeros((200, 200), bool); l[100:103, 20:180] = True
        kl = fill_mask(l, radius=r).sum() / l.sum()
        good = kd > 0.95 and kl == 0.0
        ok &= good
        print(f'KC4 器具 r={r}: 円盤 r=20 {kd:.4f} (>0.95) / 3px線 {kl:.4f} (==0)  '
              f'{"PASS" if good else "FAIL"}')
    return ok


# ---------------------------------------------------------------- 段1: 正解
def build_page(job):
    pid, work, main, textl = job
    acc = {'t4': None, 't6': None, 'ttxt': None}
    shape = [None]

    def add(key, path, radius):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return -1
        ink = im < 128
        shape[0] = shape[0] or ink.shape
        if not ink.any():
            return 0
        ys, xs = np.nonzero(ink)
        y0, y1 = max(0, ys.min() - PAD), min(ink.shape[0], ys.max() + 1 + PAD)
        x0, x1 = max(0, xs.min() - PAD), min(ink.shape[1], xs.max() + 1 + PAD)
        fm = fill_mask(ink[y0:y1, x0:x1], radius=radius)
        n = int(fm.sum())
        if n:
            if acc[key] is None:
                acc[key] = np.zeros(ink.shape, bool)
            acc[key][y0:y1, x0:x1] |= fm
        return n

    used = 0
    for lid, path in main:
        n4 = add('t4', path, 4)
        add('t6', path, 6)
        if n4 < 0:
            return dict(pid=pid, error=f'read L{lid}')
        used += bool(n4)
    for lid, path in textl:
        if add('ttxt', path, 4) < 0:
            return dict(pid=pid, error=f'read text L{lid}')
    if shape[0] is None:
        return dict(pid=pid, error='no layer readable')
    rec = dict(pid=pid, error='', n_main=len(main), n_used=used, n_text=len(textl),
               shape=f'{shape[0][0]}x{shape[0][1]}')
    for key in ('t4', 't6', 'ttxt'):
        m = acc[key] if acc[key] is not None else np.zeros(shape[0], bool)
        cv2.imwrite(os.path.join(TRUTH, f'{work}_{pid}_{key}.png'),
                    (m * 255).astype(np.uint8))
        rec[f'{key}_px'] = int(m.sum())
    return rec


def stage1():
    if not TRUTH or not os.path.isdir(TRUTH):
        print('段1 は回せない: L12_TRUTH(正解画像の置き場)が無い。'
              'ここに t4/t6/ttxt の png を書く', file=sys.stderr)
        sys.exit(1)
    cov = {(r['page_id'], r['layer_id']): float(r['cov'])
           for r in csv.DictReader(open(KEEP4))}
    main = collections.defaultdict(list); textl = collections.defaultdict(list)
    cut = collections.Counter()
    for r in csv.DictReader(open(INV, encoding='utf-8-sig')):
        if r['folder'] != 'False' or r['rendered'] != 'True':
            continue
        try:
            dr = float(r['drawn_ratio'])
        except (TypeError, ValueError):
            continue
        if dr <= 0:
            continue
        pid, lid = r['page_id'], r['layer_id']
        pd_raw = r['preview_dark'].strip()
        if pd_raw == '':
            cut['preview_dark が空 (抽出側 注意4)'] += 1; continue
        if float(pd_raw) < DARK_MIN:
            cut[f'preview_dark < {DARK_MIN} (紙面で暗くない)'] += 1; continue
        c = cov.get((pid, lid))
        if c is None:
            cut['keep4.csv に無い'] += 1; continue
        if c >= COV_MAX:
            cut[f'被覆 >= 画布の{COV_MAX:.0%} (用紙・全面トーン)'] += 1; continue
        work = work_of(pid, r.get('page_path'))
        path = f'{BASE}/{work}/all_layers/{pid}_L{lid}.png'
        tf = top_folder(r['ancestors'])
        if tf in DROP_TOP:
            cut[f'第1階層 /{tf} (A1/A2)'] += 1
            if tf == 'text':
                textl[(pid, work)].append((int(lid), path))
            continue
        main[(pid, work)].append((int(lid), path))

    print('\n正解に入れる層の選別 (これがこちら側の選別。全部開示する)')
    print(f'  通した層 {sum(len(v) for v in main.values())}')
    for k, v in cut.most_common():
        print(f'  落とした層 {v:5d}  {k}')
    pages = sorted(set(main) | set(textl))
    print(f'  ページ {len(pages)}')

    jobs = [(pid, work, sorted(main.get((pid, work), [])), sorted(textl.get((pid, work), [])))
            for pid, work in pages]
    lim = int(os.environ.get('L12_LIMIT', '0'))
    if lim:
        jobs = jobs[:lim]; print(f'  smoke: 先頭 {lim} ページだけ')
    recs = []
    with Pool(5) as pool:
        for i, rec in enumerate(pool.imap_unordered(build_page, jobs), 1):
            recs.append(rec)
            print(f'  段1 {i}/{len(jobs)} {rec["pid"]} t4={rec.get("t4_px",0):,} '
                  f'txt={rec.get("ttxt_px",0):,}', flush=True)
    bad = [r for r in recs if r['error']]
    if bad:
        print('FAIL 段1', bad[:3]); sys.exit(1)
    with open(f'{OUT}/page_truth.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['pid', 'error', 'n_main', 'n_used', 'n_text',
                                           'shape', 't4_px', 't6_px', 'ttxt_px'])
        w.writeheader()
        for r in sorted(recs, key=lambda r: r['pid']):
            w.writerow(r)
    print(f'-> {OUT}/page_truth.csv')
    return recs


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if not selfcheck():
        sys.exit(1)
    stage1()
    print('段1 完了')
