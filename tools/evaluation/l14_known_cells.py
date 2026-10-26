# -*- coding: utf-8 -*-
"""L14 既知セル。PASS しなければ本体は起動しない(登録 §5)。

KC8  器具: 線だけ閉じたリング -> 腕1 が照合 / 10px の切れ目 -> 腕1 は漏れる /
           切れ目に 10px の印を置く -> 腕2 が照合。工程1->2 の因果を3行で通す
KC9  データ経路: FILL_SOLID **⊆** truth4、頁単位 (L12 の正解は上位集合。
     **包含であって一致ではない** —— 向きは片方だけ。2026-10-25 時点で回せない)
KC11 器具: 自前 matcher の**組**が記録した定数と一致 (件数は region_metrics とも一致。
     件数の一致は組の一致ではない -- 罠 T101)
KC33 データ経路: `drawn_ratio<=0` **→** `ink_px==0` が**層単位**で例外0 (4,399層)。
     **逆向きは成り立たない** —— `ink_px==0` かつ `drawn_ratio>0` が31層
KC34 データ経路: 台帳の4量 (`drawn_ratio` / `drawn_gray_mean` / `black_among_drawn` /
     `ink_px`) が、**L 側の層 PNG から、アルファの閾値として、層単位で丸めの幅の中で
     再現する** (無作為30層・読めた16層で16一致・ずれ0)。
     **これが「どれも色を見ていない」ことの点検である。**
"""
import csv, os, sys
import numpy as np, cv2

sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import fill_mask
from region_metrics import match as rm_match
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from pid_work import work_of  # **既定値に落とさない**(罠 T104 の族)

MINREG, IOU = 500, 0.5
A = cv2.CC_STAT_AREA
PAD = 32
TRUTH = os.environ.get('L12_TRUTH', '')
INV   = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
KEEP4 = '/home/sh1/deepl/lineart-fill-region/results/l10_keep4_all_layers/keep4.csv'
BASE  = '/home/sh1/deepl/lineart/data/incoming_manuscript'
KEEP4_FILL = 0.95
DARK_MIN, COV_MAX = 0.5, 0.95
DROP_TOP = ('text', 'sketch set')


def comps(mask, conn, min_px=MINREG):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), conn)
    ids = [i for i in range(1, n) if stats[i, A] >= min_px]
    return lab, ids, stats


def match_lab(plab, pids, pstat, glab, gids, gstat, iou_thr=IOU):
    """IoU>=thr の貪欲1対1。{gt_id: (pred_id, iou)} を返す。規則は region_metrics と同じ。"""
    cand = []
    both = (plab > 0) & (glab > 0)
    if both.any():
        pairs, counts = np.unique(np.stack([plab[both], glab[both]]), axis=1,
                                  return_counts=True)
        ps, gs = set(pids), set(gids)
        for (pi, gi), inter in zip(pairs.T, counts):
            pi, gi = int(pi), int(gi)
            if pi not in ps or gi not in gs:
                continue
            union = int(pstat[pi, A]) + int(gstat[gi, A]) - int(inter)
            iou = int(inter) / union if union else 0.0
            if iou >= iou_thr:
                cand.append((iou, pi, gi))
    cand.sort(reverse=True)
    up, ug, out = set(), set(), {}
    for iou, pi, gi in cand:
        if pi in up or gi in ug:
            continue
        up.add(pi); ug.add(gi); out[gi] = (pi, iou)
    return out


def arm(barrier, glab, gids, gstat):
    """塗りつぶしツール: 障壁の補集合の4連結成分 >=500px を候補面にして照合する。"""
    plab, pids, pstat = comps(~barrier, 4)
    return plab, pids, pstat, match_lab(plab, pids, pstat, glab, gids, gstat)


def kc8():
    H = 400
    yy, xx = np.ogrid[:H, :H]
    r2 = (yy - 200) ** 2 + (xx - 200) ** 2
    disk = r2 <= 100 ** 2
    ring = (r2 <= 103 ** 2) & (r2 > 100 ** 2)
    glab, gids, gstat = comps(disk, 8)
    _, _, _, m = arm(ring, glab, gids, gstat)
    a = len(m) == 1 and list(m.values())[0][1] >= 0.95
    print(f'KC8a 閉じたリング: 腕1 照合 {len(m)}/1 IoU '
          f'{list(m.values())[0][1] if m else 0:.4f} (>=0.95)  {"PASS" if a else "FAIL"}')
    gap = ring.copy(); gap[195:205, 290:310] = False
    _, _, _, m2 = arm(gap, glab, gids, gstat)
    b = len(m2) == 0
    print(f'KC8b 10px の切れ目: 腕1 照合 {len(m2)}/1 (==0 漏れる)  {"PASS" if b else "FAIL"}')
    mark = np.zeros((H, H), bool); mark[195:205, 290:310] = True   # ベタ側に置いた印
    _, _, _, m3 = arm(gap | mark, glab, gids, gstat)
    c = len(m3) == 1 and list(m3.values())[0][1] >= 0.95
    print(f'KC8c 印を置く: 腕2 照合 {len(m3)}/1 IoU '
          f'{list(m3.values())[0][1] if m3 else 0:.4f} (>=0.95)  {"PASS" if c else "FAIL"}')
    return a and b and c


def kc11():
    H = 300
    yy, xx = np.ogrid[:H, :H]
    gt = np.zeros((H, H), bool); pr = np.zeros((H, H), bool)
    for (cy, cx, r) in ((60, 60, 30), (60, 200, 25), (200, 60, 40), (210, 210, 20)):
        gt |= ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    for (cy, cx, r) in ((62, 62, 30), (60, 200, 12), (200, 60, 39), (120, 150, 35)):
        pr |= ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    ref = rm_match(pr, gt)
    plab, pids, pstat = comps(pr, 8)
    glab, gids, gstat = comps(gt, 8)
    mine = match_lab(plab, pids, pstat, glab, gids, gstat)
    # **件数の一致は組の一致ではない**(2026-10-25、罠 T101 と同じ形)。
    # region_metrics.match は組を返さないので、**組は手で読んで書いた定数**と照合する。
    # gt1(60,60,r30) ↔ pred1(62,62,r30) / gt3(200,60,r40) ↔ pred4(200,60,r39)。
    # gt2 は pred の r12 が IoU 144/625=0.23 で届かず、かつ min_px=500 で消える。
    # gt4 と pred3(120,150) はどれとも重ならない。
    WANT = {1: (1, 0.8863), 3: (4, 0.9506)}
    pair = {g: (pi, round(iou, 4)) for g, (pi, iou) in mine.items()}
    same_pair = pair == WANT
    same_count = (len(mine) == ref['matched'] and len(pids) == ref['n_pred']
                  and len(gids) == ref['n_gt'])
    good = same_pair and same_count
    print(f'KC11 matcher: 件数 自前 matched {len(mine)} n_pred {len(pids)} n_gt {len(gids)} / '
          f'region_metrics {ref["matched"]} {ref["n_pred"]} {ref["n_gt"]} '
          f'{"一致" if same_count else "不一致"} / '
          f'**組** {pair} 対 記録 {WANT} {"一致" if same_pair else "不一致"}  '
          f'{"PASS" if good else "FAIL"}')
    return good


def topf(anc):
    p = [x for x in anc.split('/') if x]
    return p[0] if p else ''


def gated_layers():
    """L12 が通した 1,981 層を FILL/LINE に割る。選別は形の試験だけ(登録 §2)。"""
    k4 = {(r['page_id'], r['layer_id']): r for r in csv.DictReader(open(KEEP4))}
    fill, line = {}, {}
    for r in csv.DictReader(open(INV, encoding='utf-8-sig')):
        if r['folder'] != 'False' or r['rendered'] != 'True':
            continue
        try:
            if float(r['drawn_ratio']) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        pd = r['preview_dark'].strip()
        if pd == '' or float(pd) < DARK_MIN:
            continue
        kk = k4.get((r['page_id'], r['layer_id']))
        if kk is None or float(kk['cov']) >= COV_MAX:
            continue
        if topf(r['ancestors']) in DROP_TOP:
            continue
        pid = r['page_id']
        w = work_of(pid, r.get('page_path'))
        p = f'{BASE}/{w}/all_layers/{pid}_L{r["layer_id"]}.png'
        d = fill if float(kk['keep4']) >= KEEP4_FILL else line
        d.setdefault((pid, w), []).append((int(r['layer_id']), p))
    return fill, line


def page_fill_solid(layers):
    acc = None
    for lid, path in layers:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        ink = im < 128
        if acc is None:
            acc = np.zeros(ink.shape, bool)
        if not ink.any():
            continue
        ys, xs = np.nonzero(ink)
        y0, y1 = max(0, ys.min() - PAD), min(ink.shape[0], ys.max() + 1 + PAD)
        x0, x1 = max(0, xs.min() - PAD), min(ink.shape[1], xs.max() + 1 + PAD)
        acc[y0:y1, x0:x1] |= fill_mask(ink[y0:y1, x0:x1], radius=4)
    return acc


def kc9(npages=6):
    if not TRUTH or not os.path.isdir(TRUTH):
        print('KC9 回せない: L12_TRUTH が無い。**t4 画像はディスク上のどこにも無い**ので、'
              '環境変数を足すだけでは回らない(2026-10-25)。'
              '以前の走行の定数は results/l12_panel_regions/page_truth.csv と per_panel.csv')
        return False
    fill, _ = gated_layers()
    keys = sorted(fill)
    pick = [k for k in keys if k[1] == 'ako6'][:npages // 2] + \
           [k for k in keys if k[1] == 'ako7'][:npages // 2]
    ok = True
    for pid, w in pick:
        fs = page_fill_solid(fill[(pid, w)])
        t4 = cv2.imread(os.path.join(TRUTH, f'{w}_{pid}_t4.png'), cv2.IMREAD_GRAYSCALE)
        if fs is None or t4 is None:
            print(f'KC9 FAIL 読めない {pid}'); return False
        out = int((fs & ~(t4 > 127)).sum())
        ok &= out == 0
        print(f'KC9 {pid}: FILL_SOLID {int(fs.sum()):,} / truth4 の外 {out} (==0)  '
              f'{"PASS" if out == 0 else "FAIL"}')
    return ok


LEDGER_PX = '/home/sh1/deepl/lineart-fill-region/inbox/line_selection_ledger_priority65_px.csv'


def kc33():
    """KC33 データ経路: **別々に作った器具2つが層単位で一致する**(2026-10-25)。

    L の門は `drawn_ratio<=0` を「墨ゼロ」と数え、抽出側の台帳は `ink_px` を持つ。
    片側は**完全に一致する**: `drawn_ratio<=0` の 4,399 層は全部 `ink_px==0`。
    **逆向きは一致しない。31 層が `ink_px==0` で `drawn_ratio>0`**(最大 0.194)——
    `ink_px` は**墨**(黒)、`drawn_ratio` は**描画**(不透明)で、別の量である。
    31層はどれも `black_among_drawn=0.0`、`drawn_gray_mean` が 144-145 の灰色
    (3D レンダーとトーン)。**どれも L の門は通らない**(`preview_dark` が空)。

    ここを「同じ量」として扱うと、**3D とトーンの層が重み0 になって黙って消える**。
    """
    import csv as _csv
    led = {(r['page_path'], r['layer_id']): r
           for r in _csv.DictReader(open(LEDGER_PX, encoding='utf-8-sig'))}
    inv = list(_csv.DictReader(open(INV, encoding='utf-8-sig')))
    if len(led) != 7362 or len(inv) != 7362:
        print(f'KC33 FAIL: 行数 台帳 {len(led)} INV {len(inv)} (どちらも 7,362 のはず)')
        return False

    def fnum(v):
        try:
            return float((v or '').strip())
        except ValueError:
            return None

    zero_dr, leak, grey = 0, [], []
    for r in inv:
        g = led.get((r['page_path'], r['layer_id']))
        if g is None:
            print(f'KC33 FAIL: 台帳に行が無い {r["page_id"]} L{r["layer_id"]}')
            return False
        ink0 = int(g['ink_px'] or 0) == 0
        dr0 = (fnum(r['drawn_ratio']) or 0) <= 0
        if dr0:
            zero_dr += 1
            if not ink0:
                leak.append((r['page_id'], r['layer_id']))
        elif ink0:
            grey.append((r['page_id'], r['layer_id'], (fnum(r['preview_dark']) or 0) >= 0.5))
    gated = sum(1 for _, _, pd in grey if pd)
    ok = (zero_dr == 4399 and not leak and len(grey) == 31 and gated == 0)
    print(f'KC33: drawn_ratio<=0 {zero_dr} (==4,399) / そのうち ink_px>0 {len(leak)} (==0) / '
          f'ink_px==0 かつ drawn_ratio>0 {len(grey)} (==31、墨0の灰色) / '
          f'そのうち門を通る {gated} (==0)  {"PASS" if ok else "FAIL"}')
    return ok


def kc34(n=30, seed=20261025):
    """KC34 データ経路: **台帳の4量は L 側の層 PNG から、アルファの閾値として再現する**
    (2026-10-25)。

    抽出側の回答: `cov` は層のアルファの面だけ(`clip_render.py` 214行 `tile = planes[0]`)、
    統計は `g = 255 - アルファ`。L の層 PNG はその `g` なので、**同じ閾値を L 側で当てれば
    同じ数になるはず**で、ならなければどちらかの読みが違う。閾値:

        drawn_ratio       アルファ >= 26   <=>  g <= 229
        ink_px            アルファ >= 128  <=>  g <= 127
        black_among_drawn アルファ >= 224  <=>  g <= 31   (描画画素のうちの割合)

    **向きは両方、粒度は層。** 丸めの幅は台帳の桁(dr 6桁 / gray 2桁 / bad 4桁 / ink 整数)。
    `all_layers` に PNG が無い層は飛ばし、**飛ばした数も印字する**(規律7)。
    """
    import random as _random
    import csv as _csv
    led = {(r['page_path'], r['layer_id']): r
           for r in _csv.DictReader(open(LEDGER_PX, encoding='utf-8-sig'))}
    rows = [r for r in _csv.DictReader(open(INV, encoding='utf-8-sig'))
            if r['folder'] == 'False' and r['rendered'] == 'True']
    pick = _random.Random(seed).sample(rows, n)
    agree, off, skip = 0, [], 0
    for r in pick:
        pid, lid = r['page_id'], r['layer_id']
        w = work_of(pid, r.get('page_path'))
        path = f'{BASE}/{w}/all_layers/{pid}_L{lid}.png'
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None
        if g is None:
            skip += 1
            continue
        drawn = g <= 229
        dr = drawn.sum() / g.size
        gm = g[drawn].mean() if drawn.any() else 0.0
        ba = (g[drawn] <= 31).sum() / drawn.sum() if drawn.any() else 0.0
        ink = int((g <= 127).sum())
        L = led[(r['page_path'], lid)]
        d = (abs(dr - float(r['drawn_ratio'] or 0)),
             abs(gm - float(r['drawn_gray_mean'] or 0)),
             abs(ba - float(r['black_among_drawn'] or 0)),
             abs(ink - int(L['ink_px'] or 0)))
        if d[0] <= 5e-7 and d[1] <= 5e-3 and d[2] <= 5e-5 and d[3] == 0:
            agree += 1
        else:
            off.append((pid, lid, d))
    read = agree + len(off)
    ok = read > 0 and not off
    print(f'KC34 データ経路: 層 {n} のうち PNG を読めた {read} / 飛ばした {skip} —— '
          f'**4量すべて層単位で一致 {agree}/{read}**  ずれ {len(off)}  '
          f'{"PASS" if ok else "FAIL"}')
    for pid, lid, d in off[:5]:
        print(f'    ずれ {pid} L{lid}: dr={d[0]:.2e} gray={d[1]:.4f} '
              f'bad={d[2]:.2e} ink={d[3]}')
    return ok


if __name__ == '__main__':
    a, b, c, d, e = kc8(), kc11(), kc9(), kc33(), kc34()
    print('\n既知セル', 'PASS' if (a and b and c and d and e) else 'FAIL')
    sys.exit(0 if (a and b and c and d and e) else 1)
