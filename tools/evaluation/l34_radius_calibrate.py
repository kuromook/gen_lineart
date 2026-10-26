# -*- coding: utf-8 -*-
"""L34 `fill_mask` の半径を**面の単位の役割で引き直す**(登録 2026-10-26 14:43)。

`FILL_HALF_WIDTH_PX = 4.0` の出所は
`/home/sh1/deepl/lineart-solid-fill/tools/evaluation/fill_mask.py` で、
**Track J の 480x480 のタイルのために引かれている**(自己点検も 480 の尺度:
「半径20の円板はほぼ全部残り、3px の線は全部消える」)。
**L はこれを原寸の層 PNG(画布 7016x6543)にそのまま当てている** ——
**約14倍の尺度差で、一度も引き直していない**(規律7「借りた定数」)。

役割を2つに分けて測る:

  **R1(継承した役割)** 筆の線と塗った面を分ける。
      材料は **納品 beta 496層(作者の側のベタ)** 対
      **`線画` を名前に持ち、納品 beta でなく、`drawn_ratio > 0` の層 1,509層**
      (`drawn_ratio > 0` は 2026-10-26 に**登録が言い落としていた条件**として書き足した。
      棒は動かしていない —— 名前は材質を決めないし、**存在も決めない**)。
      量は `form_keep = |fill_mask(ink, r)| / |ink|`。
      **`keep4` による fill/line の分類は使わない** —— 半径4で定義されていて循環する
  **R2(L31 の (C) の役割)** opening が生の墨の連結を切らない。
      量は L31 と同じ「層x コマのうち opening が成分を増やしたものの割合」

**棒(登録。数を見る前に決めた)**:
  **R1 を満たす最大の r を採る** —— 納品 beta 側の中央 >= 0.95 かつ `線画` 側の中央 <= 0.05。
  **両方を満たす r が無ければ 判定保留**(「この解像度では1つの半径で役割を分けられない」)。
  (C) の数は採った r で出し直す。R2 は採った r で**併記**する(棒には使わない)。
  **(B) は再走しない** —— 厳しい読みは opening を通さないので r に依存しない。

既知セル(規律2。**原寸の尺度で**):
  KC38 半径100の円板は r=32 でもほぼ全部残る (>=0.95)
  KC39 幅20pxの線は r=12 で全部消える (==0)
  KC40 その同じ線が r=8 では残る (>0.5)  ← **片側しか試さない二択を避ける**(T107)
"""
import collections, csv, json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
from fill_mask import MIN_INK_PX, fill_mask
from pid_work import work_of

INV = f'{HERE}/../../inbox/page_layers_priority65.csv'
MAN = ['/home/sh1/deepl/lineart/data/incoming_manuscript/ako6/fill/beta_manifest.json',
       '/home/sh1/deepl/lineart/data/incoming_manuscript/ako7/fill/beta_manifest.json']
BASE = '/home/sh1/deepl/lineart/data/incoming_manuscript'
OUT = '/home/sh1/deepl/lineart-fill-region/results/l34_radius_calibrate'
RADII = (2, 4, 6, 8, 12, 16, 24, 32)
BETA_MIN, LINE_MAX = 0.95, 0.05        # 登録の棒
PAD = 32


# --------------------------------------------------------------- 既知セル
def known_cells():
    ok = True
    d = np.zeros((400, 400), bool)
    yy, xx = np.ogrid[:400, :400]
    d[(yy - 200) ** 2 + (xx - 200) ** 2 <= 100 ** 2] = True
    k = fill_mask(d, radius=32).sum() / d.sum()
    print(f'KC38 半径100の円板 r=32 で残る {k:.4f} (>=0.95) '
          f'{"PASS" if k >= 0.95 else "FAIL"}')
    ok &= k >= 0.95

    ln = np.zeros((400, 400), bool)
    ln[200:220, 40:360] = True          # 幅 20px
    k12 = fill_mask(ln, radius=12).sum() / ln.sum()
    print(f'KC39 幅20pxの線 r=12 で残る {k12:.4f} (==0) '
          f'{"PASS" if k12 == 0 else "FAIL"}')
    ok &= k12 == 0

    k8 = fill_mask(ln, radius=8).sum() / ln.sum()
    print(f'KC40 同じ線が r=8 では残る {k8:.4f} (>0.5) '
          f'{"PASS" if k8 > 0.5 else "FAIL"}')
    ok &= k8 > 0.5
    return bool(ok)


# --------------------------------------------------------------- 材料
def materials():
    """**納品 beta** と **`線画` を名前に持つ納品 beta 以外**。名前は R1 の材料であって
    判定ではない(規律6)—— 較正の材料に作者の納品そのものを置いている。"""
    rows = list(csv.DictReader(open(INV, encoding='utf-8-sig')))
    pids = {r['page_id'] for r in rows}
    beta = set()
    for mp in MAN:
        man = json.load(open(mp, encoding='utf-8'))
        for p in man['pages']:
            if p['page'] in pids:
                for b in p.get('beta') or []:
                    beta.add((p['page'], str(b['layer_id'])))
    jobs, cut = [], collections.Counter()
    for r in rows:
        if r['folder'] != 'False' or r['rendered'] != 'True':
            cut['フォルダー行・未描画'] += 1
            continue
        k = (r['page_id'], r['layer_id'])
        dr = r['drawn_ratio'].strip()
        dr = float(dr) if dr else None
        if k in beta:
            arm = 'beta'
        elif '線画' in r['name']:
            if dr is None or dr <= 0:
                # **登録が言い落としていた条件**(2026-10-26、基盤の訂正)。
                # 層 PNG の発注は**フォルダー以外の全層・`layer_type` を問わず**だが
                # **`drawn_ratio = 0` の層は除く**(抽出側の答)。突き合わせは例外ゼロ:
                # `drawn_ratio>0` 2,963行 PNG 無し0 / `=0` 2,581行 PNG 有り0。
                # **つまり不在は納品の欠陥ではなく、発注の範囲である。**
                # そして**この173行は墨を持たない** —— 抽出側の台帳の `ink_px` が
                # 2,581行すべてで 0(例外0。点検は空でない)。
                cut['腕 line: `drawn_ratio = 0`(発注の範囲外。墨を持たない)'] += 1
                continue
            arm = 'line'
        else:
            cut['どちらの腕でもない'] += 1
            continue
        w = work_of(r['page_id'], r.get('page_path'))
        jobs.append((arm, r['page_id'], r['layer_id'], w,
                     f'{BASE}/{w}/all_layers/{r["page_id"]}_L{r["layer_id"]}.png',
                     r['name']))
    print('材料の選別 (規律9。理由別に)')
    for kk, v in cut.most_common():
        print(f'  落とした {v:5d}  {kk}')
    n = collections.Counter(j[0] for j in jobs)
    print(f'  **腕 beta {n["beta"]} 層 / 腕 line {n["line"]} 層**')
    return jobs


def one_layer(job):
    """1層の `form_keep` を全半径で、かつ R2 の成分の増え方も返す。"""
    arm, pid, lid, w, path, name = job
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return dict(arm=arm, pid=pid, lid=lid, w=w, name=name, drop='PNG が読めない')
    ink = im < 128
    tot = int(ink.sum())
    if tot < MIN_INK_PX:
        return dict(arm=arm, pid=pid, lid=lid, w=w, name=name,
                    drop=f'墨 {tot} < MIN_INK_PX {MIN_INK_PX}')
    ys, xs = np.nonzero(ink)
    y0, y1 = max(0, ys.min() - PAD), min(ink.shape[0], ys.max() + 1 + PAD)
    x0, x1 = max(0, xs.min() - PAD), min(ink.shape[1], xs.max() + 1 + PAD)
    sub = ink[y0:y1, x0:x1]
    raw_n = cv2.connectedComponents(sub.astype(np.uint8), connectivity=8)[0] - 1
    rec = dict(arm=arm, pid=pid, lid=lid, w=w, name=name, drop='',
               ink_px=tot, raw_comp=raw_n)
    for r in RADII:
        fm = fill_mask(sub, radius=float(r))
        rec[f'keep{r}'] = float(fm.sum()) / tot
        rec[f'comp{r}'] = cv2.connectedComponents(fm.astype(np.uint8),
                                                  connectivity=8)[0] - 1
    return rec


def report_drops(recs, good):
    """**落としたものを腕別に印字する。** 理由別の1行では
    `beta 0 / line 173` が見えない —— **落としたものが腕の片側だけの母集団を変える**
    のは T112 の形そのもの(基盤の指摘 2026-10-26)。"""
    drops = collections.Counter(r['drop'] for r in recs if r['drop'])
    print()
    print('選別 (規律9。**理由別 かつ 腕別**)')
    for k, v in drops.most_common():
        byarm = collections.Counter(r['arm'] for r in recs if r['drop'] == k)
        print(f'  落とした {v:5d}  {k}   '
              f'(腕 beta {byarm["beta"]} / 腕 line {byarm["line"]})')
    for arm in ('beta', 'line'):
        tot = sum(1 for r in recs if r['arm'] == arm)
        kept = sum(1 for r in good if r['arm'] == arm)
        if not tot:
            continue
        print(f'  腕 {arm:5s} 登録の母集団 {tot:5d} -> 残った {kept:5d} '
              f'= **{kept / tot:.4f}**'
              + ('' if kept == tot else '  **← この腕の中央はこの上で計算される**'))
    print('  (**腕別に数えるのは、落としたものが腕の片側の母集団だけを変えるから**'
          ' —— T112 の形。2026-10-26 に腕 line だけから173層が落ちていた)')
    print('  (**その173層は在庫表自身の `drawn_ratio = 0` で見分けられる。**'
          ' 最初は「在庫表のどの列でも区別できない」と書いたが、**それは誤り**'
          ' —— 20列目がまさにその列で、いまは腕の定義に入っている)')
    return drops


# **撤回**: 「欠けた173層の `form_keep` を 1 と置く最も不利な読み」は
# 2026-10-26 に基盤が出し、同日に基盤自身が撤回した。**墨の無い層に 1 を置くのは
# 使える修理の中で最も間違っている** —— そして173行は `drawn_ratio = 0` で
# **実際に墨を持たない**(抽出側の台帳の `ink_px` が 0、例外0)。
# **腕の実効母集団は 173 が在っても無くても 1,509 で同じ**なので、
# 最悪の読みは「在りえない事象の下での判定」になる。登録の母集団の定義を
# 直すのが正しい修理で、それは上の `drawn_ratio > 0` である。


def main():
    os.makedirs(OUT, exist_ok=True)
    print('L34 半径の引き直し  (既知セルを先に回す。1つでも落ちたら本体を回さない)')
    if not known_cells():
        print('**既知セルが落ちた。本体を回さない。**')
        sys.exit(1)
    print()
    jobs = materials()
    lim = int(os.environ.get('L34_LIMIT', '0'))
    if lim:
        jobs = jobs[:lim]
        print(f'  smoke: 先頭 {lim} 層だけ')

    recs = []
    with Pool(4) as pool:
        for i, rec in enumerate(pool.imap_unordered(one_layer, jobs), 1):
            recs.append(rec)
            if i % 200 == 0:
                print(f'  {i}/{len(jobs)} 層', flush=True)
    good = [r for r in recs if not r['drop']]
    report_drops(recs, good)

    tag = f'_limit{lim}' if lim else ''
    cols = (['arm', 'pid', 'lid', 'w', 'name', 'ink_px', 'raw_comp']
            + [f'keep{r}' for r in RADII] + [f'comp{r}' for r in RADII])
    with open(f'{OUT}/form_keep{tag}.csv', 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, cols, extrasaction='ignore')
        wr.writeheader()
        for r in good:
            wr.writerow(r)
    print(f'-> {OUT}/form_keep{tag}.csv  行 {len(good)}')

    # ---------------- R1
    print()
    print('=' * 92)
    print('R1 筆の線と塗った面を分ける   `form_keep` の中央 '
          '(**プールを跨いで平均しない**。世代も分ける)')
    med = {}
    for arm in ('beta', 'line'):
        for w in ('ako6', 'ako7', '両'):
            rs = [r for r in good if r['arm'] == arm and (w == '両' or r['w'] == w)]
            if not rs:
                continue
            row = []
            for r in RADII:
                v = float(np.median([x[f'keep{r}'] for x in rs]))
                med[(arm, w, r)] = v
                row.append(f'{v:.4f}')
            print(f'  腕 {arm:5s} {w:4s} 層 {len(rs):4d}   ' +
                  '  '.join(f'r={r}:{s}' for r, s in zip(RADII, row)))
    print()
    print(f'  棒: **beta の中央 >= {BETA_MIN} かつ line の中央 <= {LINE_MAX} '
          f'を満たす最大の r**')
    hits = [r for r in RADII
            if med.get(('beta', '両', r), 0) >= BETA_MIN
            and med.get(('line', '両', r), 1) <= LINE_MAX]
    print('    満たす r: ' + (' '.join(str(r) for r in hits) if hits else '**無し**'))
    pick = max(hits) if hits else None
    if pick is None:
        print('    -> **判定保留。この解像度では1つの半径で役割を分けられない。**')
        print('       **(C) の数は出さない**(登録どおり)')
    else:
        print(f'    -> **採る半径 r = {pick}**  '
              f'(beta 中央 {med[("beta","両",pick)]:.4f} / '
              f'line 中央 {med[("line","両",pick)]:.4f})')
        for w in ('ako6', 'ako7'):
            b, l = med.get(('beta', w, pick)), med.get(('line', w, pick))
            if b is not None and l is not None:
                flag = 'OK' if (b >= BETA_MIN and l <= LINE_MAX) else '**この世代では棒を外す**'
                print(f'       世代 {w}: beta {b:.4f} / line {l:.4f}  {flag}')
        if pick == 4:
            print('       (**借りた 4 がそのまま残った場合でも、'
                  'それは「引き直して同じだった」であって「未較正のまま」ではない**)')

    # ---------------- R2 (併記)
    print()
    print('=' * 92)
    print('R2 (併記。棒には使わない) opening が生の墨より成分を増やした層の割合')
    for w in ('ako6', 'ako7'):
        rs = [r for r in good if r['w'] == w]
        if not rs:
            continue
        row = []
        for r in RADII:
            n = sum(1 for x in rs if x[f'comp{r}'] > x['raw_comp'])
            row.append(f'r={r}:{n / len(rs):.4f}')
        print(f'  {w}  層 {len(rs):4d}   ' + '  '.join(row))
    print('  (L31 の (C) は**層x コマ**単位。ここは**層**単位なので同じ数にはならない。'
          '**役割の比較に使い、(C) の数として引用しない**)')
    if pick is not None:
        print(f'  **採った r={pick} での L31 (C) の出し直しは、'
              f'L31 を r={pick} で再走して出す**((B) は再走しない)')


if __name__ == '__main__':
    main()
