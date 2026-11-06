"""面の不自然な結合は**何に由来するか**を、1軸ずつ切り分ける(診断。**棒は登録していない**)。

作者(2026-11-06): **「一部レイヤが不自然な結合をしているものもあった それについても調べてほしい」**

**「レイヤが」である。** 11-06 の棚卸し(`l38_merge_audit_20261106.py`)で
**面の 96.9% は1つの塗り層の中の連結成分**だと分かっているので、
**層を畳んだこと(OR)は大半を説明しない。** 残る容疑は3つ:

  軸1 **器具の opening** —— 標準は `fill_mask(radius=4)`(= `acc_add(..., 'solid')`)。
       半径4 の閉じ処理が**層の内部の隙間を埋めて**2つの塊を繋ぐ。
       対照は `'cover'`(生の墨の和集合、**半径が器具のどこにも残らない**)。
       **違うのはこの1軸だけ**(条例6)
  軸2 **層自身が繋がっている** —— 1つの層の中で、細い首で繋がった2つの塊。
       面を半径 r で侵食して割れるかで測る。**割れた最小の r が首の半径**
  軸3 **8連結** —— 斜めに1画素触れただけで1つの面になる。4連結で割れるかで測る

**どれが「正しい」とは言わない。** 面をどう割るべきかは作者が決める(規律6)。
この器具は**どの軸がどれだけ効くか**を出すだけである。

母集団は**現行の素 `pageNNNN` のみ**(`done*` = 作業途中の退避は外す。
作者 2026-10-26 / 2026-11-06)。**外した数を印字する**(規律9)。
"""
import collections
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l14_arms import fill_mask                                   # noqa: E402
from l14_known_cells import A, PAD, comps                        # noqa: E402
from l17_known_cells import (DEL, MAX_FRAC, MIN_FRAC, arms,      # noqa: E402
                             jobs, zmap)

ROOT = '/home/sh1/deepl/lineart-fill-region'
OUT = f'{ROOT}/results/l38_merge_cause_20261106.json'
WIP_WORKS = ('ako6', 'ako7')
RADII = (1, 2, 3, 5, 8)          # 侵食の半径(= 首の半径の候補)
MINPART = 500                    # 割れた片として数える最小画素(`comps` の MINREG と同じ)


def drop_wip(js, cut):
    """**`done*` を外す**(作者の裁定)。名前で引くのは `WIP_WORKS` の中だけ。"""
    keep = []
    for job in js:
        pid, work = job[0], job[1]
        if work not in WIP_WORKS:
            cut[f'この規則の外の作品(`{work}`)'] += 1
            keep.append(job)
        elif re.search(r'page\d+$', pid):
            keep.append(job)
        else:
            cut['**`done*` = 作業途中の退避**(作者 2026-10-26 / 11-06)'] += 1
    return keep


def layer_masks(fz, which, cut):
    """層ごとの被覆を頁の大きさで返す。`which` は **試験する1軸**。

    `'solid'` = `fill_mask(radius=4)`(標準)/ `'cover'` = 生の墨(半径なし)。
    **`acc_add()` と同じ窓の取り方**にする(padded bbox の上で閉じる)。
    """
    out, shape = [], None
    for lid, path in fz:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            cut['層 PNG が読めない'] += 1
            continue
        ink = im < 128
        if shape is None:
            shape = ink.shape
        elif ink.shape != shape:
            cut['層の寸法が頁と違う'] += 1
            continue
        if not ink.any():
            cut['墨が1画素も無い層'] += 1
            continue
        m = np.zeros(shape, bool)
        if which == 'cover':
            m |= ink
        else:
            ys, xs = np.nonzero(ink)
            y0, y1 = max(0, ys.min() - PAD), min(shape[0], ys.max() + 1 + PAD)
            x0, x1 = max(0, xs.min() - PAD), min(shape[1], xs.max() + 1 + PAD)
            m[y0:y1, x0:x1] = fill_mask(ink[y0:y1, x0:x1], radius=4)
        out.append((int(lid), m))
    return out, shape


def neck_radius(R):
    """面を半径 r で侵食して**割れた最小の r**(= 首の半径)。割れなければ None。

    `MINPART` 未満の片は数えない —— **1画素の欠けで「割れた」と言わない**。
    """
    for r in RADII:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        e = cv2.erode(R.astype(np.uint8), k)
        n, lab, st, _ = cv2.connectedComponentsWithStats(e, 8)
        if sum(1 for i in range(1, n) if st[i, A] >= MINPART) >= 2:
            return r
    return None


def panels(job, lm, shape, cut, count_cut=True):
    """コマ成分ごとに (P, 面のラベル, 面の id) を返す。`panels_and_regions` と同じ条件。"""
    pid, work, fz, lz, sel = job
    F = np.zeros(shape, bool)
    for _, m in lm:
        F |= m
    H, W = shape
    for fr in sel:
        km = cv2.imread(f'{DEL}/{work}/koma/{fr["file"]}', cv2.IMREAD_GRAYSCALE)
        if km is None or km.shape != (H, W):
            if count_cut:
                cut['コマ枠が読めない/寸法が違う'] += 1
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats((km > 127).astype(np.uint8), 8)
        for ci in range(1, n):
            # **コマの鍵は `(枠ファイル, ci)`。`ci` だけでは一意でない**(KC46)——
            # 2026-11-06、最初の版は `ncov[ci]` に足し込んで**複数のコマの面数を合算**し、
            # 軸1 の数を壊した。**同じ鍵の衝突を同じ日に2度踏んだ**(条例4)
            pkey = (fr['file'], int(ci))
            a = int(st[ci, A])
            if a < MIN_FRAC * H * W or a > MAX_FRAC * H * W:
                if count_cut:
                    cut['コマ成分が小さすぎ/大きすぎ'] += 1
                continue
            y0 = max(0, st[ci, cv2.CC_STAT_TOP] - 2)
            x0 = max(0, st[ci, cv2.CC_STAT_LEFT] - 2)
            y1 = min(H, y0 + st[ci, cv2.CC_STAT_HEIGHT] + 4)
            x1 = min(W, x0 + st[ci, cv2.CC_STAT_WIDTH] + 4)
            P = (lab[y0:y1, x0:x1] == ci)
            glab, gids, gstat = comps(F[y0:y1, x0:x1] & P, 8)
            if not gids:
                if count_cut:
                    cut['面が1つも立たないコマ成分'] += 1
                continue
            yield pkey, ci, (y0, y1, x0, x1), P, glab, gids


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print('L38 **結合の原因**を1軸ずつ(診断。棒は登録していない)。'
          '標準は `arms(\'gated\')`', flush=True)
    zz = zmap()
    fill, line = arms('gated')
    cut = collections.Counter()
    js = drop_wip(jobs(zz, fill, line), cut)
    print(f'\n母集団: **現行の素 {len(js)} 頁**')
    for k, v in cut.most_common():
        print(f'  外した {v}  {k}')
    cut.clear()
    if limit:
        js = js[:limit]
        print(f'  **煙試験: 先頭 {limit} 頁だけ**', flush=True)

    rows = []
    for i, job in enumerate(js, 1):
        lm, shape = layer_masks(job[2], 'solid', cut)
        if shape is None:
            cut['塗り層が1枚も読めない頁'] += 1
            continue
        # 軸1 の対照: **同じ頁を生の墨で組み直す。違うのは半径だけ**
        lmc, _ = layer_masks(job[2], 'cover', collections.Counter())
        ncov = {}
        for pkey, ci, box, P, glab, gids in panels(job, lmc, shape, cut, count_cut=False):
            assert pkey not in ncov, f'コマの鍵が衝突した: {pkey}'
            ncov[pkey] = len(gids)
        for pkey, ci, (y0, y1, x0, x1), P, glab, gids in panels(job, lm, shape, cut):
            cuts = [(lid, m[y0:y1, x0:x1] & P) for lid, m in lm]
            for gi in gids:
                R = glab == gi
                ar = int(R.sum())
                ov = sorted(((lid, int((s & R).sum())) for lid, s in cuts),
                            key=lambda t: -t[1])
                top_lid, top_px = (ov[0] if ov else (-1, 0))
                n4 = len(comps(R, 4)[1])
                rows.append(dict(work=job[1], pid=job[0], comp=int(ci), region=int(gi),
                                 frame=pkey[0], area=ar, top_lid=top_lid,
                                 n_layer=sum(1 for _, v in ov if v >= 0.01 * ar),
                                 n_comp_4=n4, neck=neck_radius(R),
                                 n_face_solid=len(gids),
                                 n_face_cover=ncov.get(pkey, -1)))
        if i % 5 == 0 or i == len(js):
            print(f'  {i}/{len(js)} 頁  面 {len(rows)}', flush=True)

    n = len(rows)
    print(f'\n=== 面 {n}(現行40頁)===')

    print('\n--- 軸1 器具の opening(`fill_mask(radius=4)` 対 生の墨)---')
    # **コマの鍵は `(作品, 頁, 枠ファイル, comp)`。`comp` だけでは一意でない**(上の KC46)
    pan = {}
    for r in rows:
        if r['n_face_cover'] < 0:
            continue
        pan[(r['work'], r['pid'], r['frame'], r['comp'])] = \
            (r['n_face_solid'], r['n_face_cover'])
    ds = collections.Counter()
    for s, c in pan.values():
        ds['生の墨の方が多い(= 半径4 が繋いだ)' if c > s else
           ('同じ' if c == s else '生の墨の方が少ない(= 半径4 が割った/粒が落ちた)')] += 1
    for k, v in ds.most_common():
        print(f'  コマ {v}  {k}')
    # **両方向を別に出す。** 同じ1軸が両向きに効く —— 生の墨にすると
    # (a) 半径4 が埋めていた隙間が開いて**割れる**が、
    # (b) 塊が痩せて `comps` の `MINREG`(500画素)を**下回って消える**。
    # 片側だけ足して「+3」と書くと、**減った分が隠れる**(規律9)。
    up = sum(max(0, c - s) for s, c in pan.values())
    dn = sum(max(0, s - c) for s, c in pan.values())
    ts, tc = sum(s for s, _ in pan.values()), sum(c for _, c in pan.values())
    print(f'  **半径4 をやめると: 面 {ts} -> {tc}**'
          f'(**割れて増えた +{up} / 痩せて MINREG を割って消えた -{dn}**)')
    print('  **この対照は半径だけを変えているが、結果は `MINREG` の門を通って出てくる** ——'
          '増減を1つの数に畳まない(条例6)')

    print('\n--- 軸2 層自身が細い首で繋がっている(侵食で割れる半径)---')
    c = collections.Counter(r['neck'] for r in rows)
    for k in list(RADII) + [None]:
        v = c.get(k, 0)
        lab = f'半径 {k} で割れた' if k is not None else f'半径 {max(RADII)} まで割れない'
        print(f'  {v:5} 面({v/max(n,1):5.1%})  {lab}')
    split = sum(v for k, v in c.items() if k is not None)
    print(f'  **首で繋がっている面 {split} / {n} = {split/max(n,1):.1%}**'
          '(**割るべきとは言わない**。作者の裁定を仰ぐ数)')

    print('\n--- 軸3 8連結(4連結にすると割れるか)---')
    c4 = collections.Counter(min(r['n_comp_4'], 5) for r in rows)
    print('  ' + ' '.join(f'{k}{"+" if k==5 else ""}片={v}' for k, v in sorted(c4.items())))
    s4 = sum(1 for r in rows if r['n_comp_4'] >= 2)
    print(f'  **4連結にすると割れる面 {s4} / {n} = {s4/max(n,1):.1%}**')

    print('\n--- 首の細い面を持つ層(上位10。作者が現物を見るため)---')
    by = collections.Counter()
    for r in rows:
        if r['neck'] is not None and r['neck'] <= 2:
            by[(r['work'], r['top_lid'])] += 1
    for (w, lid), v in by.most_common(10):
        print(f'  {v:4} 面  {w}  layer_id={lid}')
    print(f'  **首が半径2以下の面を持つ層: {len(by)} 層**')

    print('\n=== 落としたもの(理由別。規律9)===')
    for k, v in (cut.most_common() or [('(この点検は空ではない: 上の面を数えている)', 0)]):
        print(f'  {v}  {k}')

    if not limit:
        json.dump(rows, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f'\n書いた: {OUT}  ({n} 面)')


if __name__ == '__main__':
    main()
