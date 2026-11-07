# -*- coding: utf-8 -*-
"""L38 **軸4: コマの枠線を含む面**(2026-11-07)。

作者の裁定(9面)で **1層・支配率 1.0000 の面が「2つ以上」**と出た。語は **「髪とコマ枠線」** ——
**層をまたいだのではなく、コマの枠線とベタが繋がっている**。
**層の軸では説明できない原因**なので、別の軸として計算で引く。

## 述語(**答を見る前に決める**)

コマ成分 `P` の**縁の帯** `band = P & ~erode(P, 6)` を取り、
**面がその帯をどれだけ覆っているか** `cov = |R & band| / |band|` を測る。
**床は 0.5**(= 縁の帯の半分以上。「枠を一周している」の素直な形)。
**床を答の後に動かさない** —— 動かすなら前後の数を両方印字する(規律9・条例1)。

## 既知セル(規律2・規律13)

**裁定の9面でこの軸が作者の分類を再現するかを印字する** ——
**「髪とコマ枠線」の面で発火し、他で発火しない**のが期待。
**的中率と取りこぼしを必ず出す**(近い量で代用しない)。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_frame_axis_20261107.py [頁数]
"""
import collections
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
from l17_known_cells import arms, jobs, zmap                     # noqa: E402
from l38_merge_cause_20261106 import (drop_wip, layer_masks,     # noqa: E402
                                      panels)

ROOT = '/home/sh1/deepl/lineart-fill-region'
SAVE = f'{ROOT}/labels/face_meaning_count.jsonl'
OUT = f'{ROOT}/results/l38_frame_axis_20261107.json'
BAND_R = 6        # 縁の帯の厚み(画素)
FLOOR = 0.5       # **答を見る前に決めた床**


def band_of(P):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND_R + 1, 2 * BAND_R + 1))
    return P & ~cv2.erode(P.astype(np.uint8), k).astype(bool)


def rulings():
    cur = {}
    for ln in open(SAVE, encoding='utf-8'):
        if not ln.strip():
            continue
        r = json.loads(ln)
        k = (r['work'], r['pid'], r['comp'], r['region'], r['area'])
        cur.pop(k, None) if r.get('kind') == 'undo' else cur.__setitem__(k, r)
    return cur


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print('L38 **軸4: コマの枠線を含む面**(診断。棒は登録していない)', flush=True)
    zz = zmap()
    fill, line = arms('gated')
    cut = collections.Counter()
    js = drop_wip(jobs(zz, fill, line), cut)
    print(f'母集団: **現行の素 {len(js)} 頁**  帯の厚み {BAND_R}px / 床 {FLOOR}')
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
        for pkey, ci, (y0, y1, x0, x1), P, glab, gids in panels(job, lm, shape, cut):
            bd = band_of(P)
            nb = int(bd.sum())
            if nb == 0:
                cut['縁の帯が空のコマ'] += 1
                continue
            for gi in gids:
                R = glab == gi
                rows.append(dict(work=job[1], pid=job[0], comp=int(ci), region=int(gi),
                                 frame=pkey[0], area=int(R.sum()),
                                 band=nb, band_hit=int((R & bd).sum())))
        if i % 5 == 0 or i == len(js):
            print(f'  {i}/{len(js)} 頁  面 {len(rows)}', flush=True)
    for r in rows:
        r['band_cov'] = r['band_hit'] / r['band']
    json.dump(rows, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)

    n = len(rows)
    fire = [r for r in rows if r['band_cov'] >= FLOOR]
    print(f'\n=== 面 {n} / **床 {FLOOR} で発火 {len(fire)}({len(fire) / n:.1%})** ===')
    for k, v in cut.most_common():
        print(f'  外した {v}  {k}')
    q = sorted(r['band_cov'] for r in rows)
    print('  覆いの分位: ' + ' / '.join(
        f'{p}% {q[int(p / 100 * (n - 1))]:.3f}' for p in (50, 75, 90, 95, 99)))
    pg = len({(r['work'], r['pid']) for r in fire})
    print(f'  発火した面が在る頁 {pg} / {len({(r["work"], r["pid"]) for r in rows})}')

    # ---- 既知セル: 裁定の9面でこの軸が何を言うか(規律2・規律13)
    rl = rulings()
    idx = {(r['work'], r['pid'], r['comp'], r['region'], r['area']): r for r in rows}
    print('\n--- 裁定の在る面でこの軸が何を言うか(**的中率と取りこぼしを印字する**)---')
    hit = collections.Counter()
    for k, a in sorted(rl.items(), key=lambda kv: -kv[0][4]):
        r = idx.get(k)
        if r is None:
            hit['素性に当たらない'] += 1
            print(f'  **当たらない** {k}')
            continue
        f = r['band_cov'] >= FLOOR
        want = ('コマ枠' in a.get('word', '') or '枠線' in a.get('word', ''))
        hit[('的中' if f == want else '外し')] += 1
        print(f"  覆い {r['band_cov']:.3f} {'**発火**' if f else '  静か'}  "
              f"裁定 {a['label']:5} 「{a['word'] or '—'}」"
              + ('  <- **枠線と名指しされた面**' if want else ''))
    print('  ' + ' / '.join(f'{k} {v}' for k, v in hit.most_common()))
    print(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
