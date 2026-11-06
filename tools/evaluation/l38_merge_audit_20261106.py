"""L38 の面が**どれだけ層を跨いで結合しているか**を数える(診断。**棒は登録していない**)。

作者の報告(2026-11-06): **「不自然な結合が多く、レイヤの結合の仕方が誤ってる可能性がある」**。

機構はコードで確定している —— `acc_add()` は層ごとの `fill_mask(radius=4)` を
**`acc |= fm` で頁の1枚に畳む**ので、**層の同定はその OR で捨てられる**。
面は `comps(F & P, 8)` = 畳んだ後の8連結成分なので、
**別の塗り層が触れていれば1つの面になる。** この器具はその量を測るだけで、
標準を変えない(**面は行列と1画素も違わないことを点検する**)。

印字するもの(規律9: 落としたものを理由別に数える):
  - 面ごとの**寄与した塗り層の数**の分布(接触 1px 以上 / 面積の 1% 以上 の2通り)
  - 2層以上の面で、**最も広い層が面のどれだけを占めるか**(支配率)
  - **層を畳まなければ幾つに割れるか**(層ごとの連結成分の和)。**割るべきとは言わない** ——
    作者の裁定を仰ぐための数である(規律6)
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
from l14_arms import fill_mask                                   # noqa: E402
from l14_known_cells import A, PAD, comps                        # noqa: E402
from l17_known_cells import (DEL, MAX_FRAC, MIN_FRAC, arms,      # noqa: E402
                             jobs, zmap)

ROOT = '/home/sh1/deepl/lineart-fill-region'
QUEUE = f'{ROOT}/results/l38_cluster_ui/queue.json'
OUT = f'{ROOT}/results/l38_merge_audit_20261106.json'


def layer_masks(fz, cut):
    """層ごとの `fill_mask` を頁の大きさで返す。**`acc_add('solid')` と同じ作り。**"""
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
        ys, xs = np.nonzero(ink)
        y0, y1 = max(0, ys.min() - PAD), min(ink.shape[0], ys.max() + 1 + PAD)
        x0, x1 = max(0, xs.min() - PAD), min(ink.shape[1], xs.max() + 1 + PAD)
        fm = np.zeros(shape, bool)
        fm[y0:y1, x0:x1] = fill_mask(ink[y0:y1, x0:x1], radius=4)
        out.append((int(lid), fm))
    return out, shape


def one_page(job, cut):
    pid, work, fz, lz, sel = job
    lm, shape = layer_masks(fz, cut)
    if shape is None:
        cut['塗り層が1枚も読めない頁'] += 1
        return []
    F = np.zeros(shape, bool)
    for _, fm in lm:
        F |= fm
    H, W = shape
    # 線層は面の作りに入らない(`panels_and_regions` と同じ)。コマだけ要る
    rows = []
    for fr in sel:
        m = cv2.imread(f'{DEL}/{work}/koma/{fr["file"]}', cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != (H, W):
            cut['コマ枠が読めない/寸法が違う'] += 1
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), 8)
        for ci in range(1, n):
            a = int(stats[ci, A])
            if a < MIN_FRAC * H * W or a > MAX_FRAC * H * W:
                cut['コマ成分が小さすぎ/大きすぎ'] += 1
                continue
            y0 = max(0, stats[ci, cv2.CC_STAT_TOP] - 2)
            x0 = max(0, stats[ci, cv2.CC_STAT_LEFT] - 2)
            y1 = min(H, y0 + stats[ci, cv2.CC_STAT_HEIGHT] + 4)
            x1 = min(W, x0 + stats[ci, cv2.CC_STAT_WIDTH] + 4)
            P = (lab[y0:y1, x0:x1] == ci)
            f = F[y0:y1, x0:x1] & P
            glab, gids, gstat = comps(f, 8)
            if not gids:
                cut['面が1つも立たないコマ成分'] += 1
                continue
            cuts = [(lid, fm[y0:y1, x0:x1] & P) for lid, fm in lm]
            for gi in gids:
                R = glab == gi
                ar = int(R.sum())
                ov = [(lid, int((sub & R).sum())) for lid, sub in cuts]
                ov = [(lid, v) for lid, v in ov if v > 0]
                ov.sort(key=lambda t: -t[1])
                pct = [(lid, v) for lid, v in ov if v >= 0.01 * ar]
                # **層を畳まなければ幾つに割れるか**(層ごとの連結成分の和)
                parts = 0
                for lid, sub in cuts:
                    s = sub & R
                    if s.any():
                        _, pids_, _ = comps(s, 8)
                        parts += len(pids_)
                rows.append(dict(work=work, pid=pid, comp=int(ci), region=int(gi),
                                 area=ar, n_layer=len(ov), n_layer_1pct=len(pct),
                                 top_share=round(ov[0][1] / ar, 4) if ov else 0.0,
                                 split_if_unmerged=parts))
    return rows


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print('L38 **結合の棚卸し**(診断。棒は登録していない)。'
          '標準は行列と同じ `arms(\'gated\')`', flush=True)
    zz = zmap()
    fill, line = arms('gated')
    js = jobs(zz, fill, line)
    if limit:
        js = js[:limit]
        print(f'  **煙試験: 先頭 {limit} 頁だけ**', flush=True)
    print(f'頁 {len(js)}', flush=True)
    cut = collections.Counter()
    rows = []
    for i, job in enumerate(js, 1):
        rows += one_page(job, cut)
        if i % 5 == 0 or i == len(js):
            print(f'  {i}/{len(js)} 頁  面 {len(rows)}', flush=True)

    # ---- 行列と同じ面を見ているかの点検(**器具が何に取り付いているか**。条例6)
    # **鍵で突き合わせてはいけない。** `comp`(= `ci`)は**枠ファイルごとに 1 から振り直される**
    # ので頁の中で一意ではない(行列の 1,526面で **鍵の衝突 165 個・隠れる面 234**)。
    # 2026-11-06、最初の版は `(work,pid,comp,region)` を鍵にした dict で比べ、
    # **後の面が前の面を上書きして「面積が食い違う 105」という偽の指摘を出した** ——
    # **鍵は同定を含意するが、含意は点検ではない**(条例4)。
    # 直した形: **頁ごとに面積の多重集合**で比べる。コマの同定を要らなくする。
    if not limit:
        q = json.load(open(QUEUE, encoding='utf-8'))['queue']
        qa, ra = collections.defaultdict(collections.Counter), \
            collections.defaultdict(collections.Counter)
        for r in q:
            for f in r['faces']:
                qa[(r['work'], r['pid'])][f['area']] += 1
        for r in rows:
            ra[(r['work'], r['pid'])][r['area']] += 1
        miss = sum(sum((c - ra[k]).values()) for k, c in qa.items())
        extra = sum(sum((ra[k] - qa[k]).values()) for k in ra)
        nq = sum(sum(c.values()) for c in qa.values())
        print(f'\n点検(頁ごとの面積の多重集合): 行列の面 {nq} / この走行 {len(rows)}')
        print(f'  **行列に在って走行に無い面 {miss}**(0 でなければ標準が動いている)  '
              f'/ 走行にしか無い面 {extra}'
              '(= 行列が落とした**面が1つのコマ**。行列の `dropped_single` と一致するはず)')

    print('\n=== 面ごとの**寄与した塗り層の数** ===')
    for name, key in (('接触 1画素以上', 'n_layer'), ('面積の 1% 以上', 'n_layer_1pct')):
        c = collections.Counter(r[key] for r in rows)
        tot = sum(c.values())
        multi = sum(v for k, v in c.items() if k >= 2)
        print(f'  {name}: ' + ' '.join(f'{k}層={v}' for k, v in sorted(c.items())))
        print(f'    **2層以上の面 {multi} / {tot} = {multi / max(tot,1):.1%}**')

    multi = [r for r in rows if r['n_layer_1pct'] >= 2]
    if multi:
        sh = sorted(r['top_share'] for r in multi)
        print('\n=== 2層以上(1%基準)の面で、最も広い層が占める割合 ===')
        for q_ in (0.1, 0.25, 0.5, 0.75, 0.9):
            print(f'  {int(q_*100)}% 分位 {sh[int(q_*(len(sh)-1))]:.3f}')
        for th in (0.9, 0.7, 0.5):
            n = sum(1 for v in sh if v < th)
            print(f'  支配率 < {th}: {n} 面({n / len(rows):.1%} of 全面)')

    print('\n=== 層を畳まなければ幾つに割れるか(面あたり)===')
    c = collections.Counter(min(r['split_if_unmerged'], 10) for r in rows)
    print('  ' + ' '.join(f'{k}{"+" if k==10 else ""}={v}' for k, v in sorted(c.items())))
    s = sum(r['split_if_unmerged'] for r in rows)
    print(f'  面 {len(rows)} → 層ごとに割ると {s}(**{s / max(len(rows),1):.2f} 倍**)。'
          '**割るべきとは言わない —— 作者の裁定を仰ぐ数である**(規律6)')

    print('\n=== 落としたもの(理由別。規律9)===')
    if cut:
        for k, v in cut.most_common():
            print(f'  {v}  {k}')
    else:
        print('  0  (この点検は空ではない: 上の面の数を数えている)')

    if not limit:
        json.dump(rows, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f'\n書いた: {OUT}  ({len(rows)} 面)')


if __name__ == '__main__':
    main()
