# -*- coding: utf-8 -*-
"""**問題A を分離する**(2026-11-08、作者の指示「**問題Aは先に分離して**」)。

規則は `l38_units_20261107.py` の「問題A の分離」に先に書いた(z で上の層に付ける /
層ごとの排他的な片の8連結成分 / 床 6,000 以上の片が2つ以上できた面だけを置き換える /
片に入らなかった画素は小片プールへ)。

**この器具は何も裁定しない。** 分離した形を出し、**前と後の数を両方、名前を付けて印字する**
(規律10。**過去の結果は撤回しない**)。

出力: `results/l38_faces_separated_20261108.json` —— **分離後の面プール**。
行は `l38_merge_cause_20261106.json` と同じ鍵と素性を持ち、加えて:
  `origin` = 'kept'(元の面のまま)/ 'piece'(割った片)
  `parent` = 元の面の鍵 / `piece_lid` = その片の層 / `n_layer_big` = 床の上の片を持つ層の数

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_separate_problem_a_20261108.py [頁数]
**軸も同じ一度の走りで測り直す**(2026-11-08)—— 旧い軸の数は**旧い面の上の値**なので、
分離後のプールには当てられない。測るのは:
  `band_cov` = コマの縁の帯の覆い(枠線の軸。帯は `P & ~erode(P, 6)`)/
  `r_max` `r_med` `thin_frac`(形)/ `pure_black` `mean_gray`(墨。**その単位自身の層の生の濃淡**)
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l38_units_20261107 as U                                      # noqa: E402
from l14_known_cells import A, comps                                # noqa: E402
from l17_known_cells import arms, jobs, zmap                        # noqa: E402
from l38_merge_cause_20261106 import (drop_wip, layer_masks,        # noqa: E402
                                      neck_radius, panels)
from l38_frame_axis_20261107 import band_of                         # noqa: E402

import cv2                                                          # noqa: E402

THIN = 3                     # `l38_face_is_solid_20261107.py` と同じ(細い画素の半径)

ROOT = '/home/sh1/deepl/lineart-fill-region'
BEFORE = f'{ROOT}/results/l38_merge_cause_20261106.json'
OUT = f'{ROOT}/results/l38_faces_separated_20261108.json'


def pieces_in_face(R, win):
    """面 `R` を**層ごとに割る**。`(床の上の片, 床の下の片)` を返す。

    **規則は `l38_units_20261107.py` の「問題A の分離」。ここが唯一の実体である** ——
    切り出しの器具も聞き取りの器具もこの関数を呼ぶ(**規則の写しを2つ持たない**。条例4)。

    - 層ごとに `面 ∩ その層` を取り、**その層自身の8連結**で数える。**削らない** ——
      削ると下の層の塊が割れて、**問題Aではない割れ目**が生まれる(2026-11-08 に直した)
    - 層のマスクは重なるので **片の面積の和は面の面積を超える**(呼ぶ側が重なりを印字する)
    """
    big, small = [], []
    for lid, sm in win:
        part = sm & R
        if not part.any():
            continue
        lab, ids, st = comps(part, 8, 1)
        for k in ids:
            a = int(st[k, A])
            (big if a >= U.PIECE_FLOOR else small).append(
                (int(lid), a, lab == k) if a >= U.PIECE_FLOOR else (int(lid), a))
    return big, small


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f'L38 **問題A の分離**(規則の登録 {U.SEPARATED_REGISTERED}、'
          f'床 {U.FRAGMENT_FLOOR:,} / 片の床 {U.PIECE_FLOOR:,}(借りた定数))', flush=True)
    zz = zmap()
    fill, line = arms('gated')
    cut = collections.Counter()
    js = drop_wip(jobs(zz, fill, line), cut)
    print(f'母集団: **現行の素 {len(js)} 頁**')
    for k, v in cut.most_common():
        print(f'  外した {v}  {k}')
    cut.clear()
    if limit:
        js = js[:limit]
        print(f'  **煙試験: 先頭 {limit} 頁だけ**', flush=True)

    rows, frags = [], []          # 分離後の面 / 小片(**捨てない**)
    stat = collections.Counter()
    for i, job in enumerate(js, 1):
        pid, work, fz, lz, sel = job
        lm, shape = layer_masks(fz, 'solid', cut)
        if shape is None:
            cut['塗り層が1枚も読めない頁'] += 1
            continue
        zs = {}
        for lid, _ in lm:
            z = zz.get((pid, str(lid)))
            if z is None:
                cut['z が納品の表に無い塗り層(最も下として扱った)'] += 1
            zs[lid] = z
        order = sorted(lm, key=lambda t: (zs[t[0]] if zs[t[0]] is not None else -1),
                       reverse=True)                                 # 上の層が先
        paths = {int(lid): pa for lid, pa in fz}
        cache = {}                      # **頁ごとに捨てる**(層の生 PNG は1枚 10MB 級)

        def gray(lid):
            if lid not in cache:
                if len(cache) >= 6:
                    cache.pop(next(iter(cache)))
                im = cv2.imread(paths.get(lid, ''), cv2.IMREAD_GRAYSCALE)
                cache[lid] = im if (im is not None and im.shape == shape) else None
            return cache[lid]

        for pkey, ci, (y0, y1, x0, x1), P, glab, gids in panels(job, lm, shape, cut):
            win = [(lid, m[y0:y1, x0:x1] & P) for lid, m in order]
            bd = band_of(P)
            nb = int(bd.sum())

            def axes(m, lid):
                """**その単位について**軸を測る(枠線の帯の覆い / 形 / 墨)。
                `lid` はその単位自身の層 —— 墨は**その層の生の濃淡**で読む。"""
                d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)[m]
                g = gray(lid)
                pure = mg = None
                if g is None:
                    cut['層の生 PNG が読めない(墨の側だけ欠ける)'] += 1
                else:
                    v = g[y0:y1, x0:x1][m]
                    v = v[v < 128]
                    if v.size:
                        pure, mg = float((v == 0).mean()), float(v.mean())
                    else:
                        cut['単位の中にその層の墨が無い(半径4 で広がった分)'] += 1
                return dict(band=nb, band_hit=int((m & bd).sum()),
                            band_cov=(int((m & bd).sum()) / nb) if nb else None,
                            r_max=float(d.max()), r_med=float(np.median(d)),
                            thin_frac=float((d < THIN).mean()),
                            pure_black=pure, mean_gray=mg)
            for gi in gids:
                R = glab == gi
                ar = int(R.sum())
                base = dict(work=work, pid=pid, comp=int(ci), region=int(gi),
                            frame=pkey[0], area=ar)
                if ar < U.FRAGMENT_FLOOR:
                    frags.append(dict(base, origin='kept', why='元から床の下'))
                    stat['元から小片'] += 1
                    continue
                pieces, small = pieces_in_face(R, win)
                for lid, a in small:
                    frags.append(dict(base, origin='piece', piece_lid=lid,
                                      area=a, why='片が床の下'))
                if pieces:
                    # **重なりは削らずに数える**(片の面積の和 − 面の面積)
                    stat['重なりの画素(片の和 − 面。**二重に数えた分**)'] += max(
                        0, sum(a for _, a, _ in pieces) - ar)
                cov = np.zeros_like(R)
                for _, _, m in pieces:
                    cov |= m
                left = int((R & ~cov).sum())
                if left:
                    frags.append(dict(base, origin='leftover', area=left,
                                      why='どの片にも入らない画素'))
                    stat['どの片にも入らない画素が在る面'] += 1
                if len(pieces) >= 2:
                    stat['**分離した面**'] += 1
                    # 片が1枚の層だけから出た面 —— `面 ⊆ その層` なら片は1つになるので、
                    # **別の層が床の下の橋で繋いでいた**ことになる(やはり問題A)。数を見せる
                    if len({lid for lid, _, _ in pieces}) == 1:
                        stat['うち片が1枚の層だけから出た面(床の下の橋で繋がっていた)'] += 1
                    stat['そこから出た片'] += len(pieces)
                    for lid, a, m in pieces:
                        rows.append(dict(base, area=a, origin='piece', piece_lid=lid,
                                         parent=[work, pid, int(ci), int(gi), ar],
                                         siblings=len(pieces), n_layer_big=1,
                                         neck=neck_radius(m), top_lid=lid,
                                         n_comp_4=len(comps(m, 4)[1]), **axes(m, lid)))
                else:
                    stat['元の面のまま(割っても片が1つ以下)'] += 1
                    ov = [(lid, int((s & R).sum())) for lid, s in win]
                    top = max(ov, key=lambda t: t[1])[0] if ov else -1
                    rows.append(dict(base, origin='kept', parent=None,
                                     piece_lid=None, siblings=1,
                                     n_layer_big=len(pieces),
                                     neck=neck_radius(R), top_lid=top,
                                     n_comp_4=len(comps(R, 4)[1]), **axes(R, top)))
        if i % 5 == 0 or i == len(js):
            print(f'  {i}/{len(js)} 頁  分離後の面 {len(rows)}', flush=True)
    for k, v in cut.most_common():
        print(f'  落とした {v}  {k}')
    json.dump(rows, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)

    bad = [r for r in rows if r['n_layer_big'] >= 2]
    print(f'\n  **点検: 分離後に「床の上の片を2つ以上持つ面」は {len(bad)}**'
          f'(片に置き換えたので **0 が正しい**)')
    assert not bad, '分離が効いていない'
    if limit:
        print('\n  **煙試験なので前後の表は出さない**(母集団が違うものを並べると嘘になる)')
        print(f'書いた: {OUT}')
        return

    before = json.load(open(BEFORE, encoding='utf-8'))
    bf, bg = U.split(before)
    print(f'\n=== **前と後(規律10。どちらも印字する。過去の結果は撤回しない)** ===')
    print(f'  面の数:       **前 {len(bf):,}** -> **後 {len(rows):,}**'
          f'(うち割った片 {sum(1 for r in rows if r["origin"] == "piece"):,} / '
          f'元のまま {sum(1 for r in rows if r["origin"] == "kept"):,})')
    ba = sum(r['area'] for r in bf)
    aa = sum(r['area'] for r in rows)
    ovl = stat['重なりの画素(片の和 − 面。**二重に数えた分**)']
    leftover = sum(r['area'] for r in frags if r['origin'] == 'leftover')
    sub = sum(r['area'] for r in frags if r['origin'] == 'piece')
    print(f'  片の面積の和: **前 {ba:,}** -> **後 {aa:,}**({aa / ba:.2%})'
          f' —— **面の面積を超えるのが正しい**(層が重なるので、削らずに数えている)')
    print(f'    **二重に数えた分 {ovl:,}**({ovl / ba:.2%})。'
          f'小片プールへ: 床の下の片 {sub:,} / どの片にも入らない画素 {leftover:,}'
          f'(**どちらも捨てていない**)')
    print(f'  小片:         **前 {len(bg):,}** -> **後 {len(frags):,} 行**'
          f'(元から床の下 / 片が床の下 / どの層にも付かない画素)')
    for k, v in stat.most_common():
        print(f'    {v:>6}  {k}')
    print(f'  首あり: **前 {sum(1 for r in bf if r["neck"] is not None):,}**'
          f' -> **後 {sum(1 for r in rows if r["neck"] is not None):,}**')
    # **片どもが重なっている面は「同じものを重ねて塗った」疑いが在る** —— 裁定はしない(規律6)
    g = collections.defaultdict(list)
    for r in rows:
        if r['origin'] == 'piece':
            g[tuple(r['parent'])].append(r)
    q = sorted(((sum(x['area'] for x in v) / k[4], k[4], len(v)) for k, v in g.items()),
               reverse=True)
    sus = [t for t in q if t[0] > 1.5]
    print(f'\n--- 片の面積の和 / 親の面積(**1.00 近辺なら重なっていない**)---')
    print(f'  **1.5 を超える面 {len(sus)}/{len(q)}**(片 {sum(t[2] for t in sus)}/'
          f'{sum(t[2] for t in q)})—— **同じものを重ねて塗った疑い。作者に聞く対象**')
    for r0, a, n in sus:
        print(f'    {r0:.2f}  親の面積 {a:>9,}  片 {n}')
    print(f'  残りの {len(q) - len(sus)} 面は {min(t[0] for t in q):.2f}〜'
          f'{max(t[0] for t in q[len(sus):]):.2f}')

    print(f'\n書いた: {OUT}')
    print('**問題B(1枚のレイヤの中の意図的なベタ)はこの後の問いである。** '
          'この器具は裁定をしない(規律6)')


if __name__ == '__main__':
    main()
