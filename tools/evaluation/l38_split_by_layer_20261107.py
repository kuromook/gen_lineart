# -*- coding: utf-8 -*-
"""**問題A(意図しないレイヤ間の結合)を層ごとに割ってみる**(2026-11-07、記述。棒は登録していない)。

作者の裁定(2026-11-07、逐語):
> **ベタの面が意図的に1枚のレイヤの絵の中にあるものと複数のレイヤが意図せず結合している
> この2つは別の問題として切り分けて解消しないといつまでも沼にハマる**

**問題A は予測する対象ではない。器具の作りから分かる** —— 面が2つ以上の塗り層にまたがるのは、
塗り層の和を取った**こちらの操作**の結果である。だから測るのは「当たるか」ではなく:

1. **層ごとに割ると、面はいくつの片になるか**(床 6,000 の上で)
2. **割った片に、塗られた面積はどれだけ残るか**(重なりと小片でどれだけ落ちるか)
3. **割っても1片のままの面はどれか**(= 和を取ったことが原因ではない面。問題B の側)

**画素は z(重ね順)で上の層に付ける** —— 層のマスクは重なるので、どちらかに決めないと
片の面積が二重に数えられる。z は `zmap()`(納品の `z_global`)。

**何が正しい割り方かは言わない。** 割ってよいかは作者が決める(規律6)——
この器具は割った形を出すだけで、裁定はこの後の聞き取りで取る。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_split_by_layer_20261107.py [頁数]
"""
import collections
import json
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l38_units_20261107 as U                                   # noqa: E402
from l14_known_cells import A, comps                             # noqa: E402
from l17_known_cells import arms, jobs, zmap                     # noqa: E402
from l38_merge_cause_20261106 import drop_wip, layer_masks, panels  # noqa: E402

OUT = '/home/sh1/deepl/lineart-fill-region/results/l38_split_by_layer_20261107.json'


def report(rows):
    """割った形を印字する。**頁を読み直さずに**(`--report`)同じ表を出せる。"""
    F = U.FRAGMENT_FLOOR
    for r in rows:
        tops = sorted((max(p['parts'] or [0]) for p in r['pieces']), reverse=True)
        r['tops'] = tops
        r['n_big'] = sum(1 for t in tops if t >= F)
        r['big_area'] = sum(t for t in tops if t >= F)
    d = collections.Counter(r['n_big'] for r in rows)
    print('\n--- 割ると何片になるか(**床の上の片だけ数える**)---')
    for k in sorted(d):
        print(f'  {k} 片  {d[k]:>3} 面'
              + ('   <- **割っても1片。和を取ったことが原因ではない**' if k <= 1 else ''))
    ta = sum(r['area'] for r in rows)
    kb = sum(r['big_area'] for r in rows)
    print('\n--- 塗られた面積はどこへ行くか ---')
    print(f'  割る前 {ta:,} -> **床の上の片に {kb:,}({kb / ta:.2%})**'
          f' / 残り {ta - kb:,}({1 - kb / ta:.2%})は小片と重なりの取り合い')
    print(f'  片の数の中央 {statistics.median(r["n_big"] for r in rows)} / '
          f'最大 {max(r["n_big"] for r in rows)}')

    # **旗の誤りを数える。** `n_layer >= 2` は「1% 以上の重なりが在る層」を数えるが、
    # 上の層に先に付けると何も残らない層が在る —— **同じものを2枚のレイヤで塗った面**である。
    one = [r for r in rows if r['n_big'] <= 1]
    dup = [r for r in one if (r['tops'][1] if len(r['tops']) > 1 else 0) < 200]
    small = [r for r in one if r not in dup]
    print(f'\n--- **旗(層 >= 2)の誤り** —— 割れるのは {len(rows) - len(one)}/{len(rows)} 面 ---')
    print(f'  **{len(dup)} 面: 2番目の片が 200 画素未満** —— 1% 以上の重なりが在ると数えた層なのに、'
          f'上の層に先に付けると何も残らない。\n'
          f'      = **同じものを2枚のレイヤに重ねて塗った面**(問題Aではない)')
    for r in dup:
        print(f'      面積 {r["area"]:>9,}  2番目の片 {r["tops"][1] if len(r["tops"]) > 1 else 0:>7,}')
    print(f'  **{len(small)} 面: 2番目の片が床 {F:,} 未満**(1,023〜5,648 画素)—— 小片の側')
    for r in small:
        t = r['tops'][1] if len(r['tops']) > 1 else 0
        print(f'      面積 {r["area"]:>9,}  2番目の片 {t:>7,}({t / r["area"]:.2%}、床の {t / F:.2f} 倍)')
    print('\n**この器具は裁定をしない。** 割ってよいかは作者に聞く(規律6)。'
          '問題B(1枚のレイヤの中の意図的なベタ)はここに入っていない')


def main():
    if '--report' in sys.argv:                   # 頁を読み直さずに同じ表を出す
        return report(json.load(open(OUT, encoding='utf-8')))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print('L38 **層ごとに割る**(記述。棒は登録していない)。'
          f'床 {U.FRAGMENT_FLOOR:,}(登録 {U.FLOOR_REGISTERED})', flush=True)
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

    rows = []
    for i, job in enumerate(js, 1):
        pid, work, fz, lz, sel = job
        lm, shape = layer_masks(fz, 'solid', cut)
        if shape is None:
            cut['塗り層が1枚も読めない頁'] += 1
            continue
        # **z(重ね順)で上から**。無い層は 0(= 最も下)として扱い、数を印字する
        zs = {lid: zz.get((pid, str(lid)), None) for lid, _ in lm}
        for lid, v in zs.items():
            if v is None:
                cut['z が納品の表に無い塗り層(最も下として扱った)'] += 1
        order = sorted(lm, key=lambda t: (zs[t[0]] if zs[t[0]] is not None else -1),
                       reverse=True)                              # 上の層が先
        for pkey, ci, (y0, y1, x0, x1), P, glab, gids in panels(job, lm, shape, cut):
            win = [(lid, m[y0:y1, x0:x1] & P) for lid, m in order]
            for gi in gids:
                R = glab == gi
                ar = int(R.sum())
                taken = np.zeros_like(R)
                pieces, nlay = [], 0
                for lid, s in win:
                    ov = int((s & R).sum())
                    if ov < 0.01 * ar:                            # `n_layer` と同じ閾
                        continue
                    nlay += 1
                    excl = (s & R) & ~taken                       # **上の層に先に付ける**
                    taken |= excl
                    if not excl.any():
                        pieces.append({'lid': int(lid), 'z': zs[lid], 'area': 0,
                                       'parts': []})
                        continue
                    # **min_px=1** —— 片の大きさは床で切るので、器具の既定で先に捨てない(規律9)
                    lab, ids, st = comps(excl, 8, 1)
                    parts = sorted((int(st[k, A]) for k in ids), reverse=True)
                    pieces.append({'lid': int(lid), 'z': zs[lid],
                                   'area': int(excl.sum()), 'parts': parts})
                if nlay < 2:
                    continue
                rows.append(dict(work=work, pid=pid, comp=int(ci), region=int(gi),
                                 area=ar, n_layer=nlay, pieces=pieces,
                                 unassigned=int(ar - taken.sum())))
        if i % 5 == 0 or i == len(js):
            print(f'  {i}/{len(js)} 頁  層をまたぐ面 {len(rows)}', flush=True)
    for k, v in cut.most_common():
        print(f'  落とした {v}  {k}')

    rows = [r for r in rows if r['area'] >= U.FRAGMENT_FLOOR]     # 面だけ(小片は別プール)
    json.dump(rows, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'\n=== **層をまたぐ面 {len(rows)}**(床の上。書いた: {OUT})===')

    report(rows)


if __name__ == '__main__':
    main()
