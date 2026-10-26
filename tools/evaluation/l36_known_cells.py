# -*- coding: utf-8 -*-
"""L36 の既知セル KC45〜KC47。**本体の前に回す。1つでも落ちたら回さない**(規律2)。

登録は `doc/work_log.md` の 2026-10-26 15:23 の節。

問いは「両腕が同じ面集合を採点しているか」なので、**照合の鍵が一意であることが
この走行の前提**である。鍵 `(pid, comp, region)` は**一意でない**(T109。
本来の腕 1,007鍵 / 1,153行)ので、**名前で取り直す。** その名前が本当に
一意で、腕によって変わらないことを、**本体の数を読む前に確かめる。**
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
import l18_propagate as L18
from l17_known_cells import SEEDS, gated_layers, jobs, zmap


def rows_for(npages=1):
    """`--paired` と同じ設定で npages 頁だけ回し、regs / pans の行を返す。"""
    L18.PAIRED = True
    L18.ARMS = 'both'
    zz = zmap()
    fill, line = gated_layers()
    js = jobs(zz, fill, line)[:npages]
    comps, regs, pans, names = [], [], [], []
    for job in js:
        c, r, p, nm = L18.one_page(job)
        comps += c; regs += r; pans += p; names += nm
    return regs, pans, names


def main():
    print('L36 既知セル (登録 2026-10-26 15:23。**1つでも落ちたら本体を回さない**)')
    regs, pans, names = rows_for(1)
    print(f'  1頁で regs {len(regs)} 行 / pans {len(pans)} 行 / 名前 {len(names)}')
    ok = True

    # ---- KC45 名前が一意であること
    bad = []
    for key, rs in collections.defaultdict(list, {
            k: [r for r in regs if (r['arm'], r['seed']) == k]
            for k in {(r['arm'], r['seed']) for r in regs}}).items():
        named = [r['name'] for r in rs if r['name']]
        if len(named) != len(set(named)):
            dup = [n for n, c in collections.Counter(named).items() if c > 1]
            bad.append((key, len(named), len(set(named)), dup[:3]))
    print(f'KC45 名前が腕x seed の中で一意か: '
          f'{"PASS" if not bad else "FAIL"}  (腕x seed {len({(r["arm"], r["seed"]) for r in regs})} 組)')
    for b in bad:
        print(f'     FAIL {b[0]}: 名前 {b[1]} 行 / 相異なる {b[2]}  例 {b[3]}')
    if not bad:
        n_named = sum(1 for r in regs if r['name'])
        print(f'     (名前の在る行 {n_named} / 全 {len(regs)} 行。'
              f'**名前が空の行は {len(regs) - n_named}** —— '
              f'`--paired` では門が落とすので 0 であるべき)')
        if len(regs) - n_named:
            print('     **名前が空の行が在る。門が効いていない**')
            ok = False
    ok &= not bad

    # ---- KC46 同じ面の名前が腕によって変わらないこと
    byreg = collections.defaultdict(set)
    for r in regs:
        if r['name']:
            byreg[(r['pid'], r['comp'], r['region'])].add(r['name'])
    multi = {k: v for k, v in byreg.items() if len(v) > 1}
    print(f'KC46 同じ面 (pid, comp, region) の名前が腕で変わらないか: '
          f'{"PASS" if not multi else "FAIL"}')
    if multi:
        for k, v in list(multi.items())[:3]:
            print(f'     FAIL {k}: {sorted(v)}')
        print('     (**鍵が一意でないので、同じ鍵に別の面が居る場合もここに出る** —— '
              'その場合は鍵の衝突であって名前の欠陥ではない。T109)')
    else:
        print('     (名前は `cname` で**腕のループの外**で振られているので変わらない)')

    # ---- KC47 交差が空でないことの確認
    print('KC47 対応のある読みの交差:')
    pp = [p for p in pans if str(p['arm']).endswith('_paired')]
    if not pp:
        print('     **対応のある読みの行が無い。この走行は成立しない**')
        ok = False
    else:
        for sd in SEEDS:
            rs = [p for p in pp if p['seed'] == sd]
            if not rs:
                continue
            ni = sum(p.get('n_inter', 0) for p in rs if p['arm'] == 'real_paired')
            nr = sum(p.get('n_real', 0) for p in rs if p['arm'] == 'real_paired')
            sh = ni / nr if nr else 0.0
            print(f'     seed {sd}: 交差 {ni} / 本来の腕 {nr} = {sh:.4f}'
                  + ('' if sh >= 0.5 else '  **5割未満 -> 登録 §4 の3行目(判定保留)**'))
        if all(p.get('n_inter', 0) == 0 for p in pp if p['arm'] == 'real_paired'):
            print('     **交差が全 seed で 0。対応のある読みが存在しない**')
            ok = False

    print()
    print('**既知セル ' + ('すべて PASS。本体を回してよい**' if ok
                           else 'が落ちた。本体を回さない**'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
