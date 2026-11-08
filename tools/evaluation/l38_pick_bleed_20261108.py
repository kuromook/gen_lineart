# -*- coding: utf-8 -*-
"""L38 **塗りつぶしが流れ込んだ面はどれだけ在るか**の標本を選ぶ(2026-11-08)。

登録: `doc/l38_bleed_registration_20261108.md`(**走る前に固定した**)。

- **11-08 に画面に出した 74 面を母集団から外す**(見た標本は棒を担げない)。**外した数を印字する**
- **一様に 60 面**(種 20261110。**素性で選ばない**。同じ親から最大2片)
- **帯の覆い >= 0.5 の層を 15 面**(狙い撃ち。**一様標本と混ぜない。読むときも畳まない**)
- **IC1 = f09 の面 / IC2 = f45 の面**(= **作者が 11-08 に区分けした面**。
  期待値は**作者の裁定から**来ている。自分の軸からは作っていない —— 前の登録の 追補1)
- **抽出の前と後で面積の分位を印字する。群の札は `why.json` だけ**(NC4)

出力: `data/l38_bleed_20261108/{faces.json(札なし), why.json(札つき)}`

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_pick_bleed_20261108.py
"""
import collections
import json
import os
import random
import statistics
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l38_units_20261107 as U                                      # noqa: E402

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
POOL = ROOT / 'results' / 'l38_faces_separated_20261108.json'
SEEN = ROOT / 'data' / 'l38_decompose_20261108' / 'why.json'        # 11-08 に出した 74 面
OUT = ROOT / 'data' / 'l38_bleed_20261108'
SEED = 20261110
N_UNIFORM, N_BAND = 60, 15
MAX_PER_PARENT = 2
BAND_BAR = 0.5                 # 11-06 からの軸。**この走行で決めた値ではない**
IC = {'f09': '在る', 'f45': '無い'}      # **作者の 11-08 の区分けが期待値**(登録 §4)


def key(r):
    return tuple(r['parent']) if r['parent'] else ('kept', r['work'], r['pid'],
                                                   r['comp'], r['region'], r['area'])


def ident(r):
    return (r['work'], r['pid'], r['comp'], r['region'], r['area'], r['origin'],
            r.get('piece_lid'))


def q(rows):
    a = sorted(r['area'] for r in rows)
    if not a:
        return '—'
    return (f'最小 {a[0]:,} / 1/4 {a[len(a) // 4]:,} / 中央 {statistics.median(a):,.0f}'
            f' / 3/4 {a[3 * len(a) // 4]:,} / 最大 {a[-1]:,}')


def main():
    pool = json.load(open(POOL, encoding='utf-8'))
    seen = json.load(open(SEEN, encoding='utf-8'))
    seen_id = {ident(r) for r in seen}
    byfid = {r['fid']: r for r in seen}
    print(f'分離後の面プール {len(pool):,}  床 {U.FRAGMENT_FLOOR:,}')
    print(f'  抽出の前: {q(pool)}')

    rest = [r for r in pool if ident(r) not in seen_id]
    print(f'  **11-08 に画面に出した {len(seen)} 面を外した** -> 母集団 {len(rest):,}'
          f'(**見た標本は棒を担げない**。捨てたのではなく、この走行の母集団に入れない)')
    print(f'  外した後: {q(rest)}')

    rnd = random.Random(SEED)
    shuffled = rest[:]
    rnd.shuffle(shuffled)
    per, uni, cutp = collections.Counter(), [], 0
    for r in shuffled:
        k = key(r)
        if per[k] >= MAX_PER_PARENT:
            cutp += 1
            continue
        per[k] += 1
        uni.append(r)
        if len(uni) >= N_UNIFORM:
            break
    print(f'\n一様に抽出(種 {SEED}。**素性で選んでいない**)')
    print(f'  **上限 {MAX_PER_PARENT} 片/親** で間引いた {cutp}'
          f'(この回の標本に入れなかっただけ)')
    print(f'  層1 一様 {len(uni)} 面   抽出の後: {q(uni)}')
    nb = sum(1 for r in uni if r['band_cov'] >= BAND_BAR)
    print(f'  うち帯の覆い >= {BAND_BAR} の面 {nb}(**一様標本に何面混ざっているか。'
          f'層2 とは別に数える**)')

    pick = {ident(r) for r in uni}
    band_pool = [r for r in rest if r['band_cov'] >= BAND_BAR and ident(r) not in pick]
    rnd.shuffle(band_pool)
    band = band_pool[:N_BAND]
    print(f'  層2 帯の覆い >= {BAND_BAR}: 母集団に {len(band_pool) + nb} 面 -> **{len(band)} 面を引いた**'
          f'(狙い撃ち。**主値には入れない**)')
    print(f'      抽出の後: {q(band)}')
    for r in sorted(band, key=lambda r: -r['band_cov'])[:5]:
        print(f'        覆い {r["band_cov"]:.3f}  面積 {r["area"]:>9,}  ({r["origin"]})')

    rows = ([dict(r, group='uniform') for r in uni]
            + [dict(r, group='band') for r in band])
    have = {ident(r) for r in rows}
    print(f'\n--- IC(**作者の 11-08 の区分けが期待値。自分の軸からは作っていない**)---')
    for fid, exp in IC.items():
        s = byfid.get(fid)
        if s is None:
            sys.exit(f'**{fid} が 11-08 の標本に無い。止まる**')
        hit = [r for r in pool if ident(r) == ident(s)]
        if not hit:
            sys.exit(f'**{fid} の面がプールに無い。止まる**')
        if ident(s) in have:
            print(f'  {fid} は既に入っている(足さない)')
            continue
        rows.append(dict(hit[0], group=f'ic_{fid}'))
        names = [p.get('name') for p in []]
        print(f'  **IC {fid}** 面積 {s["area"]:>9,}  期待値 **{exp}**'
              f'  (帯の覆い {s["band_cov"]:.3f}。**作者が区分けした面なので見覚えが在りうる** —— 登録 §4)')

    par = collections.Counter(key(r) for r in rows if r['origin'] == 'piece')
    dup = {k: v for k, v in par.items() if v >= 2}
    print(f'\n  **同じ親から2片以上引いた親 {len(dup)}**(標本は独立でない。読むときに畳まない)')

    rnd.shuffle(rows)                       # 画面の順は群と無関係に(目隠し)
    OUT.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows, 1):
        r['fid'] = f'g{i:02d}'              # **11-08 の `fNN` と名前を分ける**(別の走行)
    json.dump(rows, open(OUT / 'why.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump([{k: v for k, v in r.items() if k != 'group'} for r in rows],
              open(OUT / 'faces.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    n = collections.Counter(r['group'] for r in rows)
    leak = sum(1 for r in rows if 'group' in {k for k in r} and False)
    print(f'\n**問いの数 {len(rows)}** —— ' + ' / '.join(f'{k} {v}' for k, v in n.items()))
    print(f'  **NC4 群の札が `faces.json` に漏れた数 {leak}**(0 でも印字する)')
    print(f'書いた: {OUT}/faces.json(札なし) と why.json(札つき。**画面に出さない**)')


if __name__ == '__main__':
    main()
