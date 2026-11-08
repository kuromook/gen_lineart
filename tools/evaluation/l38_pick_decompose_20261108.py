# -*- coding: utf-8 -*-
"""L38 **分解しないといけない面はどれだけ在るか**の標本を選ぶ(2026-11-08)。

登録: `doc/l38_decompose_registration_20261108.md`(**走る前に固定した**)。

- **一様に 60 面**(種 20261109。**素性で選ばない**。同じ親から最大2片 —— 11-08 の追補1と同じ理由)
- **枠線を飲んだ面(帯の覆い >= 0.5)を全部**(11 面)—— **作者の仮説の枠線側そのもの。
  一様標本と混ぜない**(読むときも畳まない)
- **IC1** 幾何が何も怪しくない大きいベタ(登録の条件で引く。**候補の数を印字する**)
- **IC2** 面積 2,503,898 / **P1** 面積 1,317,982(片)—— 名指しで足す(規律2)
- **11-07 に反転した面(1,293677「髪とコマ枠線」)は IC に使わない** —— 動いた錨は使い回さない。
  **ただし帯の覆いが 1.000 なので「枠線」の層には入る**(層の定義は素性で決まる。IC ではない)
- **抽出の前と後で面積の分位を印字する。群の札は `why.json` だけに置く**(NC4)

出力: `data/l38_decompose_20261108/{faces.json(札なし), why.json(札つき)}`
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
OUT = ROOT / 'data' / 'l38_decompose_20261108'
SEED = 20261109
N_UNIFORM = 60
MAX_PER_PARENT = 2
BAND_BAR = 0.5            # **枠線を飲んだ** の境(11-06 からの軸。この走行で決めた値ではない)
IC2_AREA = 2503898
P1_AREA = 1317982
# IC1 の条件(登録 §4 に文字で書いた。**ここが唯一の実体**)
IC1 = dict(band_cov=0.0, neck=None, thin_max=0.03, pure_min=0.99, rmed_min=40.0)


def key(r):
    """**片は親、元のままの面は自分自身**を「親」とする(上限の勘定のため)。"""
    return tuple(r['parent']) if r['parent'] else ('kept', r['work'], r['pid'],
                                                   r['comp'], r['region'], r['area'])


def q(rows):
    a = sorted(r['area'] for r in rows)
    if not a:
        return '—'
    return (f'最小 {a[0]:,} / 1/4 {a[len(a) // 4]:,} / 中央 {statistics.median(a):,.0f}'
            f' / 3/4 {a[3 * len(a) // 4]:,} / 最大 {a[-1]:,}')


def main():
    pool = json.load(open(POOL, encoding='utf-8'))
    print(f'分離後の面プール {len(pool):,}(割った片 '
          f'{sum(1 for r in pool if r["origin"] == "piece")} / 元のまま '
          f'{sum(1 for r in pool if r["origin"] == "kept")})  床 {U.FRAGMENT_FLOOR:,}')
    print(f'  抽出の前: {q(pool)}')

    rnd = random.Random(SEED)
    shuffled = pool[:]
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
    print(f'  {len(uni)} 面   抽出の後: {q(uni)}')
    print(f'  **上限 {MAX_PER_PARENT} 片/親** で間引いた {cutp}'
          f'(落としたのではなく、この回の標本に入れなかっただけ)')
    dup = {k: v for k, v in collections.Counter(key(r) for r in uni).items() if v >= 2}
    print(f'  **同じ親から2片以上 {len(dup)} 親**(標本は独立でない。読むときに畳まない)')
    for k, v in dup.items():
        print(f'    親の面積 {k[4] if k[0] != "kept" else k[5]:>9,} から {v} 片')
    nb = sum(1 for r in uni if r['band_cov'] >= BAND_BAR)
    nn = sum(1 for r in uni if r['neck'])
    print(f'  一様標本の中身(読むときの小行): 枠線を飲んだ {nb} / 首あり {nn}')

    rows = [dict(r, group='uniform') for r in uni]
    have = {(r['origin'], r['area'], r['piece_lid']) for r in uni}

    band = [r for r in pool if r['band_cov'] >= BAND_BAR]
    add = [r for r in band if (r['origin'], r['area'], r['piece_lid']) not in have]
    print(f'\n**枠線を飲んだ層**(帯の覆い >= {BAND_BAR}): プールに {len(band)} 面。'
          f'うち一様標本に既に在る {len(band) - len(add)} / 足した {len(add)}')
    for r in sorted(add, key=lambda r: -r['band_cov']):
        print(f'    覆い {r["band_cov"]:.3f}  面積 {r["area"]:>9,}  ({r["origin"]})')
    rows += [dict(r, group='band') for r in add]
    have |= {(r['origin'], r['area'], r['piece_lid']) for r in add}

    cand = [r for r in pool
            if r['band_cov'] == IC1['band_cov'] and not r['neck']
            and r['thin_frac'] < IC1['thin_max'] and r['pure_black'] > IC1['pure_min']
            and r['r_med'] > IC1['rmed_min']]
    print(f'\n  **IC1 の候補 {len(cand)} 面**(登録 §4 の条件)。'
          f'**その中で面積が最大のものを採る**(再現できるように)')
    for nm, want, extra in (('IC1', max(cand, key=lambda r: r['area']) if cand else None,
                            'このままで1つ'),
                            ('IC2', next((r for r in pool if r['area'] == IC2_AREA), None),
                             'このままで1つ(帯の覆い 0.198。外れても問いの欠陥とは言えない)'),
                            ('P1', next((r for r in pool if r['area'] == P1_AREA
                                         and r['origin'] == 'piece'), None),
                             '分解が要る(**予測。点検ではない**)')):
        if want is None:
            print(f'  **{nm} がプールに無い。足せない**(条例4: 黙って飛ばさない)')
            continue
        k = (want['origin'], want['area'], want['piece_lid'])
        if k in have:
            print(f'  {nm} 面積 {want["area"]:,} は**既に標本に在る**(足さない)。期待 {extra}')
        else:
            rows.append(dict(want, group=f'known_{nm.lower()}'))
            have.add(k)
            print(f'  {nm} を足した 面積 {want["area"]:,}。期待 {extra}')

    rnd.shuffle(rows)                                   # 画面の順は群と無関係に(目隠し)
    OUT.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows, 1):
        r['fid'] = f'f{i:02d}'
    json.dump(rows, open(OUT / 'why.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    plain = [{k: v for k, v in r.items() if k != 'group'} for r in rows]
    json.dump(plain, open(OUT / 'faces.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    leak = sum(1 for r in plain for v in r.values()
               if isinstance(v, str) and v in ('uniform', 'band')
               or isinstance(v, str) and v.startswith('known_'))
    n = collections.Counter(r['group'] for r in rows)
    print(f'\n**問いの数 {len(rows)}** —— ' + ' / '.join(f'{k} {v}' for k, v in n.items()))
    print(f'  **NC4 札の漏れ(札なしの JSON に群の名前) {leak} 回**(0 が正しい。0 でも印字する)')
    print(f'書いた: {OUT}/faces.json(札なし) と why.json(札つき。画面に出さない)')


if __name__ == '__main__':
    main()
