# -*- coding: utf-8 -*-
"""L38 **層の軸**の対を選ぶ(2026-11-07)。登録: `doc/l38_layer_axis_registration_20261107.md`。

**洗った面プール**(床 6,000。`l38_units_20261107`)の上で:

- **群T** = **層 >= 2** の面を**全部**(32面。層化抽出の層が小さいので抜かない)
- **群C** = 1対1の対照。**試験する軸だけで違う**(条例6)——
  同じ作品 / 首の有無が同じ / **面積比 <= 1.5** / **層 = 1** / **枠線の帯被覆 < 0.5**
  の中から**面積が最も近いもの**を、重複なしで貪欲に取る(大きい面から先に当てる)
- **対が成立しなかった群Tの面は、聞くが主値の 2x2 には入れない。理由別に印字する**(規律9)
- **既知セル**(規律2): 既に裁定の在る面を**名指しで入れる** —— 器具が壊れていれば外れる

出力: `data/l38_layeraxis_20261107/{faces.json, why.json}`(群の札は `why.json` の側。
`faces.json` は画面に出す順で、札を持たない)

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_pick_layer_axis_20261107.py
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l38_units_20261107 as U                                    # noqa: E402

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
CAUSE = ROOT / 'results' / 'l38_merge_cause_20261106.json'
FRAME = ROOT / 'results' / 'l38_frame_axis_20261107.json'
OUT = ROOT / 'data' / 'l38_layeraxis_20261107'
SEED = 20261107
RATIO = 1.5
BAND = 0.5
# 既に裁定の在る面(面積で名指し。`comp` は鍵にならない)。**答を先に書く** —— 規律2
KNOWN = {2232129: ('many', '髪と衣類'), 2503898: ('one', '背景のベタのみ'),
         1293677: ('many', '髪とコマ枠線')}


def key(r):
    return (r['work'], r['pid'], r['comp'], r['region'], r['area'])


def main():
    cause = json.load(open(CAUSE, encoding='utf-8'))
    cov = {key(r): r['band_cov'] for r in json.load(open(FRAME, encoding='utf-8'))}
    assert len(cov) == len(cause), f'素性の結合が1対1でない {len(cov)} != {len(cause)}'
    face, frag = U.split(cause)
    for r in face:
        r['cov'] = cov[key(r)]
    print(f'母集団: 面 {len(face):,} / 小片 {len(frag):,}(床 {U.FRAGMENT_FLOOR:,}、'
          f'登録 {U.FLOOR_REGISTERED})')

    T = sorted([r for r in face if r['n_layer'] >= 2], key=lambda r: -r['area'])
    pool = [r for r in face if r['n_layer'] < 2 and r['cov'] < BAND]
    drop = {'層=1 だが枠線の帯を 0.5 以上覆う(別の軸なので対照にしない)':
            sum(1 for r in face if r['n_layer'] < 2 and r['cov'] >= BAND)}
    print(f'群T(層 >= 2): {len(T)}   対照の候補: {len(pool)}')
    for k, v in drop.items():
        print(f'  候補から外した: {v} —— {k}')

    used, pairs, unpaired = set(), [], []
    for r in T:
        best = None
        for i, s in enumerate(pool):
            if i in used or s['work'] != r['work']:
                continue
            if (s['neck'] is None) != (r['neck'] is None):
                continue
            q = max(s['area'], r['area']) / min(s['area'], r['area'])
            if best is None or q < best[0]:
                best = (q, i)
        if best and best[0] <= RATIO:
            used.add(best[1])
            pairs.append({'t': r, 'c': pool[best[1]], 'ratio': best[0]})
        else:
            unpaired.append({'r': r, 'nearest': None if not best else round(best[0], 2)})

    print(f'\n対が成立: **{len(pairs)}/{len(T)}**(面積比 <= {RATIO}、同じ作品・首の有無が同じ)')
    for p in pairs:
        print(f"  {p['t']['work']} 層{p['t']['n_layer']} {p['t']['area']:>9,} <-> "
              f"層1 {p['c']['area']:>9,}  比 {p['ratio']:.2f}  "
              f"首 {'有' if p['t']['neck'] is not None else '無'}")
    print(f'\n**対に入れなかった群Tの面 {len(unpaired)}**(聞くが 2x2 には入れない。規律9):')
    for u in unpaired:
        print(f"  {u['r']['work']} {u['r']['area']:>9,} —— 最も近い候補でも比 {u['nearest']}")

    rows = []
    for i, p in enumerate(pairs):
        rows.append(dict(p['t'], group='T', pair=i))
        rows.append(dict(p['c'], group='C', pair=i))
    for u in unpaired:
        rows.append(dict(u['r'], group='T_unpaired', pair=None))

    have = {r['area'] for r in rows}
    for a, (ans, what) in KNOWN.items():
        if a in have:
            print(f'  既知セル {a:,} は既に入っている(答: {ans} / {what})')
            continue
        hit = [r for r in face if r['area'] == a]
        if not hit:
            print(f'  **既知セル {a:,} が面プールに無い**(床の下か、別の母集団)')
            continue
        rows.append(dict(hit[0], group='known', pair=None))
        print(f'  既知セルを足した {a:,}(答: {ans} / {what})')

    seen = set()
    uniq = []
    for r in rows:
        if key(r) in seen:
            print(f"  同じ面が2度: {r['area']:,} —— 後の方を落とす")
            continue
        seen.add(key(r))
        uniq.append(r)
    rows = uniq

    rnd = random.Random(SEED)
    rnd.shuffle(rows)                                  # 画面の順は群と無関係に(目隠し)
    OUT.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows, 1):
        r['fid'] = f'f{i:02d}'
    json.dump(rows, open(OUT / 'why.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump([{k: v for k, v in r.items() if k not in ('group', 'pair')} for r in rows],
              open(OUT / 'faces.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    n = {g: sum(1 for r in rows if r['group'] == g) for g in
         ('T', 'C', 'T_unpaired', 'known')}
    print(f'\n**問いの数 {len(rows)}** —— 群T {n["T"]} / 群C {n["C"]} / '
          f'対なし {n["T_unpaired"]} / 既知セル {n["known"]}')
    print(f'書いた: {OUT}/faces.json(札なし) と why.json(札つき。画面に出さない)')


if __name__ == '__main__':
    main()
