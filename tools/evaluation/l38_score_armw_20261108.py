# -*- coding: utf-8 -*-
"""L38 **腕W を作者の区分けに当てて採点する**(2026-11-08)。

定義・`r` の引き方・`t` は**区分けを集める前に固定した**
(`doc/l38_decompose_registration_20261108.md` §6)。`r` は 追補3 で**線の層から引いた = 2**。
走る前の予測は 追補3 の **P2 / P3**。

- **母集団は作者が「分解が要る」と答えた 10 面だけ** —— **感度しか出ない。精度は出ない**(規律7)
- **棒は登録していない。これは記述である**
- 落としたものを理由別に印字する(規律9)

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_score_armw_20261108.py
"""
import collections
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
D = ROOT / 'data' / 'l38_decompose_20261108'
CROPS = ROOT / 'results' / 'decompose20261108'
R = 2                 # **線の層から引いた**(追補3)。塗りの面からは引いていない
T = 0.5               # **登録 §6 で先に決めた**
FLOOR = 6000          # `PIECE_FLOOR`(借りた定数であると `l38_units_20261107.py` に書いてある)


def armw(mask):
    """面のマスクに腕W を当てる。**返すのは数で、裁定ではない**。"""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    core = (d >= R) & mask
    n, lab, st, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= FLOOR]
    area = int(mask.sum())
    thin_raw = 1.0 - core.sum() / area                       # **主値(字面どおり。床の前)**
    kept = np.isin(lab, big) if big else np.zeros_like(lab, bool)
    thin_floor = 1.0 - kept.sum() / area                     # 床で間引いた版(併記)
    return dict(n_core=len(big), thin=float(thin_raw), thin_floor=float(thin_floor),
                lab=lab, big=big, st=st, area=area)


def main():
    raw = [json.loads(x) for x in open(D / 'l38_decompose_rulings.jsonl', encoding='utf-8')
           if x.strip()]
    last = {}
    for r in raw:
        last[r['item_id']] = r
    wins = {w['fid']: w for w in json.load(open(CROPS / 'windows.json', encoding='utf-8'))}
    labels = collections.defaultdict(list)
    for p in sorted((D / 'labels').glob('*.png')):
        labels[p.name.split('_')[0]].append(p)

    print(f'腕W の採点  r = {R}(**線の層から引いた**)/ t = {T} / 床 {FLOOR:,}')
    print(f'母集団: **作者が「分解が要る」と答えた {len(last)} 面だけ**。'
          f'「要らない」側は1つも無いので**精度は出ない**(規律7)')
    print('**棒は登録していない。記述である**\n')

    drop = collections.Counter()
    rows = []
    for fid, r in sorted(last.items()):
        mp = CROPS / f'{fid}_mask.png'
        if not mp.is_file():
            drop['単位のマスクが無い'] += 1
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
        if int(m.sum()) != wins[fid]['area']:
            drop['マスクの画素数が面積と合わない'] += 1
            continue
        lp = labels.get(fid)
        if not lp:
            drop['作者の札の画像が無い'] += 1
            continue
        img = cv2.imread(str(lp[-1]), cv2.IMREAD_UNCHANGED)   # **最後の保存が有効**(規律7)
        if img is None or img.shape[:2] != m.shape:
            drop['札の画像の大きさがマスクと合わない'] += 1
            continue
        al = img[:, :, 2].astype(np.int32)                    # R = 部品の番号
        w = armw(m)
        # 作者の部品(床の上/下)
        parts = r.get('parts', [])
        big_parts = [q for q in parts if q.get('area', 0) >= FLOOR]
        nm = {q['n']: (q.get('name') or '(名前なし)') for q in parts}
        # 核ごとに、**一番重なる作者の部品**を出す(IoU)
        hits = []
        for i in w['big']:
            cm = w['lab'] == i
            best, bi = 0.0, None
            for q in big_parts:
                pm = (al == q['n']) & m
                inter = int((cm & pm).sum())
                if inter == 0:
                    continue
                iou = inter / int((cm | pm).sum())
                if iou > best:
                    best, bi = iou, q['n']
            hits.append((int(cm.sum()), bi, best))
        says = (w['n_core'] >= 2) or (w['thin'] >= T)
        rows.append(dict(fid=fid, n_core=w['n_core'], thin=w['thin'],
                         thin_floor=w['thin_floor'], says=says, area=w['area'],
                         n_parts=len(parts), n_big=len(big_parts),
                         names=[nm[q['n']] for q in big_parts], hits=hits,
                         band=wins[fid].get('band_cov')))

    print('--- 面ごと(**作者の裁定は全部「分解が要る」**)---')
    print(f'{"面":<5}{"面積":>10}  {"核":>3} {"細い残り":>8} {"(床後)":>7}  '
          f'{"腕W":<6} 作者の部品(床の上/全部)')
    for r in rows:
        print(f'{r["fid"]:<5}{r["area"]:>10,}  {r["n_core"]:>3} {r["thin"]:>8.3f} '
              f'{r["thin_floor"]:>7.3f}  {"要る" if r["says"] else "**要らない**":<6} '
              f'{r["n_big"]}/{r["n_parts"]}  ' + ' / '.join(r['names']))

    ok = [r for r in rows if r['says']]
    print(f'\n**腕W が「要る」と言った面 {len(ok)}/{len(rows)}** '
          f'(= 感度。**精度ではない** —— 「要らない」側の面がこの標本に無い)')
    leg_core = [r for r in rows if r['n_core'] >= 2]
    leg_thin = [r for r in rows if r['thin'] >= T]
    print(f'  脚の内訳: `核の数 >= 2` で発火 {len(leg_core)} / '
          f'`細い残り >= {T}` で発火 {len(leg_thin)}(**P2 の予測は 0**)')
    tf = sorted(r['thin'] for r in rows)
    print(f'  細い残りの分布: 最小 {tf[0]:.3f} / 中央 {statistics.median(tf):.3f} / '
          f'最大 {tf[-1]:.3f}')

    print('\n--- **核の数 と 床の上の部品の数**(P3。合うか)---')
    same = sum(1 for r in rows if r['n_core'] == r['n_big'])
    print(f'  一致 {same}/{len(rows)}  '
          f'(核 < 部品 {sum(1 for r in rows if r["n_core"] < r["n_big"])} / '
          f'核 > 部品 {sum(1 for r in rows if r["n_core"] > r["n_big"])})')
    print('\n--- 核が作者のどの部品に当たったか(重なりの比 IoU)---')
    agree = collections.Counter()
    for r in rows:
        for a, bi, iou in sorted(r['hits'], key=lambda t: -t[0]):
            nm = next((q for q in r['names']), None)
            print(f'  {r["fid"]:<5} 核 {a:>9,}  -> 部品 {bi}  IoU {iou:.3f}')
            agree['当たり(IoU>=0.5)' if iou >= 0.5 else '当たらない(IoU<0.5)'] += 1
    print(f'  {dict(agree)}')

    print('\n--- **流れ込みだけの面では発火しないか**(P3 の裏側)---')
    for r in rows:
        only_bleed = all('流れ込' in n or '枠線' not in n for n in r['names'])
        print(f'  {r["fid"]:<5} 核 {r["n_core"]}  床の上の部品 {r["n_big"]}  '
              f'{"枠線が床の上に在る" if any("枠線" in n for n in r["names"]) else "**枠線は床の上に無い**"}'
              f'   {"" if not only_bleed else ""}')
    if drop:
        print('\n落としたもの(理由別。規律9):')
        for k, v in drop.most_common():
            print(f'  {v:>3}  {k}')
    print('\n**この採点が決められないこと**: 腕W の**精度**(「要らない」面が標本に無い)/ '
          '**面プール全体での率**(母集団は 10 面)/ **(a) 制作の欠陥と (b) 読みの分割の切り分け**')


if __name__ == '__main__':
    main()
