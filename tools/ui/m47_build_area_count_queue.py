"""M47: **「この1成分は、あなたにとって何面ですか」を聞く UI の queue** を作る。

**サーバは書かない。** Track L の `l19_fill_label_server.py` を**そのまま流用**する
(L25 が同じ形で流用している実績がある):

```
python ../lineart-fill-region/tools/ui/l19_fill_label_server.py \
    --host 0.0.0.0 --queue results/m47_ui/queue.json \
    --save labels/area_count.jsonl --extra ''
-> http://<host>:8475
```

**なぜ UI か**(基盤、2026-10-26): 作者の回答で **「いくつかの面は一つのマスクで扱う」**
(= 標準は作者の面を束ねている)と **「離れていれば別の面」**(= 作者の面は連結)が確定し、
**成分 <= 作者の面** が導出された。**係数(1成分が何面か)は測るしかなく、
作者の判定以外に経路が無い。** 切り出し2件では「複数」という定性の答までしか出ない。

**設計(走らせる前に登録。条件は基盤の7つ)**:

1. **出す絵**: 原寸1:1 + マゼンタの輪郭。窓(1400×1400)に収まらない成分は
   **(a) 形の把握用の縮小 + (b) 判定用の原寸(くびれ)** の2枚組(`m35` と同じ登録)
2. **答の形**: **1 / 2 / 3 / 4以上**(キー 1〜4)+ **自由入力**(どこで分かれるか)+ **保留**
3. **画面に出さないもの**: **層の名前・こちらの量**(`drawn_ratio` などの数字)。
   **出すと答が名前に汚染される**(L25/L27 の64問が汚染されていない理由)
4. **門(既知セル)**: **作者が既に答えた2件**を混ぜる ——
   `illust201710\page0003` L50(件1b のスーツ)と `skima11\skima12chara` L269(件2 の胴と両脚)。
   **位置は先に記録し、画面には印を出さない**
5. **標本**: **層別**(**名前が `HAIR`/`SKIN` の層からの成分** と **その他**)。
   各層 **14件**(+ 既知セル2件 = **30件**)。**種は 20261026 で固定**
6. **落としたものを理由別に数えて印字**(規律9)
7. **作者の時間**: L の実績(中央 2.0秒/問、34問で5.1分)から、**30問で5分前後**を見込む

**向き(走らせる前に登録)**:

| | 予測 |
|---|---|
| **どの成分も 1以上** | **定義上**(作者の面は連結、連結集合は1成分に収まる)。**0 が出たら器具の誤り** |
| 名前が `HAIR`/`SKIN` の層 | **1 が多い**(作者「髪と肌はそれぞれ1マスクであることが多い」) |
| その他の層 | **2以上が多い** |

**記録する(作者には見せない)**: **その成分の頁が「標準が頁全体を覆う群」かどうか**
(追補34: 8頁中3頁)。
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'colour'))
import m12_score_a0_nulls as m12                         # noqa: E402
import m16_score_corrected_standard as m16               # noqa: E402
import m19_version_resolution as m19                     # noqa: E402
import m27_author_crops as m27                           # noqa: E402
import m35_case1_remake as m35                           # noqa: E402

DELIV = m27.DELIV
OUT = Path('results/m47_ui'); (OUT / 'crops').mkdir(parents=True, exist_ok=True)
SEED = 20261026
PER = 14
WIN = 1400
MAT = re.compile(r'(hair|skin|髪|肌)', re.I)
MINFRAC = 0.0005
KNOWN = [('000_2018_illust201710_page0003', 50),
         ('000_2018_skima11_skima12chara', 269)]
DOMINATED = {'000_2017_illustration_sunshine2_page0002',
             '000_2018_skima11_skima12H', '000_2018_skima11_skima12chara'}
EXPECT = {'bases': 183, 'components': 823, 'areas': 801}


def main():
    rows = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r['page_path']].append(r)
    pages = [json.loads(l) for l in
             open(DELIV / 'colour_layers_pages.jsonl', encoding='utf-8')
             if l.strip()]
    m12.D = DELIV
    tot, drop = collections.Counter(), collections.Counter()
    cand = []
    for p in pages:
        if not p['layers']:
            drop['ベース層が無い頁'] += 1
            continue
        prows = byp[p['source']]
        byid = {str(r['layer_id']): r for r in prows}
        s = int(round(p['preview']['縮小率']))
        pw, ph = p['preview']['size']
        grid = (ph * s, pw * s)
        floor_px = max(1, round(m12.FLOOR * grid[0] * grid[1]))
        comps = []
        (_, _), (_, garea_a), st = m16.measure_standard(
            p, prows, grid, floor_px, out_comps=comps)
        tot['bases'] += st['bases']
        tot['components'] += st['components']
        tot['areas'] += len(garea_a)
        files = {l['layer_id']: l['file'] for l in p['layers']}
        zmap = {c['layer_id']: c['z'] for c in comps}
        page_ids = [i for i, _ in sorted(
            ((l['layer_id'], zmap.get(l['layer_id'], 0)) for l in p['layers']),
            key=lambda t: t[1])]
        for c in comps:
            if c['area'] < MINFRAC * grid[0] * grid[1]:
                drop['成分が小さすぎる(画布の0.05%未満)'] += 1
                continue
            r = byid.get(str(c['layer_id']), {})
            cand.append(dict(c, page=p['page'], source=p['source'], scale=s,
                             grid=grid, files=files, page_ids=page_ids,
                             name=r.get('name', '?') or '?'))
        del comps
        print(f'  {p["page"]:44s} 成分{st["components"]:>4} 面{len(garea_a):>4}',
              flush=True)

    ok = all(tot[k] == EXPECT[k] for k in EXPECT)
    print(f'\n== 既知セル ==')
    for k in ('bases', 'components', 'areas'):
        print(f'  {k:12s} {tot[k]:>5} / 期待 {EXPECT[k]:>5}  '
              f'{"PASS" if tot[k] == EXPECT[k] else "**FAIL**"}')
    if not ok:
        print('**既知セルが合わない。queue を書かない。**')
        return 1

    for c in cand:
        c['stratum'] = '名前が HAIR/SKIN' if MAT.search(c['name']) \
            else 'その他の名前'
    rng = np.random.default_rng(SEED)
    picked = []
    for st_lab in ('名前が HAIR/SKIN', 'その他の名前'):
        pool = sorted([c for c in cand if c['stratum'] == st_lab],
                      key=lambda c: (c['page'], c['layer_id'], -c['area']))
        take = min(PER, len(pool))
        idx = rng.permutation(len(pool))[:take]
        picked += [pool[int(i)] for i in idx]
        print(f'  層 {st_lab:20s} 候補 {len(pool):>4} → 引いた {take}')
    for pg, lid in KNOWN:
        hit = [c for c in cand if c['page'] == pg and c['layer_id'] == lid]
        if hit and not any(c['page'] == pg and c['layer_id'] == lid
                           for c in picked):
            picked.append(max(hit, key=lambda c: c['area']))
            print(f'  既知セルを足した: {pg} L{lid}')
        elif not hit:
            print(f'  **既知セルが候補に無い**: {pg} L{lid}')

    q = []
    for n, c in enumerate(picked, start=1):
        H, W = c['grid']
        y, x, h, w = c['box']
        full = np.zeros((H, W), bool)
        full[y:y + h, x:x + w] = c['mask']
        fits = h <= WIN and w <= WIN
        if fits:
            cy, cx = y + h // 2, x + w // 2
        else:
            wst = m35.find_waist(c, c['mask'].astype(np.uint8))
            cy, cx = (wst[0], wst[1]) if wst else (y + h // 2, x + w // 2)
        y0 = max(0, min(H - WIN, cy - WIN // 2))
        x0 = max(0, min(W - WIN, cx - WIN // 2))
        y1, x1 = min(H, y0 + WIN), min(W, x0 + WIN)
        img = m27.composite(c['page_ids'], c['files'], c['box'], (y0, x0, y1, x1))
        sub = full[y0:y1, x0:x1].astype(np.uint8)
        mk = img.copy()
        cs, _ = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mk, cs, -1, (255, 0, 255), 3)
        base = f'q{n:03d}_{c["page"]}_L{c["layer_id"]}'
        cv2.imwrite(str(OUT / 'crops' / f'{base}_crop.png'), mk)
        pv = cv2.imread(str(DELIV / [l for l in pages
                                     if l['page'] == c['page']][0]['preview']['file']),
                        cv2.IMREAD_COLOR)
        sm = full[::c['scale'], ::c['scale']].astype(np.uint8)
        sh, sw = min(sm.shape[0], pv.shape[0]), min(sm.shape[1], pv.shape[1])
        ctx = pv.copy()
        cs2, _ = cv2.findContours(sm[:sh, :sw], cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ctx, cs2, -1, (255, 0, 255), 2)
        cv2.rectangle(ctx, (x0 // c['scale'], y0 // c['scale']),
                      (x1 // c['scale'], y1 // c['scale']), (0, 180, 255), 3)
        cv2.imwrite(str(OUT / 'crops' / f'{base}_ctx.png'), ctx)
        q.append({'order': n - 1, 'crop': f'{base}_crop.png',
                  'ctx': f'{base}_ctx.png',
                  # **画面に出さない記録**
                  'pid': c['source'], 'lid': c['layer_id'],
                  'why': ('known' if (c['page'], c['layer_id']) in
                          [(a, b) for a, b in KNOWN] else c['stratum']),
                  'work': c['page'], 'fits_window': bool(fits),
                  'page_covered_by_one_area': c['page'] in DOMINATED,
                  'area_px': int(c['area']),  # numpy の型は json に出せない
                  'cands': [], 'word_share': {}, 'top_word': '', 'purity': 0.0,
                  'n_lay': 0, 'n_lay90': 0, 'auto': 0, 'comp': 0,
                  'region': c['layer_id']})
        del full, sub, img, mk, pv, ctx
        print(f'  描いた {n:>3}/{len(picked)}  {base}', flush=True)

    json.dump(dict(n=len(q), n_ask=len(q), n_check=0, n_auto=0, n_all=len(q),
                   words=['1', '2', '3', '4以上'],
                   ask='マゼンタで囲んだ範囲は、何面ですか?',
                   help='原寸 1:1 の切り出し。マゼンタ = こちらが「1つの面」と'
                        '数えた範囲の外周。C を押すと頁のどこかが分かる'
                        '(縮小。オレンジの枠が今見ている範囲)。'
                        '分かれると思う数を押し、どこで分かれるかは T で一言。',
                   others=[['free', 'どこで分かれるか(一言)', 'T'],
                           ['mixed', '保留(後で見直す)', 'M']],
                   queue=q),
              open(OUT / 'queue.json', 'w'), ensure_ascii=False)
    print(f'\n== 落としたもの(理由別)==')
    for k, v in drop.most_common():
        print(f'  {k:34s} {v:>5}')
    print(f'\n== 標本 ==')
    for k, v in collections.Counter(x['why'] for x in q).most_common():
        print(f'  {k:20s} {v:>3} 件')
    print(f'  窓に収まらない成分(2枚組になる) '
          f'{sum(1 for x in q if not x["fits_window"])}')
    print(f'  標準が頁全体を覆う群の成分 '
          f'{sum(1 for x in q if x["page_covered_by_one_area"])}'
          f'(**作者には見せない記録**)')
    print(f'\nwrote {OUT}/queue.json  ({len(q)} 問)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
