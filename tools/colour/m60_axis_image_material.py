# -*- coding: utf-8 -*-
"""m60: **`has_image_material` が「資料用の画像」を予測するか**(追補51 §C に事前登録)。

**篩には使わない。** **作者の答を軸で上書きしない**(規律6)。
**結合は `page_path` + `layer_id`。ファイル名では結合しない。**
**対照として他の欄も同じ表で測る** —— 「画像素材だけが効く」には、他が効かないことが要る。
"""
import collections
import csv
import json
import sys
from pathlib import Path

IN = Path('../lineart/inbox/m_request6a_layer_fields_315.csv')
ANS = Path('labels/layer_kind_round1.jsonl')
QUEUE = Path('results/m56_ui/queue.json')
OUT = Path('results/m60_axis'); OUT.mkdir(parents=True, exist_ok=True)
TARGET = '資料用の画像'
FIELDS = ['has_image_material', 'has_text', 'has_gradation', 'has_vector',
          'has_3d', 'has_filter', 'has_paper_fill', 'has_mask']


def main():
    fields = {}
    for r in csv.DictReader(open(IN, encoding='utf-8-sig')):
        fields[(r['page_path'], str(r['layer_id']))] = r
    q = {it['order']: it for it in json.load(open(QUEUE))['queue']}
    last = {}
    for l in open(ANS, encoding='utf-8'):
        if l.strip():
            r = json.loads(l); last[r['order']] = r

    rows, drop = [], collections.Counter()
    for o, a in last.items():
        it = q.get(o)
        if it is None:
            drop['行列に無い order'] += 1; continue
        f = fields.get((it['pid'], str(it['lid'])))
        if f is None:
            drop['**依頼6 の表に無い層**'] += 1; continue
        if a['choice'] == 'word':
            kind = a['word']
        elif a['choice'] == 'free':
            kind = a['free'].strip()
        else:
            kind = 'わからない'
        rows.append(dict(order=o, kind=kind, page=it['pid'], layer_id=it['lid'],
                         **{k: (f.get(k) == 'true') for k in FIELDS},
                         layer_type=f.get('layer_type'), name=f.get('name')))

    print('== 母集団の差分(理由別。規律9)==')
    print(f'  作者の答            {len(last)}')
    for k, v in drop.most_common():
        print(f'  {k:28s} -{v}')
    print(f'  **測った層            {len(rows)}**')
    if drop:
        print('  **落ちた層が在る。棒を読む前に理由を見ること**')

    print('\n== 作者の答の内訳(測った層だけ)==')
    kc = collections.Counter(r['kind'] for r in rows)
    for k, v in kc.most_common():
        print(f'  {k:26s} {v:>4}')

    print(f'\n== 混同表: 欄ごと x 作者の答(**「{TARGET}」の升だけを見ない**)==')
    for key in FIELDS:
        on = [r for r in rows if r[key]]
        if not on:
            print(f'\n  -- {key}: **真の層が 0。測れない** --'); continue
        tp = sum(1 for r in on if r['kind'] == TARGET)
        tot_t = sum(1 for r in rows if r['kind'] == TARGET)
        prec = tp / len(on)
        rec = tp / tot_t if tot_t else float('nan')
        print(f'\n  -- {key}: 真 {len(on)} / {len(rows)} --')
        print(f'     適合率({TARGET} | 真) = {tp}/{len(on)} = **{prec:.3f}**')
        print(f'     再現率(真 | {TARGET})   = {tp}/{tot_t} = **{rec:.3f}**')
        c = collections.Counter(r['kind'] for r in on)
        print('     真の層の答の内訳: ' + ' / '.join(
            f'{k} {v}' for k, v in c.most_common()))
        bar = ('**棒: 再現率>=0.8 かつ 適合率>=0.8 -> 聞かずに代えられる**'
               if rec >= 0.8 and prec >= 0.8 else
               '**棒: 再現率>=0.8 / 適合率<0.8 -> 篩にはなるが答にならない**'
               if rec >= 0.8 else
               '**棒: 再現率<0.8 -> 聞くのを省けない(取りこぼす)**')
        print('     ' + bar)

    print('\n== 対照: `layer_type` ごとの答 ==')
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r['layer_type']][r['kind']] += 1
    for lt in sorted(by):
        tot = sum(by[lt].values())
        t = by[lt][TARGET]
        print(f'  layer_type {lt:>4}: {tot:>4} 層  うち{TARGET} {t:>3} '
              f'({t/tot:.1%})  ' + ' / '.join(f'{k} {v}' for k, v
                                              in by[lt].most_common(3)))

    with open(OUT / 'axis_vs_author.csv', 'w', newline='',
              encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {OUT / "axis_vs_author.csv"}  ({len(rows)} 層。'
          '**この45頁の層であって corpus の標本ではない**)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
