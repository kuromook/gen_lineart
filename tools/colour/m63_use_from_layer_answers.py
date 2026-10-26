# -*- coding: utf-8 -*-
"""m63: **作者の層ごとの答から `use` を導く**(追補59 §E に事前登録)。

**規則(走る前に登録済み。ここでは実装するだけ)**:
  `仕上がった絵(統合済み)` が1つでも在る            -> 採点する(仕上がった絵が在る)
  `線画` と `色・ベタ` の両方が在る                  -> 採点する(工程が揃っている)
  `下絵・ラフ` だけで上のどちらでもない               -> 採点しない(下絵のみ)
  `資料用の画像` が在り上のどれでもない               -> 採点しない(資料用の画像)
  どれにも当たらない                               -> undecided。作者に聞く

**既知セル(追補59 §F)**:
  `gran page0016`  -> **採点する側に来る**(作者「完成済みカラーイラストの合成レイヤ」)
  `koa` の2頁      -> **`仕上がった絵` の層が1つも無い**(作者が名指しで「全部下絵」)

**結合は `source`(フルパス)。ファイル名では結合しない。**
**飛ばした行は理由別に件数を印字する。**
"""
import collections
import csv
import json
import sys
from pathlib import Path

OUT = Path('results/m63_use_from_answers'); OUT.mkdir(parents=True, exist_ok=True)
FIN = '仕上がった絵(統合済み)'
KNOWN_SCORE = r'F:\manuscripts\2016\gran\gran\page0016.clip'
KNOWN_ROUGH = [r'F:\manuscripts\2017\toho\koa\koakuma.clip',
               r'F:\manuscripts\2017\toho\koa\koakuma_check.clip']


def main():
    q = {it['order']: it for it in
         json.load(open('results/m56_ui/queue.json'))['queue']}
    last, drop = {}, collections.Counter()
    for l in open('labels/layer_kind.jsonl', encoding='utf-8'):
        if l.strip():
            r = json.loads(l); last[r['order']] = r      # 後勝ち = 第2走
    bypage = collections.defaultdict(list)
    for o, a in last.items():
        it = q.get(o)
        if it is None:
            drop['行列に無い order'] += 1; continue
        kind = (a['word'] if a['choice'] == 'word'
                else (a['free'].strip() if a['choice'] == 'free' else 'わからない'))
        bypage[it['pid']].append(kind)

    print('== 母集団の差分(理由別。規律9)==')
    print(f'  作者の答        {len(last)}')
    for k, v in drop.most_common():
        print(f'  {k:24s} -{v}')
    print(f'  **頁            {len(bypage)}**')

    rows = []
    for src, kinds in bypage.items():
        s = set(kinds)
        if FIN in s:
            use, why = '採点する(仕上がった絵が在る)', FIN
        elif '線画' in s and '色・ベタ(塗り)' in s:
            use, why = '採点する(工程が揃っている)', '線画+色・ベタ'
        elif s - {'下絵・ラフ', 'ノンブル', 'トンボ', '印刷用の枠', 'メモ',
                  'セリフ・文字', 'わからない', '添削', 'メモ 添削'} == set():
            use, why = '採点しない(下絵のみ)', '下絵・ラフ以外に絵が無い'
        elif '資料用の画像' in s:
            use, why = '採点しない(資料用の画像)', '資料用の画像'
        else:
            use, why = 'undecided', '規則のどれにも当たらない'
        rows.append(dict(page_path=src, use=use, 根拠=why, 層=len(kinds),
                         種類=' / '.join(f'{k}×{kinds.count(k)}'
                                       for k in sorted(s, key=kinds.count,
                                                       reverse=True))))

    print(f'\n== 導いた `use`({len(rows)} 頁)==')
    for k, v in collections.Counter(r['use'] for r in rows).most_common():
        print(f'  {k:34s} {v:>3}')

    print('\n== 既知セル(追補59 §F)==')
    ok = True
    g = [r for r in rows if r['page_path'] == KNOWN_SCORE]
    if not g:
        print('  **`gran page0016` が答の中に無い。読まない**'); ok = False
    else:
        hit = g[0]['use'].startswith('採点する')
        print(f'  `gran page0016` -> {g[0]["use"]}  '
              f'{"PASS" if hit else "**FAIL**"}  (根拠 {g[0]["根拠"]})')
        ok &= hit
    for p in KNOWN_ROUGH:
        g = [r for r in rows if r['page_path'] == p]
        if not g:
            print(f'  **{p} が答の中に無い**'); ok = False; continue
        hit = FIN not in g[0]['種類']
        print(f'  {p.split(chr(92))[-1]:22s} -> {FIN} の層 '
              f'{"無し" if hit else "**在り**"}  {"PASS" if hit else "**FAIL**"}'
              f'  ({g[0]["use"]})')
        ok &= hit
    print(f'\n  **既知セル {"すべて PASS" if ok else "**FAIL**。他を読まない"}**')

    und = [r for r in rows if r['use'] == 'undecided']
    if und:
        print(f'\n== `undecided`({len(und)} 頁。**作者に聞く対象**)==')
        for r in und:
            print(f'  {r["page_path"].split("manuscripts" + chr(92))[-1]:46s} {r["種類"][:70]}')
    with open(OUT / 'use_from_layer_answers.csv', 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r['page_path']))
    print(f'\nwrote {OUT / "use_from_layer_answers.csv"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
