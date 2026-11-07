# -*- coding: utf-8 -*-
"""L38 **層の軸**の問いを、共通部品の形で書き出す(2026-11-07)。

登録: `doc/l38_layer_axis_registration_20261107.md`(**走る前に固定した**)。
対の選びは `tools/evaluation/l38_pick_layer_axis_20261107.py`(`data/l38_layeraxis_20261107/`)。

- **問いの文・単位・選択肢は第1回と逐語で同じ** —— 変えたら帰無値(面だけ 0/17)と比べられない
- **目隠し**: `id` は `f01` …(作品名も群も入れない)。絵は `crops/fNN.png` に**複製**する
  —— 元の名前は `..._L2_neck1_a_face.png` で、**L2 が群の札そのもの**である
- **並びは対の選びの側で種 20261107 で混ぜてある**(`why.json` の `fid` の順)
- 鍵(群の札)は `map.json` だけに置き、**画面・URL・端末の印字に出さない**

    /home/sh1/deepl/lineart/venv/bin/python tools/ui/l38_build_layer_axis_interview_20261107.py
"""
import collections
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
CROPS = ROOT / 'results' / 'l38_neck_crops_20261107' / 'layeraxis20261107'
WHY = ROOT / 'data' / 'l38_layeraxis_20261107' / 'why.json'
OUT = ROOT / 'data' / 'l38_layeraxis_20261107'
NAME = re.compile(r'^(?P<work>[^_]+)_(?P<page>.+?)_c(?P<comp>\d+)_r(?P<region>\d+)'
                  r'_area(?P<area>\d+)_L(?P<nl>\d+)_neck(?P<neck>\d+|None)_a_face\.png$')

# **第1回と逐語で同じ**(`data/l38_meaning_20261107/questions.json` から写した)
QUESTION = '印を付けた面は、1つのものですか、それとも2つ以上が繋がっていますか'
UNIT = '面(ひとつながりのベタ)'
DECIDES = 'この答えで、ベタの面をそのまま1つの単位として扱えるかが決まります'
CHOICES = [
    {'value': 'one', 'label': '1つのもの',
     'help': 'この面ぜんぶで、1つのものを表している'},
    {'value': 'many', 'label': '2つ以上が繋がっている',
     'help': '別のものどうしが繋がって1つの面になっている。何と何かは一言の欄に書けます'},
]


def main():
    rows = json.load(open(WHY, encoding='utf-8'))
    src = {}
    for fn in sorted(CROPS.iterdir()):
        m = NAME.match(fn.name)
        if not m:
            continue
        g = m.groupdict()
        src.setdefault((g['work'], g['page'], int(g['comp']), int(g['region']),
                        int(g['area'])), []).append(fn)
    items, kmap = [], {}
    img = OUT / 'crops'
    if img.exists():
        shutil.rmtree(img)
    img.mkdir(parents=True)
    for r in sorted(rows, key=lambda r: r['fid']):
        k = (r['work'], r['pid'].split('_', 1)[-1] if '_' in r['pid'] else r['pid'],
             r['comp'], r['region'], r['area'])
        hit = [v for kk, v in src.items()
               if kk[0] == r['work'] and r['pid'].endswith(kk[1]) and kk[2:] == k[2:]]
        if len(hit) != 1 or len(hit[0]) != 1:
            raise SystemExit(f'切り出しが1枚に決まらない: {r["fid"]} {k} -> {hit}')
        shutil.copyfile(hit[0][0], img / f'{r["fid"]}.png')
        items.append({'id': r['fid'], 'version': 'crop20261107b',
                      'images': [{'src': f'crops/{r["fid"]}.png', 'target': True,
                                  'caption': '原稿の絵(原寸)。橙の輪郭と枠が問う面'}]})
        kmap[r['fid']] = {kk: r[kk] for kk in
                          ('work', 'pid', 'comp', 'region', 'area', 'group', 'pair',
                           'n_layer', 'neck')}
    spec = {'interview_id': 'l38_layer_axis_20261107', 'question': QUESTION,
            'unit': UNIT, 'decides': DECIDES, 'choices': CHOICES, 'items': items}
    (OUT / 'questions.json').write_text(
        json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT / 'map.json').write_text(
        json.dumps(kmap, ensure_ascii=False, indent=1), encoding='utf-8')

    sys.path.insert(0, str(ROOT / 'tools' / 'ui'))
    from interview_spec import load_questions
    load_questions(OUT / 'questions.json')          # 部品の点検を自分で通してから終わる

    n = collections.Counter(v['group'] for v in kmap.values())
    print(f'問い {len(items)} 件 -> {OUT}/questions.json(部品の点検を通った)')
    for k, v in n.most_common():
        print(f'  {v:>3}  群 {k}')
    print(f'  鍵は {OUT}/map.json(**画面には出さない**)/ 絵は {len(items)} 枚を複製')
    print('  **問いの文は第1回と逐語で同じ**(帰無値と比べるため)。'
          '**首の窓は出していない**(誘導になる)')


if __name__ == '__main__':
    main()
