# -*- coding: utf-8 -*-
"""L38 **分離は効いたか / 問題B の率**の問いを、共通部品の形で書き出す(2026-11-08)。

登録: `doc/l38_problem_b_registration_20261108.md`(**走る前に固定した**)。
標本は `tools/evaluation/l38_pick_problem_b_20261108.py`、
切り出しは `tools/evaluation/l38_piece_crops_20261108.py`(`fNN.png` で出る)。

- **問いの文・単位・選択肢は 11-07 と逐語で同じ** —— 変えたら過去の帰無値と比べられない
- **目隠し**: 画面に出るのは `fNN` と絵だけ。群(片か元のままか)は `map.json` の側
- 切り出しの名前が既に `fNN.png` なので、**名前から群は読めない**

    /home/sh1/deepl/lineart/venv/bin/python tools/ui/l38_build_problem_b_interview_20261108.py
"""
import collections
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
CROPS = ROOT / 'results' / 'l38_piece_crops_20261108'
D = ROOT / 'data' / 'l38_problemb_20261108'

QUESTION = '印を付けた面は、1つのものですか、それとも2つ以上が繋がっていますか'
UNIT = '面(ひとつながりのベタ)'
DECIDES = 'この答えで、ベタの面をそのまま1つの単位として扱えるかが決まります'
CHOICES = [
    {'value': 'one', 'label': '1つのもの',
     'help': 'この面ぜんぶで、1つのものを表している'},
    {'value': 'many', 'label': '2つ以上が繋がっている',
     'help': '別のものどうしが繋がって1つの面になっている。何と何かは一言の欄に書けます'},
]
CAPTION = '原稿の絵(原寸)。橙の輪郭と枠が問う面。**淡い灰は同じコマの別の面**'


def main():
    why = json.load(open(D / 'why.json', encoding='utf-8'))
    img = D / 'crops'
    if img.exists():
        shutil.rmtree(img)
    img.mkdir(parents=True)
    items, kmap, miss = [], {}, []
    for r in sorted(why, key=lambda r: r['fid']):
        src = CROPS / f'{r["fid"]}.png'
        if not src.is_file():
            miss.append(r['fid'])
            continue
        shutil.copyfile(src, img / f'{r["fid"]}.png')
        items.append({'id': r['fid'], 'version': 'crop20261108b',
                      'images': [{'src': f'crops/{r["fid"]}.png', 'target': True,
                                  'caption': CAPTION}]})
        kmap[r['fid']] = {k: r[k] for k in
                          ('work', 'pid', 'comp', 'region', 'area', 'group', 'origin',
                           'piece_lid', 'parent', 'siblings', 'neck', 'band_cov',
                           'thin_frac', 'pure_black')}
    if miss:
        sys.exit(f'**切り出しが無い単位 {len(miss)}: {miss}** —— '
                 f'数が合うまで聞き取りに出さない(登録の NC1/NC2)')
    spec = {'interview_id': 'l38_problem_b_20261108', 'question': QUESTION,
            'unit': UNIT, 'decides': DECIDES, 'choices': CHOICES, 'items': items}
    (D / 'questions.json').write_text(
        json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    (D / 'map.json').write_text(
        json.dumps(kmap, ensure_ascii=False, indent=1), encoding='utf-8')

    sys.path.insert(0, str(ROOT / 'tools' / 'ui'))
    from interview_spec import load_questions
    load_questions(D / 'questions.json')           # 部品の点検を自分で通してから終わる

    n = collections.Counter(v['group'] for v in kmap.values())
    print(f'問い {len(items)} 件 -> {D}/questions.json(部品の点検を通った)')
    for k, v in n.most_common():
        print(f'  {v:>3}  群 {k}')
    print(f'  鍵は {D}/map.json(**画面には出さない**)/ 絵は {len(items)} 枚を複製')
    print('  **問いの文は 11-07 と逐語で同じ**(過去の帰無値と比べるため)')


if __name__ == '__main__':
    main()
