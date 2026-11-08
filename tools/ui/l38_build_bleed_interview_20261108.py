# -*- coding: utf-8 -*-
"""L38 **塗りつぶしが流れ込んだ面はどれだけ在るか**の問いを共通部品の形で書き出す(2026-11-08)。

登録: `doc/l38_bleed_registration_20261108.md`(**走る前に固定した**)。
標本は `tools/evaluation/l38_pick_bleed_20261108.py`、切り出しは
`tools/evaluation/l38_piece_crops_20261108.py --out-name bleed20261108`(`gNN.png`)。

- **問いは (a) 制作の流れ込みだけを聞く。(b) 読みとしての分割は聞かない**(登録の冒頭)
- **仮説は作者自身のものなので、問いに書くのは誘導ではない。機械の発火は画面に出さない**
- **目隠し**: 画面に出るのは `gNN` と絵だけ。群(一様 / IC)は `why.json` の側(NC4)

    /home/sh1/deepl/lineart/venv/bin/python tools/ui/l38_build_bleed_interview_20261108.py
"""
import collections
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
CROPS = ROOT / 'results' / 'bleed20261108'
D = ROOT / 'data' / 'l38_bleed_20261108'

QUESTION = 'この面に、作るときに塗りつぶしが流れ込んだ所が在りますか'
UNIT = '面(ひとつながりのベタ)'
DECIDES = 'この答えで、流れ込みを機械で見つけに行くか、1つずつ直すかが決まります'
CHOICES = [
    {'value': 'bleed', 'label': '在る',
     'help': '線や枠線に塗りが流れ込んで、この面にくっついている所が在る'},
    {'value': 'clean', 'label': '無い',
     'help': 'この面は、流れ込み無しで、描いたとおりのベタである'},
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
        items.append({'id': r['fid'], 'version': 'crop20261108d',
                      'images': [{'src': f'crops/{r["fid"]}.png', 'target': True,
                                  'caption': CAPTION}]})
        kmap[r['fid']] = {k: r[k] for k in
                          ('work', 'pid', 'comp', 'region', 'area', 'group', 'origin',
                           'piece_lid', 'parent', 'neck', 'band_cov', 'thin_frac',
                           'pure_black')}
    if miss:
        sys.exit(f'**切り出しが無い単位 {len(miss)}: {miss}** —— '
                 f'数が合うまで聞き取りに出さない(登録の NC1/NC2)')
    spec = {'interview_id': 'l38_bleed_20261108', 'question': QUESTION,
            'unit': UNIT, 'decides': DECIDES, 'choices': CHOICES, 'items': items}
    (D / 'questions.json').write_text(
        json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    (D / 'map.json').write_text(
        json.dumps(kmap, ensure_ascii=False, indent=1), encoding='utf-8')

    sys.path.insert(0, str(ROOT / 'tools' / 'ui'))
    from interview_spec import load_questions
    load_questions(D / 'questions.json')          # 部品の点検を自分で通してから終わる

    leak = sum(1 for it in items
               for s in json.dumps(it, ensure_ascii=False).split()
               if 'uniform' in s or 'band' in s or 'ic_' in s)
    print(f'問い {len(items)} 件 -> {D}/questions.json(部品の点検を通った)')
    for k, v in collections.Counter(v['group'] for v in kmap.values()).most_common():
        print(f'  {v:>3}  群 {k}')
    print(f'  **NC4 群の札が画面側に漏れた数 {leak}**(0 でも印字する)')
    print(f'  鍵は {D}/map.json(**画面には出さない**)')
    print('  **問いは (a) 流れ込みだけ。(b) 読みの分割は聞いていない**')


if __name__ == '__main__':
    main()
