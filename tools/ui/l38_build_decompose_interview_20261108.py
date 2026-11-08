# -*- coding: utf-8 -*-
"""L38 **分解しないといけない面はどれだけ在るか**の問いを、共通部品の形で書き出す(2026-11-08)。

登録: `doc/l38_decompose_registration_20261108.md`(**走る前に固定した**)。
標本は `tools/evaluation/l38_pick_decompose_20261108.py`、
切り出しは `tools/evaluation/l38_piece_crops_20261108.py --masks`。

- **問いの文に作者の仮説(塗りつぶしが流れ込んだ)を書かない**(登録 NC5。**証人を誘導する**)。
  **選択肢の助けの文は左右対称に保つ** —— どちらの側にも手がかりを足さない
- **これは「1つのものか2つ以上か」の言い換えではない。** あちらは**読み**の問い、
  こちらは**作りの欠陥**の問い。枠線を飲んだ面は**読みとしては1つのもの**になる
- **目隠し**: 画面に出るのは `fNN` と絵だけ。群(一様 / 枠線 / 既知)は `map.json` の側

    /home/sh1/deepl/lineart/venv/bin/python tools/ui/l38_build_decompose_interview_20261108.py
"""
import collections
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
CROPS = ROOT / 'results' / 'decompose20261108'
D = ROOT / 'data' / 'l38_decompose_20261108'

QUESTION = '印を付けた面は、いくつかに分解しないといけないものですか'
UNIT = '面(ひとつながりのベタ)'
DECIDES = 'この答えで、面の標準をもう一度作り直すかどうかが決まります'
CHOICES = [
    {'value': 'split', 'label': '分解しないといけない',
     'help': '別のものが繋がってしまっている。どこで分けるかは次の画面で区分けできます'},
    {'value': 'whole', 'label': 'このままで1つ',
     'help': '1つのものとして塗った面で、分ける必要はない'},
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
        items.append({'id': r['fid'], 'version': 'crop20261108c',
                      'images': [{'src': f'crops/{r["fid"]}.png', 'target': True,
                                  'caption': CAPTION}]})
        kmap[r['fid']] = {k: r[k] for k in
                          ('work', 'pid', 'comp', 'region', 'area', 'group', 'origin',
                           'piece_lid', 'parent', 'siblings', 'neck', 'band_cov',
                           'thin_frac', 'pure_black', 'r_med', 'r_max')}
    if miss:
        sys.exit(f'**切り出しが無い単位 {len(miss)}: {miss}** —— '
                 f'数が合うまで聞き取りに出さない(登録の NC1/NC2)')
    spec = {'interview_id': 'l38_decompose_20261108', 'question': QUESTION,
            'unit': UNIT, 'decides': DECIDES, 'choices': CHOICES, 'items': items}
    (D / 'questions.json').write_text(
        json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    (D / 'map.json').write_text(
        json.dumps(kmap, ensure_ascii=False, indent=1), encoding='utf-8')

    sys.path.insert(0, str(ROOT / 'tools' / 'ui'))
    from interview_spec import load_questions
    load_questions(D / 'questions.json')           # 部品の点検を自分で通してから終わる

    # **NC4 群の札が画面に行く道に無いことを数えて印字する**(0 でも印字する。条例4)
    page = json.dumps(spec, ensure_ascii=False)
    leak = sum(page.count(w) for w in ('uniform', 'band', 'known_', '枠線', '一様'))
    n = collections.Counter(v['group'] for v in kmap.values())
    print(f'問い {len(items)} 件 -> {D}/questions.json(部品の点検を通った)')
    for k, v in n.most_common():
        print(f'  {v:>3}  群 {k}')
    print(f'  **NC4 画面に行く JSON の中の群の名前 {leak} 回**(0 が正しい)')
    print(f'  鍵は {D}/map.json(**画面には出さない**)/ 絵は {len(items)} 枚を複製')
    print('  **問いの文に作者の仮説は書いていない**(NC5)。選択肢の助けは左右対称')


if __name__ == '__main__':
    main()
