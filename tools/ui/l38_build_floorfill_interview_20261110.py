# -*- coding: utf-8 -*-
"""床の材料を足す聞き取り(登録 `doc/l38_floor_fill_registration_20261110.md`)。

標本 `data/l38_floorfill_20261110/order.json`(並べ替えた順)の先頭から、切り出しの在るものを 31 件。
**画面に出るのは `uNNN` と絵だけ。面積・層・片かどうかは `map.json`**(条例11)。

    /home/sh1/deepl/lineart/venv/bin/python tools/ui/l38_build_floorfill_interview_20261110.py
"""
import collections
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
CROPS = ROOT / 'results' / 'floorfill20261110'
D = ROOT / 'data' / 'l38_floorfill_20261110'
WANT = 31
QUESTION = '印を付けた部分は、ベタの面ですか、線ですか'
UNIT = '印を付けた部分(ひとつながりに塗られた所)'
DECIDES = 'この答えで、小さすぎる部分を「面」と数えない境目(床)を引き直すかどうかが決まります'
CHOICES = [
    {'value': 'solid', 'label': 'ベタの面', 'help': '塗りの面として塗られている'},
    {'value': 'line', 'label': '線', 'help': '線として引かれている(輪郭・枠線・描き込みなど)'},
]
CAPTION = '原稿の絵(原寸)。橙の輪郭と枠が問う部分。淡い灰は同じコマの他の塗り'


def main():
    order = json.load(open(D / 'order.json', encoding='utf-8'))
    if (D / 'crops').exists():
        shutil.rmtree(D / 'crops')
    (D / 'crops').mkdir(parents=True)
    items, kmap, miss = [], {}, []
    for r in order:
        if len(items) == WANT:
            break
        src = CROPS / f'{r["fid"]}.png'
        if not src.is_file():
            miss.append(r['fid'])
            continue
        shutil.copyfile(src, D / 'crops' / src.name)
        items.append({'id': r['fid'], 'version': 'ff1',
                      'images': [{'src': f'crops/{src.name}', 'target': True, 'caption': CAPTION}]})
        kmap[r['fid']] = r
    if len(items) < WANT:
        sys.exit(f'切り出しが {len(items)} 件しかない。order を増やす')
    spec = {'interview_id': 'l38_floorfill_20261110', 'page_title': 'L38 ベタか線か',
            'question': QUESTION, 'unit': UNIT, 'decides': DECIDES, 'choices': CHOICES,
            'items': items}
    (D / 'questions.json').write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    (D / 'map.json').write_text(json.dumps(kmap, ensure_ascii=False, indent=1), encoding='utf-8')
    sys.path.insert(0, '/home/sh1/deepl/lineart/tools/ui')
    from interview_spec import load_questions
    load_questions(D / 'questions.json')
    shown = json.dumps(items, ensure_ascii=False) + QUESTION + UNIT + DECIDES + json.dumps(CHOICES, ensure_ascii=False)
    leak = sum(shown.count(w) for w in ('ako', 'page', 'piece', '面積', '層'))
    print(f'問い {len(items)} 件(切り出しが無くて飛ばした {len(miss)} {miss})-> {D}/questions.json')
    print(f'  床の上 {sum(v["area"] >= 6000 for v in kmap.values())} / 床の下 '
          f'{sum(v["area"] < 6000 for v in kmap.values())} / 片 '
          f'{collections.Counter(v["origin"] for v in kmap.values())["piece"]}')
    print(f'  **画面側への作品名・頁・機械の見立ての漏れ {leak}**(0 が正しい)')


if __name__ == '__main__':
    main()
