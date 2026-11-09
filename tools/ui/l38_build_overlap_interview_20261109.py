# -*- coding: utf-8 -*-
"""重ね塗りの疑い 4 面の問いを共通部品の形で書き出す(登録 `doc/l38_overlap_registration_20261109.md`)。

切り出しは `tools/evaluation/l38_overlap_crops_20261109.py`(親ごとに同じ窓)。
**機械の見立て(疑い・比・層番号)は画面に出さない**(条例11)。鍵は `map.json`。

    /home/sh1/deepl/lineart/venv/bin/python tools/ui/l38_build_overlap_interview_20261109.py
"""
import collections
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
CROPS = ROOT / 'results' / 'overlap20261109'
D = ROOT / 'data' / 'l38_overlap_20261109'
QUESTION = '並べた塗り(各画像の橙)どうしは、同じものを別々のレイヤに重ねて塗ったものですか'
UNIT = '面(並べた画像は同じ場所の同じ窓。片ごとに1枚)'
DECIDES = 'この答えで、この面を1つとして数えるか、片ごとに数えるかが決まります'
CHOICES = [
    {'value': 'same', 'label': '同じものを重ねた', 'help': '重ねて塗っただけで、描いているものは1つ'},
    {'value': 'different', 'label': '別々のもの', 'help': 'それぞれ別のものとして塗っている'},
]


def main():
    meta = json.load(open(CROPS / 'meta.json', encoding='utf-8'))
    by = collections.OrderedDict()
    for m in meta:
        by.setdefault((m['work'], m['pid'], m['comp'], m['region']), []).append(m)
    if (D / 'crops').exists():
        shutil.rmtree(D / 'crops')
    (D / 'crops').mkdir(parents=True)
    items, kmap = [], {}
    for n, (k, ps) in enumerate(by.items(), 1):
        oid = f'o{n:02d}'
        imgs = []
        for j, m in enumerate(ps, 1):
            shutil.copyfile(CROPS / f'{m["fid"]}.png', D / 'crops' / f'{oid}_{j}.png')
            imgs.append({'src': f'crops/{oid}_{j}.png', 'target': True,
                         'caption': f'塗り {j}(橙)。淡い灰は同じ場所の他の塗り。原寸'})
        items.append({'id': oid, 'version': 'ov1', 'images': imgs})
        kmap[oid] = ps
    spec = {'interview_id': 'l38_overlap_20261109', 'page_title': 'L38 重ねた塗り',
            'question': QUESTION, 'unit': UNIT, 'decides': DECIDES, 'choices': CHOICES,
            'items': items}
    (D / 'questions.json').write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    (D / 'map.json').write_text(json.dumps(kmap, ensure_ascii=False, indent=1), encoding='utf-8')
    sys.path.insert(0, '/home/sh1/deepl/lineart/tools/ui')
    from interview_spec import load_questions
    load_questions(D / 'questions.json')
    page = json.dumps(items, ensure_ascii=False) + QUESTION + UNIT + DECIDES + json.dumps(CHOICES, ensure_ascii=False)
    leak = sum(page.count(w) for w in ('ako', 'page', '疑い', '1.5', '層'))
    print(f'問い {len(items)} 件 / 画像 {sum(len(i["images"]) for i in items)} -> {D}/questions.json')
    print(f'  **画面側への作品名・頁・機械の見立ての漏れ {leak}**(0 が正しい)')


if __name__ == '__main__':
    main()
