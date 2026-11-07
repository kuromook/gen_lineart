# -*- coding: utf-8 -*-
"""L38 第2回の聞き取り: **その面はベタか、線などの別のものか**(2026-11-07)。

第1回(`l38_face_meaning_20261107`)で **8/30 が「分からない」「どれでもない」**になり、
一言が **「これは線」** と言った。**答えは直さない。直すのは問いと母集団**(指針 §2-3)——
**選択肢に「ベタではない」を足す**。

出す24面の選び方は `tools/evaluation/l38_pick_solid_check_20261107.py` が持つ
(**当たり12 + 面積を合わせた対照12**。条例6)。この器具は**問いを書くだけ**。

目隠しは第1回と同じ: **id は連番、絵は複製して名前を消す、鍵は `map.json`**。
**当たりか対照かは画面に出さない** —— 出したら作者は器具の答を見ることになる。

    /home/sh1/deepl/lineart/venv/bin/python tools/ui/l38_build_solid_interview_20261107.py
"""
import collections
import json
import os
import random
import re
import shutil
import sys

import cv2
from pathlib import Path

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
CROPS = ROOT / 'results' / 'l38_neck_crops_20261107' / 'solidcheck'
FEAT = ROOT / 'results' / 'l38_merge_cause_20261106.json'
OUT = ROOT / 'data' / 'l38_solidcheck_20261107'
SEED = 20261107
NAME = re.compile(r'^(?P<work>[^_]+)_(?P<page>.+?)_c(?P<comp>\d+)_r(?P<region>\d+)'
                  r'_area(?P<area>\d+)_L(?P<nl>\d+)_neck(?P<neck>\d+|None)_a_face\.png$')

QUESTION = '印を付けた面は、ベタのひとまとまりですか、それとも線など別のものですか'
UNIT = '面(ひとつながりに塗られた部分)'
DECIDES = 'この答えで、その面を「ベタの面」として数えるかどうかが決まります'
CHOICES = [
    {'value': 'solid_one', 'label': 'ベタで、1つのもの',
     'help': 'ベタで、この面ぜんぶで1つのものを表している'},
    {'value': 'solid_many', 'label': 'ベタで、2つ以上が繋がっている',
     'help': '別のものどうしが繋がって1つの面になっている。何と何かは一言の欄に書けます'},
    {'value': 'line', 'label': '線(ベタではない)',
     'help': '輪郭の線・コマの枠線・フキダシの線など'},
    {'value': 'not_solid_other', 'label': 'ベタでも線でもない別のもの',
     'help': '網・効果・文字など。何かは一言の欄に書けます'},
]


def main():
    feats = json.load(open(FEAT, encoding='utf-8'))
    why = json.loads((OUT / 'why.json').read_text(encoding='utf-8'))
    items, kmap, seen = [], {}, []
    for fn in sorted(os.listdir(CROPS)):
        m = NAME.match(fn)
        if not m:
            continue
        g = m.groupdict()
        hit = [r for r in feats
               if r['work'] == g['work'] and r['pid'].endswith(g['page'])
               and r['comp'] == int(g['comp']) and r['region'] == int(g['region'])
               and r['area'] == int(g['area'])]
        if len(hit) != 1:
            raise SystemExit(f'素性が {len(hit)} 行当たった(1行でない): {fn}')
        r = hit[0]
        k = f"{r['work']}|{r['pid']}|{r['comp']}|{r['region']}|{r['area']}"
        if k not in why:
            raise SystemExit(f'選んだ理由が無い面が混ざった: {k}')
        seen.append(dict(r, src=str(CROPS / fn), **why[k]))
    if len(seen) != len(why):
        raise SystemExit(f'絵 {len(seen)} / 名指し {len(why)}。**数が合わないので止める**')
    random.Random(SEED).shuffle(seen)

    img = OUT / 'crops'
    if img.exists():
        shutil.rmtree(img)
    img.mkdir(parents=True)
    for i, f in enumerate(seen, 1):
        iid = f'f{i:02d}'
        shutil.copyfile(f['src'], img / f'{iid}.png')
        # **線の墨が無い窓は、そう画面に書く。** 隠すと作者は「絵が出ていない」と思う ——
        # 原稿のその場所に線の層の墨が無いのは**原稿の性質**であって、器具の不具合ではない
        im = cv2.imread(str(img / f'{iid}.png'))
        it = {'id': iid, 'version': 'crop20261107b',
              'images': [{'src': f'crops/{iid}.png', 'target': True,
                          'caption': '原稿の絵(原寸)。橙の輪郭と枠が問う面'}]}
        if im is not None and (im.max(axis=2) < 40).mean() < 0.002:
            it['note'] = ('この窓には線の層の墨がありません(原稿がそうなっています)。'
                          '塗りの形だけで判断できなければ「分からない」で構いません')
        items.append(it)
        kmap[iid] = {k: f[k] for k in
                     ('work', 'pid', 'comp', 'region', 'area', 'why',
                      'thin_frac', 'pure_black', 'mean_gray', 'r_max')}
    spec = {'interview_id': 'l38_face_is_solid_20261107', 'question': QUESTION,
            'unit': UNIT, 'decides': DECIDES, 'choices': CHOICES, 'items': items}
    (OUT / 'questions.json').write_text(
        json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT / 'map.json').write_text(
        json.dumps(kmap, ensure_ascii=False, indent=1), encoding='utf-8')
    sys.path.insert(0, str(ROOT / 'tools' / 'ui'))
    from interview_spec import load_questions
    load_questions(OUT / 'questions.json')
    n = collections.Counter(v['why'].split('(')[0] for v in kmap.values())
    print(f'問い {len(items)} 件 -> {OUT}/questions.json(部品の点検を通った)')
    for k, v in n.most_common():
        print(f'  {v}  {k}')
    print(f'  鍵は {OUT}/map.json(**当たりか対照かは画面に出さない**)')


if __name__ == '__main__':
    main()
