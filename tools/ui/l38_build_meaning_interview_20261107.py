# -*- coding: utf-8 -*-
"""L38: **面の意味の数**を聞く問いを、共通部品の形で書き出す(2026-11-07)。

基盤の通達 `reply_interview_ui_parts_20261107.md` —— **画面は書かない。問いを JSON に書く**。
指針は `doc/ui_guidelines_20261107.md`、部品は `tools/ui/interview_*`。

    /home/sh1/deepl/lineart/venv/bin/python tools/ui/l38_build_meaning_interview_20261107.py
    -> data/l38_meaning_20261107/{questions.json, map.json, img/*.png}

## 目隠しを部品の上でも保つ

- **id は `f01` …** の連番で、**作品名も群も入れない**。画像も `img/f01.png` に**複製**する
  (元の名前は `2L_neck/..._L2_neck1_a_face.png` で、**名前が答を教えてしまう**)
- **並びは固定の種で混ぜる**。`map.json`(= 鍵)は**画面に出さない**
- **首の窓は出さない。** 「細い首が在る」と見せること自体が誘導になる
  (前の UI では出していた。今回の帰無値は**面1枚だけ**で聞く)

## 何を聞くか

- **無作為30面**(`--random`。素性で選んでいない)= **帰無値**になる
- **再掲3面**(既に裁定の在る面。指針 §6-3「同じ問いをもう一度出せる作り」)——
  **本人内の一貫性**と、**描き方を変えた(半透明+輪郭+在りかの枠)後も同じ答が返るか**の点検
"""
import collections
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
CROPS = ROOT / 'results' / 'l38_neck_crops_20261107'
FEAT = ROOT / 'results' / 'l38_merge_cause_20261106.json'
OLD = ROOT / 'labels' / 'face_meaning_count.jsonl'
OUT = ROOT / 'data' / 'l38_meaning_20261107'
SEED = 20261107
# **再掲する面**(既に裁定が在る3面。面積で名指しする —— `comp` は鍵にならない)
# 2,232,129 = 「髪と衣類」 / 2,503,898 = 「1つのもの(背景のベタのみ)」 /
# 1,293,677 = 「髪とコマ枠線」(軸4 が唯一発火した面)
REPEAT = (2232129, 2503898, 1293677)
NAME = re.compile(r'^(?P<work>[^_]+)_(?P<page>.+?)_c(?P<comp>\d+)_r(?P<region>\d+)'
                  r'_area(?P<area>\d+)_L(?P<nl>\d+)_neck(?P<neck>\d+|None)_a_face\.png$')

QUESTION = '印を付けた面は、1つのものですか、それとも2つ以上が繋がっていますか'
UNIT = '面(ひとつながりのベタ)'
DECIDES = 'この答えで、ベタの面をそのまま1つの単位として扱えるかが決まります'
CHOICES = [
    {'value': 'one', 'label': '1つのもの',
     'help': 'この面ぜんぶで、1つのものを表している'},
    {'value': 'many', 'label': '2つ以上が繋がっている',
     'help': '別のものどうしが繋がって1つの面になっている。何と何かは一言の欄に書けます'},
]


def faces():
    feats = json.load(open(FEAT, encoding='utf-8'))
    out = []
    for cell in sorted(os.listdir(CROPS)):
        d = CROPS / cell
        if not d.is_dir():
            continue
        for fn in sorted(os.listdir(d)):
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
            out.append(dict(hit[0], cell=cell, src=str(d / fn)))
    return out


def main():
    allf = faces()
    cut = collections.Counter()
    pick = []
    for f in allf:
        if f['cell'] == 'random':
            pick.append(dict(f, why='無作為(帰無値)'))
        elif f['area'] in REPEAT:
            pick.append(dict(f, why='再掲(裁定済み。一貫性の点検)'))
        else:
            cut['**選んだ面で、再掲に入れなかった分**(裁定済み。答は消していない)'] += 1
    if len(pick) != 33:
        raise SystemExit(f'選んだ面が {len(pick)}。33(無作為30 + 再掲3)でない')
    random.Random(SEED).shuffle(pick)

    img = OUT / 'crops'
    if img.exists():
        shutil.rmtree(img)
    img.mkdir(parents=True)
    items, kmap = [], {}
    for i, f in enumerate(pick, 1):
        iid = f'f{i:02d}'
        shutil.copyfile(f['src'], img / f'{iid}.png')
        items.append({'id': iid, 'version': 'crop20261107b',
                      'images': [{'src': f'crops/{iid}.png', 'target': True,
                                  'caption': '原稿の絵(原寸)。橙の輪郭と枠が問う面'}]})
        kmap[iid] = {k: f[k] for k in
                     ('work', 'pid', 'comp', 'region', 'area', 'cell',
                      'n_layer', 'neck', 'why')}
    spec = {'interview_id': 'l38_face_meaning_20261107', 'question': QUESTION,
            'unit': UNIT, 'decides': DECIDES, 'choices': CHOICES, 'items': items}
    (OUT / 'questions.json').write_text(
        json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT / 'map.json').write_text(
        json.dumps(kmap, ensure_ascii=False, indent=1), encoding='utf-8')

    # **部品の点検を自分で通してから終わる**(問いの形で落ちるなら、ここで落ちるべき)
    sys.path.insert(0, str(ROOT / 'tools' / 'ui'))
    from interview_spec import load_questions
    load_questions(OUT / 'questions.json')

    n = collections.Counter(v['why'] for v in kmap.values())
    print(f'問い {len(items)} 件 -> {OUT}/questions.json(部品の点検を通った)')
    for k, v in n.most_common():
        print(f'  {v}  {k}')
    for k, v in cut.most_common():
        print(f'  入れなかった {v}  {k}')
    print(f'  鍵は {OUT}/map.json(**画面には出さない**)/ 絵は {len(items)} 枚を複製')
    print('  **首の窓は出していない**(細い首が在ると見せること自体が誘導になる)')


if __name__ == '__main__':
    main()
