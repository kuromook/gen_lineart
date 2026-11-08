# -*- coding: utf-8 -*-
"""L39 第2の棒(人の判定)の問いを共通部品の形で書き出す(登録 §8 / 追補1)。

- 種 20261108 の重みの予測。各作品の試験コマ全部から 6 つずつ(`default_rng(20261108)`、重複なし)
- 左 = 線の層だけ / 右 = 線の層 + 機械の塗り(黒)。コマの外は淡い灰。**作者の塗りは出さない**
- 画面に出るのは `hNN` と絵だけ。作品・コマ・点は `map.json` の側

    /home/sh1/deepl/lineart/venv/bin/python tools/ui/l39_build_human_bar_20261109.py
"""
import collections
import glob
import json
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
sys.path.insert(0, str(ROOT / 'tools' / 'evaluation'))
from l39_prepare_20261108 import CACHE, unpack                      # noqa: E402
from l17_known_cells import arms, jobs, zmap                        # noqa: E402

SEED = 20261108
PER_WORK = 6
D = ROOT / 'data' / 'l39_human_20261109'
QUESTION = '右の黒い塗りは、この線画に作者が塗る面として読めますか'
UNIT = 'コマ(右の絵の黒い塗り全体)'
DECIDES = 'この答えで、線だけから塗りを当てる機械が人の目に使えるかが決まります'
CHOICES = [
    {'value': 'yes', 'label': '読める', 'help': 'この線画なら、塗りはおおむねこう入る'},
    {'value': 'partly', 'label': '一部', 'help': '合っている所もあるが、外れている所が目立つ'},
    {'value': 'no', 'label': '読めない', 'help': 'この線画の塗りとしては読めない(何も塗っていない場合も含む)'},
]


def main():
    rng = np.random.default_rng(SEED)
    pages = {w: [d for d in (pickle.load(open(p, 'rb'))
                             for p in sorted(glob.glob(f'{CACHE}/[0-9]*.pkl'))) if d['work'] == w]
             for w in ('ako6', 'ako7')}
    picks = []
    for w in ('ako6', 'ako7'):
        allp = [(pg['pid'], i) for pg in pages[w] for i in range(len(pg['panels']))]
        ix = rng.choice(len(allp), PER_WORK, replace=False)
        picks += [(w,) + allp[k] for k in sorted(ix)]
        print(f'{w}: 試験コマ {len(allp)} から {PER_WORK}')
    order = rng.permutation(len(picks))
    fill, line = arms('gated')
    lz = {(pid, w): l for pid, w, fz, l, sel in jobs(zmap(), fill, line)}
    if D.exists():
        shutil.rmtree(D / 'crops', ignore_errors=True)
    (D / 'crops').mkdir(parents=True, exist_ok=True)
    items, kmap = [], {}
    for n, k in enumerate(order, 1):
        w, pid, pi = picks[k]
        pg = next(p for p in pages[w] if p['pid'] == pid)
        pan = pg['panels'][pi]
        y0, y1, x0, x1 = pan['box']
        P = unpack(pan['P'])
        L = np.zeros(P.shape, bool)
        for lid, path, z in lz[(pid, w)]:
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if im is not None and im.shape == tuple(pg['shape']):
                L |= im[y0:y1, x0:x1] < 128
        pr = cv2.imread(f'{ROOT}/results/l39_runs/test_{w}_seed{SEED}/prob/{pid}.png',
                        cv2.IMREAD_GRAYSCALE)
        H, W = pg['shape']
        M = cv2.resize(pr, (W, H), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1] >= 128
        M &= P
        def draw(fillm):
            o = np.full(P.shape, 255, np.uint8)
            o[~P] = 225
            o[fillm] = 0
            o[L & P] = 0
            return o
        hid = f'h{n:02d}'
        cv2.imwrite(str(D / 'crops' / f'{hid}_line.png'), draw(np.zeros_like(P)))
        cv2.imwrite(str(D / 'crops' / f'{hid}_fill.png'), draw(M))
        items.append({'id': hid, 'version': 'l39h1',
                      'images': [{'src': f'crops/{hid}_line.png', 'caption': '線画だけ(原寸)'},
                                 {'src': f'crops/{hid}_fill.png', 'target': True,
                                  'caption': '線画 + 機械の塗り(黒)。淡い灰はコマの外'}]})
        kmap[hid] = dict(work=w, pid=pid, panel=[pan['key'][0], int(pan['key'][1])], box=[int(v) for v in pan['box']],
                         n_faces=len(pan['faces']), fill_px=int(M.sum()))
    spec = {'interview_id': 'l39_human_20261109', 'page_title': 'L39 線画の塗り',
            'question': QUESTION, 'unit': UNIT, 'decides': DECIDES, 'choices': CHOICES,
            'items': items}
    (D / 'questions.json').write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    (D / 'map.json').write_text(json.dumps(kmap, ensure_ascii=False, indent=1), encoding='utf-8')
    sys.path.insert(0, '/home/sh1/deepl/lineart/tools/ui')
    from interview_spec import load_questions
    load_questions(D / 'questions.json')
    leak = sum(json.dumps(it, ensure_ascii=False).count(s) for it in items
               for s in ('ako', 'seed', 'page'))
    print(f'問い {len(items)} 件 -> {D}/questions.json(部品の点検を通った)')
    print(f'  作品ごと {collections.Counter(v["work"] for v in kmap.values())} / '
          f'予測が空のコマ {sum(1 for v in kmap.values() if v["fill_px"] == 0)} / '
          f'面の無いコマ {sum(1 for v in kmap.values() if v["n_faces"] == 0)}')
    print(f'  **NC 作品・種・頁の名前が画面側に漏れた数 {leak}**(0 でも印字する)')


if __name__ == '__main__':
    main()
