# -*- coding: utf-8 -*-
"""L38: **「ベタか線か」を作者に聞く24面**を選ぶ(2026-11-07)。

無作為30面で **8/30 が「分からない」「どれでもない」**になり、一言が **「これは線」** と言った。
面の母集団に**ベタでないものが混ざっている**。洗う前に、**代用の量が作者の分類を再現するか**を
確かめる(規律13: 的中率と取りこぼしを印字してから使う)。

## 選び方(**対照は試験する1軸だけで違う**。条例6)

- **当たり 12面** —— `l38_face_is_solid_20261107.json` の2つの側:
  **形**(`thin_frac >= 0.55`)/ **墨**(`pure_black < 0.80`)。
  **両方・形だけ・墨だけ**から取り、**作者が既に「これは線」と答えた2面を必ず入れる**(既知セル)
- **対照 12面** —— **どちらにも当たらない面から、面積が最も近いものを1対1で取る**。
  **面積を揃えるのは、小ささが答を決めている可能性と区別するため**
  (当たった面は小さい。面積を揃えない対照だと、軸が2つ違う)

**床は答を見た後に引いたもの**(前の節)。**だからこれは試験ではなく、床の素性を作者に当てる手である。**

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_pick_solid_check_20261107.py
"""
import collections
import json
import random
from pathlib import Path

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
SOL = ROOT / 'results' / 'l38_face_is_solid_20261107.json'
IV = ROOT / 'data' / 'l38_meaning_20261107'
OUT = ROOT / 'data' / 'l38_solidcheck_20261107'
THIN, PURE = 0.55, 0.80
SEED = 20261107
# 作者が「これは線」「枠線とキャラクターの線…」と答えた2面(面積で名指し)
KNOWN = {1464: 'これは線', 32243: '枠線とキャラクターの線、さらにフキダシの線'}


def key(r):
    return (r['work'], r['pid'], r['comp'], r['region'], r['area'])


def main():
    rows = json.load(open(SOL, encoding='utf-8'))
    for r in rows:
        r['form'] = r['thin_frac'] >= THIN
        r['ink'] = r['pure_black'] is not None and r['pure_black'] < PURE
    # 既に答えた面は、既知セル以外は出さない(同じ絵を二度聞かない)
    kmap = json.loads((IV / 'map.json').read_text(encoding='utf-8'))
    asked = {(m['work'], m['pid'], m['comp'], m['region'], m['area'])
             for m in kmap.values()}
    hit = [r for r in rows if r['form'] or r['ink']]
    clean = [r for r in rows if not (r['form'] or r['ink']) and key(r) not in asked]
    print(f'面 {len(rows)} / 当たり {len(hit)} / 当たらない(未出題) {len(clean)}')

    rnd = random.Random(SEED)
    pick, cut = [], collections.Counter()
    for r in hit:
        if r['area'] in KNOWN:
            pick.append(dict(r, why=f'既知セル(作者「{KNOWN[r["area"]]}」)'))
    want = {'両方': 2, '形だけ': 6, '墨だけ': 2}
    for nm, n in want.items():
        pool = [r for r in hit if key(r) not in {key(x) for x in pick}
                and key(r) not in asked
                and ((nm == '両方' and r['form'] and r['ink'])
                     or (nm == '形だけ' and r['form'] and not r['ink'])
                     or (nm == '墨だけ' and r['ink'] and not r['form']))]
        rnd.shuffle(pool)
        pick += [dict(r, why=f'当たり({nm})') for r in pool[:n]]
        if len(pool) < n:
            cut[f'{nm}: 足りない(在庫 {len(pool)} / 欲しい {n})'] += 1
    # 対照: 面積が最も近い「当たらない」面を1対1で
    used = set()
    ctl = []
    for r in pick:
        cand = sorted((c for c in clean if key(c) not in used),
                      key=lambda c: abs(c['area'] - r['area']))
        if not cand:
            cut['対照の在庫が尽きた'] += 1
            continue
        c = cand[0]
        used.add(key(c))
        ctl.append(dict(c, why=f'対照(面積を合わせた。相手の面積 {r["area"]:,})'))
    out = pick + ctl
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'faces.json').write_text(json.dumps(
        [{k: r[k] for k in ('work', 'pid', 'comp', 'region', 'area')} for r in out],
        ensure_ascii=False), encoding='utf-8')
    (OUT / 'why.json').write_text(json.dumps(
        {f"{r['work']}|{r['pid']}|{r['comp']}|{r['region']}|{r['area']}":
         dict(why=r['why'], thin_frac=r['thin_frac'], pure_black=r['pure_black'],
              mean_gray=r['mean_gray'], r_max=r['r_max'], area=r['area'])
         for r in out}, ensure_ascii=False, indent=1), encoding='utf-8')
    n = collections.Counter(r['why'].split('(')[0] for r in out)
    print(f'選んだ {len(out)} 面 -> {OUT}/faces.json')
    for k, v in n.most_common():
        print(f'  {v}  {k}')
    for k, v in cut.most_common():
        print(f'  **{k}**')
    a = sorted(r['area'] for r in pick)
    b = sorted(r['area'] for r in ctl)
    print(f'  面積: 当たり 中央 {a[len(a) // 2]:,} / 対照 中央 {b[len(b) // 2]:,}'
          f'(**揃えてある**)')


if __name__ == '__main__':
    main()
