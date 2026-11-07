# -*- coding: utf-8 -*-
"""L38 第2回の答を素性に結ぶ: **代用の2つの量は、作者の「線」を再現したか**(2026-11-07)。

規律13 —— **近い列で代用するなら、作者の分類を再現すると示し、的中率と取りこぼしを印字する。**
**示せなければ目視か `census`。** この器具はその印字をする。**棒は登録していない。**

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_solidcheck_join_20261107.py
"""
import collections
import json
import math
from pathlib import Path

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
D = ROOT / 'data' / 'l38_solidcheck_20261107'
SOL = ROOT / 'results' / 'l38_face_is_solid_20261107.json'
THIN, PURE = 0.55, 0.80


def main():
    kmap = json.loads((D / 'map.json').read_text(encoding='utf-8'))
    cur = {}
    rows = [json.loads(x) for x in
            open(D / 'l38_face_is_solid_20261107_answers.jsonl', encoding='utf-8')
            if x.strip()]
    for r in rows:
        cur[r['item_id']] = r
    print(f'答えの行 {len(rows)} / 生きている答 {len(cur)}'
          f'(上書き {len(rows) - len(cur)}。**行は消えていない**)')

    rec = []
    for iid, a in cur.items():
        m = kmap[iid]
        rec.append(dict(m, iid=iid, label=a['label'], value=a['value'],
                        comment=a.get('comment', ''),
                        hit=m['why'].startswith(('当たり', '既知')),
                        form=m['thin_frac'] >= THIN,
                        ink=m['pure_black'] is not None and m['pure_black'] < PURE))

    print('\n--- 既知セル(前に「これは線」と答えられた2面)---')
    for r in rec:
        if r['why'].startswith('既知'):
            print(f"  面積 {r['area']:>7,}  今「{r['label']}」"
                  f"  {'**一致**' if r['value'] == 'line' else '**食い違う**'}"
                  + (f"  一言「{r['comment']}」" if r['comment'] else ''))

    print('\n--- 的中と取りこぼし(規律13)---')
    t = collections.Counter((r['hit'], r['value']) for r in rec)
    print(f"{'':12}{'線':>5}{'ベタ':>5}{'分からない':>8}")
    for g, nm in ((True, '当たり12'), (False, '対照12')):
        print(f"  {nm:10}{t[(g, 'line')]:>5}{t[(g, 'solid_one')] + t[(g, 'solid_many')]:>5}"
              f"{t[(g, '__unknown__')] + t[(g, '__none__')]:>8}")
    ln = [r for r in rec if r['value'] == 'line']
    print(f"  **作者が「線」と答えた {len(ln)} 面のうち、当たりに入っていたのは "
          f"{sum(1 for r in ln if r['hit'])}** —— "
          f"**取りこぼし {sum(1 for r in ln if not r['hit'])}**")
    for nm, pr in (('形(細い割合 >= 0.55)', lambda r: r['form']),
                   ('墨(純黒 < 0.80)', lambda r: r['ink'])):
        g = [r for r in rec if pr(r)]
        o = [r for r in rec if not pr(r)]
        gl = sum(1 for r in g if r['value'] == 'line')
        ol = sum(1 for r in o if r['value'] == 'line')
        print(f'  {nm:22} 当たり {gl}/{len(g)} が「線」 / 外 {ol}/{len(o)}')
    und = [r for r in rec if r['value'] in ('__unknown__', '__none__')]
    print(f"\n  **「分からない」{len(und)}/{len(rec)}** ——"
          f" 選択肢を足しても残った(面積 {min(r['area'] for r in und):,}"
          f"〜{max(r['area'] for r in und):,})")

    # ---- 小さい面に床を引いたら何が落ちるか(**まだ引かない。数を出すだけ**)
    pop = json.load(open(SOL, encoding='utf-8'))
    tot_n, tot_a = len(pop), sum(r['area'] for r in pop)
    print(f'\n--- 面積の床を引いたら何が落ちるか(母集団 {tot_n} 面 / 面積計 {tot_a:,})---')
    print('  **まだ引かない。** 引くなら作者に聞いてから、前後の数を両方印字する(規律9・規律10)')
    for f in (1000, 2000, 5000, 10000, 20000):
        d = [r for r in pop if r['area'] < f]
        print(f'  床 {f:>6,} 未満: 面 {len(d):>4}({len(d) / tot_n:>5.1%})/ '
              f'面積 {sum(r["area"] for r in d):>10,}({sum(r["area"] for r in d) / tot_a:.2%})')


if __name__ == '__main__':
    main()
