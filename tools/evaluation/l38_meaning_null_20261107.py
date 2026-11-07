# -*- coding: utf-8 -*-
"""L38 **帰無値**: 無作為に取った面のうち、作者が「2つ以上」と答える割合(2026-11-07)。

**これは測定ではなく記述である**(棒を登録していない)。だが **帰無値は棒の前に要る**(規律9)——
11-07 の 9面は**私が選んだ**ので、その 6/9 は母集団の割合ではない。

読むもの: `data/l38_meaning_20261107/`(部品の答・鍵)と `labels/face_meaning_count.jsonl`
(**前の画面の答。形が違うので別に読む**)、素性は `l38_merge_cause` / `l38_merge_audit`。

    /home/sh1/deepl/lineart/venv/bin/python tools/evaluation/l38_meaning_null_20261107.py
"""
import collections
import json
import math
from pathlib import Path

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
IV = ROOT / 'data' / 'l38_meaning_20261107'
ANS = IV / 'l38_face_meaning_20261107_answers.jsonl'
OLD = ROOT / 'labels' / 'face_meaning_count.jsonl'
CAUSE = ROOT / 'results' / 'l38_merge_cause_20261106.json'
AUDIT = ROOT / 'results' / 'l38_merge_audit_20261106.json'
FRAME = ROOT / 'results' / 'l38_frame_axis_20261107.json'


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def key(r):
    return (r['work'], r['pid'], r['comp'], r['region'], r['area'])


def live(path, kf):
    """**ファイルの順に畳む。** 後の行が生きる。`undo` はその時の答を消す(行は残る)。"""
    rows = [json.loads(x) for x in open(path, encoding='utf-8') if x.strip()]
    cur, over = {}, 0
    for r in rows:
        k = kf(r)
        if r.get('kind') == 'undo':
            cur.pop(k, None)
        else:
            over += k in cur
            cur[k] = r
    return rows, cur, over


def main():
    kmap = json.loads((IV / 'map.json').read_text(encoding='utf-8'))
    rows, cur, over = live(ANS, lambda r: r['item_id'])
    cause = {key(r): r for r in json.load(open(CAUSE, encoding='utf-8'))}
    audit = {key(r): r for r in json.load(open(AUDIT, encoding='utf-8'))}
    frame = {key(r): r for r in json.load(open(FRAME, encoding='utf-8'))}
    _, old, _ = live(OLD, key)

    rec = []
    for iid, a in cur.items():
        m = kmap[iid]
        k = (m['work'], m['pid'], m['comp'], m['region'], m['area'])
        rec.append(dict(m, iid=iid, value=a['value'], label=a['label'],
                        comment=a.get('comment', ''), ts=a['ts'],
                        top_share=audit[k]['top_share'],
                        band_cov=frame[k]['band_cov'] if k in frame else None,
                        old=old.get(k, {}).get('label', '')))
    print(f'答えの行 {len(rows)} / 生きている答 {len(cur)} / 上書き {over}'
          f'(**行は消えていない**)')

    # ---- 1. 再掲(本人内の一貫性。指針 §6-3)
    rp = [r for r in rec if r['why'].startswith('再掲')]
    print(f'\n--- 再掲 {len(rp)} 面(前の画面での裁定と比べる)---')
    agree = 0
    for r in sorted(rp, key=lambda r: -r['area']):
        same = r['label'].startswith(r['old'][:3]) if r['old'] else False
        agree += same
        print(f"  面積 {r['area']:>9,}  前「{r['old']}」 -> 今「{r['label']}」"
              f"  {'**一致**' if same else '**食い違う**'}"
              + (f"  一言「{r['comment']}」" if r['comment'] else ''))
    print(f'  一致 {agree} / {len(rp)}')

    # ---- 2. 帰無値(無作為のみ)
    rd = [r for r in rec if r['why'].startswith('無作為')]
    c = collections.Counter(r['label'] for r in rd)
    many = c['2つ以上が繋がっている']
    lo, hi = wilson(many, len(rd))
    print(f'\n--- **帰無値**: 無作為 {len(rd)} 面 ---')
    for k, v in c.most_common():
        print(f'  {v:>3}  {k}')
    print(f'  **「2つ以上」{many}/{len(rd)} = {many / len(rd):.1%}'
          f'(Wilson 95% {lo:.1%}〜{hi:.1%})**')
    nd = c['分からない'] + c['どれでもない']
    print(f'  **「分からない」+「どれでもない」= {nd}/{len(rd)} = {nd / len(rd):.0%}** ——'
          f' 指針 §2-3: 多ければ**問いの作りの欠陥として記録する**(答えは直さない)')
    for r in rd:
        if r['comment']:
            print(f"  一言: 「{r['comment']}」({r['label']}、面積 {r['area']:,})")

    # ---- 3. 素性ごと(記述。棒は登録していない)
    print('\n--- 無作為30面を素性で割る(**記述**。棒も帰無値の比較も登録していない)---')
    print(f"{'素性':24}{'面':>4}{'2つ以上':>7}{'分からない等':>9}")
    for nm, pr in [('層 >= 2', lambda r: r['n_layer'] >= 2),
                   ('層 == 1', lambda r: r['n_layer'] == 1),
                   ('首あり', lambda r: r['neck'] is not None),
                   ('首なし', lambda r: r['neck'] is None),
                   ('面積 >= 100,000', lambda r: r['area'] >= 100000),
                   ('面積 < 100,000', lambda r: r['area'] < 100000)]:
        g = [r for r in rd if pr(r)]
        if not g:
            print(f'  {nm:22}{0:>4}      —        —')
            continue
        m = sum(1 for r in g if r['value'] == 'many')
        u = sum(1 for r in g if r['value'] in ('__unknown__', '__none__'))
        print(f'  {nm:22}{len(g):>4}{m:>7}{u:>9}')
    big = [r for r in rd if r['area'] >= 100000]
    print(f'\n  **無作為30面のうち面積 10万以上は {len(big)} 面**'
          f'(選んだ9面は**全部 43万以上**だった)——'
          f' **同じ母集団を見ていない**ので、6/9 と並べて読まない')


if __name__ == '__main__':
    main()
