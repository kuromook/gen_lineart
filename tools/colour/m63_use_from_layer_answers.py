# -*- coding: utf-8 -*-
"""m63: **作者の層ごとの答から `use` を導く**(追補59 §E → **追補77 §C で書き直し**)。

**非対称にする(追補77 §C。走る前に登録済み。ここでは実装するだけ)**:

  | 根拠 | 向き |
  |---|---|
  | **作者の直接の言明**(頁の名指し) | **両方向。これが錨**(規律6) |
  | **答の集合に在る種類**(肯定) | **「採点する」側にだけ** |
  | **答の集合に無い種類**(欠測) | **どちら側にも使えない** |

**理由**: **45頁のうち全層を聞いた頁は 0**(非フォルダー層の中央 50%、最小 5.9%)。
→ **「`仕上がった絵` の答が無い」は「`仕上がった絵` の層が無い」を意味しない。**
**`drawn_ratio = 0` を「画素が無い」と読んだ罠(追補53)と同じ形。**

**既知セル(追補77 §D)**:
  **錨の頁すべて** -> **作者の言明と同じ側。かつ根拠が「作者の言明」**(答の集計ではなく)
  **新しい既知セル** -> **答の集計だけの判定と最終の判定の差が 0 でない**
                       (0 なら書き直しが何も変えていない = 器具を疑う)

**錨の表は追跡外**(`results/alias/author_anchors.json`。追補73 の形)。
**結合は `page_path`(フルパス)。ファイル名では結合しない。**
**飛ばした行は理由別に件数を印字する。**
"""
import collections
import csv
import json
import sys
from pathlib import Path

OUT = Path('results/m63_use_from_answers'); OUT.mkdir(parents=True, exist_ok=True)
FIN = '仕上がった絵(統合済み)'
ANCHORS = Path('results/alias/author_anchors.json')   # **追跡外**
ROUGH_ONLY = {'下絵・ラフ', 'ノンブル', 'トンボ', '印刷用の枠', 'メモ',
              'セリフ・文字', 'わからない', '添削', 'メモ 添削'}


def load_anchors():
    if not ANCHORS.exists():
        raise SystemExit(f'**錨の表が無い: {ANCHORS}。走らせない**')
    return {a['page_path']: a for a in
            json.loads(ANCHORS.read_text(encoding='utf-8'))['anchors']}


def tally_only(s):
    """**答の集計だけの判定**(追補59 §E の元の規則。差を見るために残す)。"""
    if FIN in s:
        return '採点する(仕上がった絵が在る)', FIN
    if '線画' in s and '色・ベタ(塗り)' in s:
        return '採点する(工程が揃っている)', '線画+色・ベタ'
    if s - ROUGH_ONLY == set():
        return '採点しない(下絵のみ)', '下絵・ラフ以外に絵が無い'
    if '資料用の画像' in s:
        return '採点しない(資料用の画像)', '資料用の画像'
    return 'undecided', '規則のどれにも当たらない'


def main():
    anchors = load_anchors()
    q = {it['order']: it for it in
         json.load(open('results/m56_ui/queue.json'))['queue']}
    last, drop = {}, collections.Counter()
    for l in open('labels/layer_kind.jsonl', encoding='utf-8'):
        if l.strip():
            r = json.loads(l); last[r['order']] = r      # 後勝ち = 第2走
    bypage = collections.defaultdict(list)
    for o, a in last.items():
        it = q.get(o)
        if it is None:
            drop['行列に無い order'] += 1; continue
        kind = (a['word'] if a['choice'] == 'word'
                else (a['free'].strip() if a['choice'] == 'free' else 'わからない'))
        bypage[it['pid']].append(kind)

    print('== 母集団の差分(理由別。規律9)==')
    print(f'  作者の答        {len(last)}')
    for k, v in drop.most_common():
        print(f'  {k:24s} -{v}')
    print(f'  **頁            {len(bypage)}**')

    rows = []
    for src, kinds in bypage.items():
        s = set(kinds)
        t_use, t_why = tally_only(s)
        a = anchors.get(src)
        if a:
            # **作者の言明は両方向に使える。これが錨**(追補77 §C)
            use = (f'{a["verdict"]}(作者の言明)' if a['verdict'] == '採点する'
                   else f'{a["verdict"]}(作者の言明)')
            why = f'作者の言明「{a["words"]}」'
        elif t_use.startswith('採点する'):
            # **肯定の答は「採点する」側にだけ使える**
            use, why = t_use, t_why
        else:
            # **答の集合に無いことは、どちら側にも使えない**
            use = 'undecided'
            why = f'答が薄い(聞いた層 {len(kinds)})。集計だけなら「{t_use}」'
        rows.append(dict(page_path=src, use=use, 根拠=why,
                         集計だけの判定=t_use, 層=len(kinds),
                         種類=' / '.join(f'{k}×{kinds.count(k)}'
                                       for k in sorted(s, key=kinds.count,
                                                       reverse=True))))

    print(f'\n== 導いた `use`({len(rows)} 頁)==')
    for k, v in collections.Counter(r['use'] for r in rows).most_common():
        print(f'  {k:34s} {v:>3}')

    print('\n== 既知セル(追補77 §D)==')
    ok = True
    for pp, a in anchors.items():
        g = [r for r in rows if r['page_path'] == pp]
        if not g:
            print(f'  **錨の頁が答の中に無い: ...{pp[-28:]}**'); ok = False; continue
        r = g[0]
        side = r['use'].startswith(a['verdict'])
        byword = r['根拠'].startswith('作者の言明')
        print(f'  ...{pp[-26:]:28s} -> {r["use"]:24s} '
              f'側 {"PASS" if side else "**FAIL**"} / '
              f'根拠が作者の言明 {"PASS" if byword else "**FAIL**"}')
        ok &= side and byword
    diff = [r for r in rows if r['use'] != r['集計だけの判定']]
    print(f'  **集計だけの判定と最終の判定が違う頁 {len(diff)}**  '
          f'{"PASS" if diff else "**FAIL**(書き直しが何も変えていない)"}')
    ok &= bool(diff)
    print(f'\n  **既知セル {"すべて PASS" if ok else "**FAIL**。他を読まない"}**')
    if not ok:
        return 1

    print('\n== 書き直しで動いた頁(理由別)==')
    mv = collections.Counter(
        (r['集計だけの判定'], r['use']) for r in diff)
    for (a_, b_), n in mv.most_common():
        print(f'  {a_:30s} -> {b_:28s} {n:>3}')

    und = [r for r in rows if r['use'] == 'undecided']
    if und:
        print(f'\n== `undecided`({len(und)} 頁。**作者に聞く対象**)==')
        for r in und:
            print(f'  {r["page_path"].split("manuscripts" + chr(92))[-1]:46s} {r["種類"][:70]}')
    with open(OUT / 'use_from_layer_answers.csv', 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r['page_path']))
    print(f'\nwrote {OUT / "use_from_layer_answers.csv"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
