# -*- coding: utf-8 -*-
"""L25 の集計: 作者の層ごとの答で、**3つの定数がどれだけずれているか**を印字する。

事前登録 `doc/work_log.md § 2026-10-25 L25 事前登録` と、その更新(語彙・無作為階級・既知セル)。
**ここで定数は決めない。** 92層で棒は立たない。出すのは**ずれの大きさ**まで。

読む順も登録どおり:
  1 既知セル(同じ層の再出題)—— **答が再現しなければ、境界の話は始められない**
  2 錨(境界から遠い層)—— 作業自体が難しいのかを見る
  3 無作為の階級 —— **境界から離れた所の誤り率**(これが無いと母集団の話ができない)
  4 3つの定数ごとの食い違い

  python tools/evaluation/l25_calibrate.py
"""
import csv, json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, HERE)
import l14_known_cells as K

Q = '/home/sh1/deepl/lineart-fill-region/results/l25_layer_ui/queue.json'
LAB = '/home/sh1/deepl/lineart-fill-region/labels/layer_kind.jsonl'
OUT = '/home/sh1/deepl/lineart-fill-region/results/l25_layer_ui'
# 定数ごとの「通す側」の意味。**この表が、何と何を比べているかの正本**
CONST = [
    ('preview_dark >= 0.5', 'L', 'preview_dark', lambda v: v is not None and v >= 0.5,
     '通す = L の母集団に入れる'),
    ('keep4 >= 0.95', 'L', 'keep4', lambda v: v is not None and v >= 0.95,
     '通す = ベタの形(FILL)として扱う'),
    ('black_among_drawn <= 0.98', '抽出側', 'black_among_drawn',
     lambda v: v is not None and v <= 0.98, '通す = 線画ラスタに入れる'),
]
# 役割の読み替え(**形の軸と役割の軸を分けたまま**)
LINEISH = ('線(完成の線画)',)
FILLISH = ('ベタ(完成の塗り)',)
MIXED = ('線とベタが混じっている',)
ROUGH = ('下絵・ラフ(ペンでも。完成の絵ではない)',)
NOTART = ('絵ではない(マスク・紙・下地・トーン)',)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load():
    q = json.load(open(Q))
    byo = {r['order']: r for r in q['queue']}
    last = {}
    for ln in open(LAB):
        r = json.loads(ln)
        last[r['order']] = r
    inv = {(r['page_id'], r['layer_id']): r
           for r in csv.DictReader(open(K.INV, encoding='utf-8-sig'))}
    k4 = {(r['page_id'], r['layer_id']): r for r in csv.DictReader(open(K.KEEP4))}
    rows = []
    for o, r in sorted(last.items()):
        it = byo[o]
        key = (it['pid'], str(it['lid']))
        iv = inv.get(key, {})
        kk = k4.get(key, {})
        ans = r['word'] if r['choice'] == 'word' else (
            r['free'].strip() if r['choice'] == 'free' and r.get('free') else r['choice'])
        rows.append(dict(order=o, pid=it['pid'], lid=it['lid'], work=it['work'],
                         stratum=it['why'], ans=ans, ms=r['ms'], area=it['area'],
                         name=iv.get('name', ''),
                         preview_dark=fnum(iv.get('preview_dark')),
                         black_among_drawn=fnum(iv.get('black_among_drawn')),
                         keep4=fnum(kk.get('keep4'))))
    return rows, q


def main():
    rows, q = load()
    print(f'答えた層 {len(rows)} / 行列 {q["n"]}  '
          f'中央 {sorted(r["ms"] for r in rows)[len(rows)//2]/1000:.1f}s')

    print('\n[0] 答の分布 (全体)')
    for a, c in Counter(r['ans'] for r in rows).most_common():
        print(f'  {c:>4}  {a}')

    # --- 1. 既知セル: 同じ層をもう一度
    print('\n[1] 既知セル: 同じ層の再出題 —— **答が再現しなければ、境界の話は始められない**')
    byl = defaultdict(list)
    for r in rows:
        byl[(r['pid'], r['lid'])].append(r)
    rep = {k: v for k, v in byl.items() if len(v) > 1}
    same = sum(1 for v in rep.values() if len({x['ans'] for x in v}) == 1)
    print(f'  2回出した層 {len(rep)}  **同じ答 {same}**  違う答 {len(rep)-same}')
    for k, v in rep.items():
        mark = 'ok' if len({x['ans'] for x in v}) == 1 else '**違う**'
        print(f'    {k[0]} L{k[1]}: ' + ' / '.join(x['ans'] for x in v) + f'  {mark}')
    if len(rep) < 3:
        print('  (**再出題が少ないので、再現性は数として言わない**)')

    # --- 2/3. 階級ごと
    print('\n[2] 階級ごとの答 (**階級は作者に見せていない**)')
    order = ['A_preview_dark', 'B_keep4', 'C_bad_max', 'D_anchor', 'E_random']
    cats = [('線', LINEISH), ('ベタ', FILLISH), ('線+ベタ', MIXED),
            ('下絵', ROUGH), ('絵でない', NOTART)]
    print(f'  {"階級":<16}{"層":>4}' + ''.join(f'{n:>9}' for n, _ in cats)
          + f'{"その他":>9}{"分からない":>11}')
    for sid in order:
        s = [r for r in rows if r['stratum'].startswith(sid)]
        if not s:
            continue
        line = f'  {sid:<16}{len(s):>4}'
        got = 0
        for n, ws in cats:
            c = sum(1 for r in s if r['ans'] in ws)
            got += c
            line += f'{c:>9}'
        unk = sum(1 for r in s if r['ans'] == '分からない')
        line += f'{len(s)-got-unk:>9}{unk:>11}'
        print(line)
    print('  **E_random が母集団の構成**。他の階級はそこからの偏りとして読む')

    # --- 4. 定数ごとの食い違い
    print('\n[3] 定数ごとの食い違い (**この表が本題**)')
    for label, who, col, keep, mean in CONST:
        print(f'\n  --- {label}  ({who}。{mean}) ---')
        s = [r for r in rows if r[col] is not None]
        print(f'  値のある層 {len(s)} / {len(rows)}')
        tab = defaultdict(Counter)
        for r in s:
            tab['通す' if keep(r[col]) else '落とす'][r['ans']] += 1
        for side in ('通す', '落とす'):
            tot = sum(tab[side].values())
            if not tot:
                continue
            print(f'    [{side}] {tot} 層')
            for a, c in tab[side].most_common():
                print(f'      {c:>4} ({c/tot:.3f})  {a}')
        # **濃縮を分けて出す** —— 境界の階級は意図的に濃くしてあるので、
        # 全体の比を母集団の誤り率として読んではいけない(規律1)
        own = {'preview_dark': 'A_preview_dark', 'keep4': 'B_keep4',
               'black_among_drawn': 'C_bad_max'}[col]
        for sid, note in ((own, 'この定数の境界ぎわ。**意図的に濃くした階級**'),
                          ('E_random', '**無作為**。母集団の比に近いのはこちら')):
            t = [r for r in s if r['stratum'].startswith(sid)]
            if not t:
                continue
            kp = [r for r in t if keep(r[col])]
            dr = [r for r in t if not keep(r[col])]
            print(f'    [{sid}] {len(t)} 層 ({note})')
            for nm, u in (('通す', kp), ('落とす', dr)):
                if not u:
                    continue
                print(f'      {nm} {len(u):>3}: '
                      + '  '.join(f'{a}:{c}' for a, c in Counter(
                          r['ans'] for r in u).most_common(4)))
        # **ずれ**の言い方は定数ごとに違う。意味を書いてから数える
        if col == 'preview_dark':
            bad = sum(tab['落とす'][a] for a in list(LINEISH) + list(FILLISH) + list(MIXED))
            n0 = sum(tab['落とす'].values())
            bad2 = sum(tab['通す'][a] for a in NOTART)
            n1 = sum(tab['通す'].values())
            print('    (下の2つは**濃縮された標本の上の比**。母集団の誤り率ではない)')
            print(f'    **落としている中の「完成の絵」 {bad}/{n0}'
                  f'{"" if not n0 else f" = {bad/n0:.3f}"}**')
            print(f'    通している中の「絵ではない」 {bad2}/{n1}'
                  f'{"" if not n1 else f" = {bad2/n1:.3f}"}')
        elif col == 'keep4':
            a1 = sum(tab['通す'][a] for a in LINEISH)
            n1 = sum(tab['通す'].values())
            a0 = sum(tab['落とす'][a] for a in FILLISH)
            n0 = sum(tab['落とす'].values())
            print('    (下の2つは**濃縮された標本の上の比**。母集団の誤り率ではない)')
            print(f'    **ベタの形として通したが作者は「線」 {a1}/{n1}'
                  f'{"" if not n1 else f" = {a1/n1:.3f}"}**')
            print(f'    **線の形として落としたが作者は「ベタ」 {a0}/{n0}'
                  f'{"" if not n0 else f" = {a0/n0:.3f}"}**')
        else:
            a1 = sum(tab['通す'][a] for a in list(NOTART) + list(ROUGH))
            n1 = sum(tab['通す'].values())
            a0 = sum(tab['落とす'][a] for a in LINEISH)
            n0 = sum(tab['落とす'].values())
            print('    (下の2つは**濃縮された標本の上の比**。母集団の誤り率ではない)')
            print(f'    **線画ラスタに通したが「絵ではない/下絵」 {a1}/{n1}'
                  f'{"" if not n1 else f" = {a1/n1:.3f}"}**')
            print(f'    **落としたが作者は「線」 {a0}/{n0}'
                  f'{"" if not n0 else f" = {a0/n0:.3f}"}**')

    with open(f'{OUT}/layer_truth.csv', 'w') as fh:
        ks = ['order', 'work', 'pid', 'lid', 'name', 'stratum', 'ans', 'area',
              'preview_dark', 'keep4', 'black_among_drawn', 'ms']
        fh.write(','.join(ks) + '\n')
        for r in rows:
            fh.write(','.join(f'"{r.get(k)}"' if k == 'name' else str(r.get(k, ''))
                              for k in ks) + '\n')
    print(f'\n-> {OUT}/layer_truth.csv')
    print('**92層で定数は決めない。** 出したのはずれの大きさまで')


if __name__ == '__main__':
    main()
