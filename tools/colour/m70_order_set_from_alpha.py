#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追補91 —— 追補88 の登録どおり、来た列で発注の集合を決める。**

材料: `../lineart/inbox/faint_alpha_905_20261027.csv`(905頁 / 62,105層)。

**追補88 の登録した順序を守る**:
  1. **まず `body_init == 255` の一部保存を数える**(値が下限になる層)
  2. 272層 / 229層を**5分類に割る**(理由別。残り 0)
  3. **空だと示せた層だけ外す**。残りが発注の集合

**除外の規則(追補88 の登録)**: `alpha_ge1_ratio == 0` かつ `body_init == 0`
かつ `body_stored == 0`。

**ここで1つ締める(登録より厳しい側へ)**: **`rendered` が偽の層は外さない。**
描けていない層の 0 は**測定ではない**ので、「空であることが示せた」に当たらない
(追補88 の「0 は空ではない」そのもの)。**この締めは候補を増やす向きにしか働かない**
—— 緩める方向ではないので、棒を動かしていない。

  <venv>/python tools/colour/m70_order_set_from_alpha.py
"""
import collections
import csv
import json
import pathlib
import sys

SRC = pathlib.Path('../lineart/inbox/faint_alpha_905_20261027.csv')
MISS = pathlib.Path('results/m68_sample_gap/missing_layers_undecided38.csv')
PAGES = pathlib.Path('results/alias/occluder_check_pages.json')
LEDGER = pathlib.Path('../lineart/inbox/m_request6b_layer_fields_872pages.csv')
R5 = pathlib.Path('/home/sh1/deepl/lineart/data/incoming_manuscript/'
                  'colour_request5/colour_layers_pages.jsonl')
OUT = pathlib.Path('results/m70_order_set')


def truthy(v):
    return str(v).strip().lower() not in ('', '0', 'false', 'none')


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def case_of(r):
    """納品便 §2 の5分類。**残り 0 にする**(規律9)。"""
    bi, bb, bs = num(r['body_init']), num(r['body_blocks']), num(r['body_stored'])
    if bi == 255 and bs == 0:
        return '1 255・保存0(0 と出るが全面)'
    if bi == 0 and bs == 0:
        return '2 0・保存0(空)'
    if bi == 0 and bs and bs > 0:
        return '3 0・保存>0(値どおり)'
    if bi == 255 and bb is not None and bs == bb:
        return '4 255・全部保存(値どおり)'
    if bi == 255 and bs and bb and 0 < bs < bb:
        return '5 255・一部保存(**下限**)'
    return '9 どれでもない'


def main():
    for p in (SRC, MISS, PAGES):
        if not p.exists():
            raise SystemExit(f'**材料が無い: {p}。走らせない**')
    pg = json.loads(PAGES.read_text(encoding='utf-8'))
    p45 = set(pg['p45'])
    rows = list(csv.DictReader(SRC.open(encoding='utf-8')))
    tbl = {(r['page_path'], r['layer_id']): r for r in rows}
    print(f'材料 {SRC}  層 {len(rows)}  頁 {len({r["page_path"] for r in rows})}')
    print(f'  エラーの行 {sum(1 for r in rows if truthy(r["error"]))}'
          f'  鍵の重複 {len(rows) - len(tbl)}')

    # ---- 被覆。欠けたら読まない ----
    pages = {r['page_path'] for r in rows}
    print(f'  45頁の被覆 {len(p45 & pages)}/{len(p45)}')
    if len(p45 & pages) != len(p45):
        print('  **被覆が欠けている。結果を読まない**')
        return 1

    # ---- 取り込みの点検: 納品 manifest の drawn_ratio と一致するか ----
    # **用紙は別に見る。** 納品側は用紙を描画色で全面に塗り直し、この表は画素データの
    # ままなので、**読み手の差ではなく定義の差**である(納品便 §3)。
    # **公差を緩めない。用紙は向きで見る** —— 下限なら必ず `新表 <= 納品` になる
    same = diff = absent = 0
    paper_rows = []
    with R5.open(encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            for l in rec['layers']:
                k = (rec['source'], str(l['layer_id']))
                if k not in tbl:
                    absent += 1
                    continue
                a, b = num(l.get('drawn_ratio')), num(tbl[k]['drawn_ratio'])
                if 'paper' in str(l.get('formats', '')).lower():
                    paper_rows.append((a, b, tbl[k]))
                    continue
                if a is None or b is None or abs(a - b) <= 1e-6:
                    same += 1
                else:
                    diff += 1
    print(f'  取り込み(用紙以外): `drawn_ratio` 一致 {same} / 違う {diff}'
          f' / 表に無い {absent}')
    pdiff = [(a, b, t) for a, b, t in paper_rows
             if a is not None and b is not None and abs(a - b) > 1e-6]
    bad_dir = [1 for a, b, t in pdiff if not b <= a + 1e-6]
    bad_case = [1 for a, b, t in pdiff if not case_of(t).startswith('5')]
    print(f'  用紙 {len(paper_rows)}層: 違う {len(pdiff)}'
          f'  **向きが逆(新表 > 納品) {len(bad_dir)}**'
          f'  **分類5 でない {len(bad_case)}**')
    print('    用紙の差は**定義の差**(納品側は描画色で塗り直し = 1.0、'
          'この表は画素のまま = **下限**)。**公差は緩めていない**')
    if diff or absent or bad_dir or bad_case:
        print('  **点検が落ちた。先に理由を調べる。発注しない**')
        return 1

    # ---- 1. まず一部保存の 255 を数える(追補88 の登録した順序)----
    sub45 = [r for r in rows if r['page_path'] in p45]
    p255 = [r for r in sub45 if case_of(r).startswith('5')]
    print(f'\n== 1. **まず数える**: 45頁の `body_init == 255` の一部保存 ==')
    print(f'  **{len(p255)}層 / {len({r["page_path"] for r in p255})}頁**'
          f'  (45頁の層 {len(sub45)})')
    print('  **この層の `drawn_ratio`・`alpha_ge1_ratio` はどの値も下限である**'
          '(追補84)')

    # ---- 2. 272層 / 229層を5分類に割る ----
    # 台帳(フォルダー以外)− 納品 = 未納品
    led = [r for r in csv.DictReader(LEDGER.open(encoding='utf-8'))
           if r['page_path'] in p45 and not truthy(r['folder'])]
    deliv = set()
    with R5.open(encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            for l in rec['layers']:
                deliv.add((rec['source'], str(l['layer_id'])))
    missing = [r for r in led if (r['page_path'], r['layer_id']) not in deliv]
    cand = [r for r in csv.DictReader(MISS.open(encoding='utf-8'))]
    print(f'\n== 2. 5分類に割る(**理由別。残り 0**)==')
    for lab, keys in (('未納品 272層', [(r['page_path'], r['layer_id'])
                                        for r in missing]),
                      ('発注候補 229層', [(r['page_path'], r['layer_id'])
                                          for r in cand])):
        got = [tbl[k] for k in keys if k in tbl]
        print(f'  {lab}: 表に在る {len(got)} / 無い {len(keys) - len(got)}')
        c = collections.Counter(case_of(r) for r in got)
        for k in sorted(c):
            print(f'    {k:34s} {c[k]:>5}')
        print(f'    {"合計":34s} {sum(c.values()):>5}'
              f'   **残り {len(got) - sum(c.values())}**')

    # ---- 3. 空だと示せた層だけ外す ----
    print('\n== 3. **空だと示せた層だけ外す**(追補88 の登録)==')
    # **この列は描画用 mipmap しか読まない。** だから**画素の置き場が他に在る層には
    # 使えない** —— 画像素材は `ResizableOriginalMipmap`(追補60/65。追補80・82 で
    # 「この層の判断には列を使わない」と既に登録済み)、ベクターは描画時にラスタ化。
    # **追補88 の規則2 はこれを書き落としていた**(§D に欠陥として記録した)。
    # **この締めは候補を増やす向きにしか働かない。**
    OTHER = ('has_image_material', 'has_text', 'has_gradation', 'has_vector')
    ledk = {(r['page_path'], r['layer_id']): r
            for r in csv.DictReader(LEDGER.open(encoding='utf-8'))}

    def other_store(k):
        l = ledk.get(k)
        if l is None:
            return ['台帳に無い']
        f = [c for c in OTHER if truthy(l[c])]
        if num(l['vector_objects'] or 0):
            f.append('vector_objects>0')
        return f

    keep, drop = [], []
    why = collections.Counter()
    for r in cand:
        k = (r['page_path'], r['layer_id'])
        t = tbl.get(k)
        if t is None:
            keep.append(r); why['表に無い(測れていない)'] += 1; continue
        a = num(t['alpha_ge1_ratio'])
        bi, bs = num(t['body_init']), num(t['body_stored'])
        store = other_store(k)
        if not truthy(t['rendered']):
            keep.append(r); why['`rendered` が偽(0 は測定ではない)'] += 1
        elif a == 0 and bi == 0 and bs == 0 and store:
            keep.append(r)
            why[f'**画素の置き場が他に在る(列の射程の外)**: {"|".join(store)}'] += 1
        elif a == 0 and bi == 0 and bs == 0:
            drop.append(r); why['**空だと示せた → 外す**'] += 1
        else:
            keep.append(r)
            why['空だと示せない(アルファ>0 / 255 / ブロック在り)'] += 1
    for k in sorted(why, key=lambda x: -why[x]):
        print(f'  {k:46s} {why[k]:>5}')
    print(f'\n  **発注する層 {len(keep)} / 外す層 {len(drop)}'
          f'  (候補 {len(cand)})**')
    pk = {r['page_path'] for r in keep}
    print(f'  **頁 {len(pk)} / 38**')
    per = collections.Counter(r['page_path'] for r in keep)
    v = sorted(per.values())
    if v:
        print(f'  1頁あたり: 中央 {v[len(v)//2]} / 最小 {v[0]} / 最大 {v[-1]}')
    # 薄い層(drawn 0 だがアルファ 1〜25 在り)が何層入ったか
    faint = [r for r in keep
             if (k := (r['page_path'], r['layer_id'])) in tbl
             and num(tbl[k]['drawn_ratio']) == 0
             and num(tbl[k]['alpha_1_25_px'] or 0) > 0]
    print(f'  うち**薄い層(`drawn_ratio` 0 だがアルファ 1〜25 在り)**: {len(faint)}'
          '  —— **墨で絞っていたら全部落ちていた層**')

    # ---- 棒: 追補65 の層に規則を当てたらどうなるか(**既知セル**)----
    print('\n== 棒(追補88 §D の既知セル): 追補65 の層に規則を当てる ==')
    k65 = next(((r['page_path'], r['layer_id']) for r in ledk.values()
                if 'slimeH1' in (r['name'] or '')
                for r in [r]), None)
    if k65 is None:
        print('  **台帳に見つからない。棒を点検できない**')
    else:
        t = tbl.get(k65)
        store = other_store(k65)
        would_drop_old = (t and truthy(t['rendered'])
                          and num(t['alpha_ge1_ratio']) == 0
                          and num(t['body_init']) == 0 and num(t['body_stored']) == 0)
        print(f'  旧規則(追補88 §C)で外れるか: **{bool(would_drop_old)}**'
              '  ← **外れるなら規則が不健全**(この層は原寸ミップマップに'
              '7016×6543 全面アルファ255 の完成した絵を持つ)')
        print(f'  新規則で外れるか: **{bool(would_drop_old and not store)}**'
              f'   (別の置き場: {"|".join(store) or "無し"})')
        print(f'  この層の頁は発注の対象か: '
              f'{k65[0] in {r["page_path"] for r in cand}}'
              '  ← **偽なら、頁が作者の言明で決まっているから**(追補88 §E)')

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'order_set.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(cand[0]) + ['case', 'alpha_ge1_ratio',
                                                          'body_init', 'body_stored',
                                                          'alpha_1_25_px'])
        w.writeheader()
        for r in keep:
            t = tbl.get((r['page_path'], r['layer_id']), {})
            w.writerow({**r, 'case': case_of(t) if t else '表に無い',
                        'alpha_ge1_ratio': t.get('alpha_ge1_ratio', ''),
                        'body_init': t.get('body_init', ''),
                        'body_stored': t.get('body_stored', ''),
                        'alpha_1_25_px': t.get('alpha_1_25_px', '')})
    print(f'\nwrote {OUT}/order_set.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
