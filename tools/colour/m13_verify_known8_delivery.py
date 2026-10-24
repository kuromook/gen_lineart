"""M13: 依頼3 の**部分納品(既知セルの8ページ)を、抽出側の報告ではなく中身で
確かめる**。`note_m_request3_known8_20261024.md`(2026-10-24 14:08)が主張した数を
こちらで数え直す。

名前は点検ではない(2026-10-23 の教訓)。抽出側は「706/706 一致・不一致0」と
報告しているが、**報告は点検ではない**。m5 と同じ `verify_*` の形で、
**こちらが数えた数と報告の数を並べて印字する**。

主張(報告側):
  V1 1,005行 × 64列
  V2 8ページ(v2 一覧の 1〜8行 = 登録集合)
  V3 パイロットの manifest の **706層 × 20列が全一致、不一致 0**
  V4 残る 299行 = フォルダー **173** + 描画画素0の層 **126**
  V5 126 はパイロットの `skipped`(「1画素も描かれていない」)126 と同数
  V6 `page_id` の先頭は全部 `000`(= 参考。結合は `page_path`)
  V7 用紙の行は `drawn_ratio` = 1.0(ラスタの被覆ではない)

**この8ページは登録集合**なので、ここから腕の主張は出ない。これは
**配達の既知セル**(規律2: 新しい測定は既知セルを1つ含める)である。
"""
import ast
import csv
import json
import sys
from pathlib import Path

CSV = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_known8.csv')
MANIFEST = Path('/home/sh1/deepl/lineart/inbox/colour_layers_manifest_pilot.json')

# 報告された数。こちらが数えた数と並べて印字する。
CLAIM = {'rows': 1005, 'cols': 64, 'pages': 8, 'matched': 706, 'mismatch': 0,
         'folders': 173, 'zero_drawn': 126, 'skipped': 126}

SAT_COLS = [f'sat_{i*8:03d}_{i*8+7:03d}' for i in range(32)]
# 比較する20列(報告が照合したと書いている列)
PAIRS = [('name', 'name'), ('ancestors', 'ancestors'),
         ('layer_type', 'layer_type'), ('visibility', 'visibility'),
         ('opacity', 'opacity'), ('has_mask', 'has_mask'),
         ('composite', 'composite'), ('clip', 'clip'),
         ('parent_id', 'parent_id'), ('order_in_parent', 'order_in_parent'),
         ('formats', 'formats'), ('draw_color_enable', 'draw_color_enable'),
         ('draw_color_main', 'draw_color_main'),
         ('draw_color_sub', 'draw_color_sub'), ('drawn_ratio', 'drawn_ratio'),
         ('sat_mean', 'sat_mean'), ('colours_q16', 'colours_q16'),
         ('top_colour_share', 'top_colour_share'),
         ('top_colour_q16', 'top_colour_q16')]


def norm(v):
    """CSV の文字列と JSON の値を同じ形にする。"""
    if isinstance(v, str):
        t = v.strip()
        if t == '':
            return None
        if t in ('True', 'False'):
            return t == 'True'
        try:
            return ast.literal_eval(t)
        except (ValueError, SyntaxError):
            return t
    return v


def same(a, b, tol=5e-6):
    a, b = norm(a), norm(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    return a == b


def verify():
    with open(CSV, encoding='utf-8') as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames
        rows = list(rdr)
    man = json.load(open(MANIFEST, encoding='utf-8'))['pages']
    by_src = {p['source']: p for p in man}

    got = {'rows': len(rows), 'cols': len(cols),
           'pages': len({r['page_path'] for r in rows}),
           'folders': sum(1 for r in rows if norm(r['folder']) is True),
           'zero_drawn': 0, 'matched': 0, 'mismatch': 0,
           'skipped': sum(len(p['skipped']) for p in man)}

    bad, unseen = [], []
    for src in {r['page_path'] for r in rows}:
        if src not in by_src:
            unseen.append(src)
    for p in man:
        layers = {l['layer_id']: l for l in p['layers']}
        mine = [r for r in rows if r['page_path'] == p['source']]
        seen = set()
        for r in mine:
            lid = int(r['layer_id'])
            if norm(r['folder']) is True:
                continue
            if lid not in layers:
                got['zero_drawn'] += 1
                continue
            l, seen = layers[lid], seen | {lid}
            diffs = []
            for c, k in PAIRS:
                if c == 'formats':
                    # CSV はスカラー文字列('5')、manifest は1要素のリスト(['5'])。
                    # どちらも8ページ全部で1要素なので、文字列の一覧にして比べる。
                    if [r[c]] != [str(x) for x in l[k]]:
                        diffs.append((c, r[c], l[k]))
                elif not same(r[c], l[k]):
                    diffs.append((c, r[c], l[k]))
            hist = [int(x) if r[c] != '' else 0 for c, x in
                    ((c, r[c] or 0) for c in SAT_COLS)]
            if hist != l['sat_hist']:
                diffs.append(('sat_hist', 'differs', ''))
            if diffs:
                got['mismatch'] += 1
                bad.append((p['page'], lid, diffs))
            else:
                got['matched'] += 1
        miss = set(layers) - seen
        if miss:
            bad.append((p['page'], 'MISSING', sorted(miss)))

    print('== V: 報告の数 と こちらが数えた数 ==')
    for k in ('rows', 'cols', 'pages', 'matched', 'mismatch', 'folders',
              'zero_drawn', 'skipped'):
        mark = 'OK' if got[k] == CLAIM[k] else '**DIFFERS**'
        print(f'  {k:12s} 報告 {CLAIM[k]:>6} / 実測 {got[k]:>6}  {mark}')
    if unseen:
        print(f'  manifest に無い page_path: {unseen}')
    pid = {r['page_id'][:3] for r in rows}
    print(f'  page_id の先頭3文字: {sorted(pid)} (V6: 全部 000 なら参考扱いで正しい)')
    paper = [r for r in rows if r['layer_type'] == '1584']
    print(f'  用紙の行 {len(paper)} / drawn_ratio: '
          f'{sorted({r["drawn_ratio"] for r in paper})} (V7)')
    if bad:
        print(f'\n不一致 {len(bad)} 件(先頭10件):')
        for b in bad[:10]:
            print('  ', b)
    ok = all(got[k] == CLAIM[k] for k in CLAIM) and not bad and not unseen
    print(f'\n配達の既知セル: {"PASS" if ok else "FAIL"}')
    print('この8ページは登録集合なので、ここから腕の主張は出ない。')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(verify())
