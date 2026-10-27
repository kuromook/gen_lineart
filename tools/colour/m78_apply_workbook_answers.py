# -*- coding: utf-8 -*-
"""m78: **作者が記入した版の表(Excel)を札に反映する**(裁定3 の回答)。

材料は作者が記入して返した `data/version_workbook/returned/` の1本だけ。
**`data/` は追跡外**なので、この器具は**追跡される側に答だけを写す**
(`results/m78_workbook_applied/`)。

**答の読み方**(workbook の `読み方` シートに**書いてあるのは「○で選ぶ」だけ**。
下の3つのうち 2 と 3 は**表に書いていない形**なので、この器具が読みを宣言する):

  | 塊の中の ○ | 読み |
  |---|---|
  | 1つ | **○の行が代表。同じ塊の他の行は採点しない**(作者の判断) |
  | **2つ**(備考「全然別物」) | **塊を解く。両方が独立した頁** —— 重複ではなかった |
  | **0個** | **未決。札を動かさない。** 書かれた理由は記録する |

**根拠の強さを列に分ける**(規律9 の形。選別は測定である):
  `作者が選んだ` / **`作者が「違いがわからなかった」と書いた上で選んだ`**
  —— 後者は**内容の判断ではない**。あとで区別できるように残す。

**既知セル**:
  - 塊 56 / 行 119(表を作ったときの数)
  - 全部の行が `version_roles_current.csv` の 872頁に結べる(**フルパスで結合**)
  - **裁定3 で止まっていた 6頁が全部この表に在る**

**主値 = `採点する` と数える頁の数**(いまの 694)。
**帰無値 = 動かない。** 動いたら「複製が畳まれていなかった」側の証拠。
"""
import collections
import csv
import os
import sys
from pathlib import Path

import openpyxl

# **表は複数。後の表が前の表を上書きする**(= 答え直し。m63 と同じ規則)。
# 2本目は ○ が入らなかった3塊だけを切り出して出し直した表(作者が記入して返した)
WB = [Path('data/version_workbook/returned/version_decide_20261027_answered.xlsx'),
      Path('data/version_workbook/returned/version_decide_rest3_20261027_answered.xlsx')]
ROLES = Path('results/m76_version_ruling/version_roles_current.csv')
OUT = Path('results/m78_workbook_applied'); OUT.mkdir(parents=True, exist_ok=True)
# **原稿の根のフォルダーは書かない。** 追跡される器具に書くと、それ自身が公開される
# (`check_public_names.py` が「新しいフルパス」として拾った。2026-10-27)。
# **札の表から導く** —— 表は追跡外で、手元では今までどおり辿れる
MARK = {'〇', '○', 'o', 'O', '◯', '●'}
WEAK = '違いがわからなかった'
STALLED = '採点する(同年代。作者に問い合わせ中。裁定3)'

# **版の軸と内容の軸は別の軸である。**(この器具を最初に書いたときに混ぜた)
# 作者に出した問いは「**どちらを使うか**」= 版の軸だけ。
# 「下絵のみ」「使わないと述べた」は**内容の軸**で、版の答では覆らない ——
# 覆すと、作者の版の答で**母集団が広がる**(曝露や母集団を広げるのも棒を動かすこと)。
CONTENT_EXCLUDE = ('採点しない(下絵のみ',
                   '採点しない(作者が使わないと述べた')


def root_of(roles):
    """札の表の `page_path` の共通の頭から、原稿の根のフォルダーを導く。"""
    pre = os.path.commonprefix([r['page_path'] for r in roles])
    return pre[:pre.rfind('\\') + 1]


def read_one(path, root):
    ws = openpyxl.load_workbook(path, data_only=True)['問い(同年代)']
    hdr = [c for c in next(ws.iter_rows(min_row=2, max_row=2, values_only=True))]
    ix = {h: i for i, h in enumerate(hdr) if h}
    ci, cc = ix['相対パス'], ix['塊']
    cm = [i for h, i in ix.items() if h.startswith('【判断】使う')][0]
    cn = [i for h, i in ix.items() if h.startswith('【判断】理由')][0]
    out, drop = [], collections.Counter()
    for r in ws.iter_rows(min_row=3, values_only=True):
        if r[cc] in (None, ''):
            drop['塊の列が空(塊の間の空行)'] += 1; continue
        m = (str(r[cm]).strip() if r[cm] not in (None, '') else '')
        if m and m not in MARK:
            drop[f'印が読めない({m!r})'] += 1
        out.append(dict(cluster=str(r[cc]), page_path=root + str(r[ci]),
                        rel=str(r[ci]), mark=(m in MARK),
                        note=(str(r[cn]).strip() if r[cn] else '')))
    return out, drop


def read_workbook(root):
    """**複数の表を順に読み、同じ頁は後の表で上書きする**(規律9: 差分を印字)。"""
    print('== 材料 ==')
    rows, drop = {}, collections.Counter()
    order = []
    for wb in WB:
        got, d = read_one(wb, root)
        over = sum(1 for w in got if w['page_path'] in rows)
        print(f'  {wb.name}  {len(got)} 行  (うち前の表を上書き {over})')
        drop.update(d)
        for w in got:
            if w['page_path'] not in rows:
                order.append(w['page_path'])
            rows[w['page_path']] = w
    print('== 母集団の差分(理由別。規律9)==')
    for k, v in drop.most_common():
        print(f'  {k:34s} -{v}')
    cl = collections.defaultdict(list)
    for pp in order:
        cl[rows[pp]['cluster']].append(rows[pp])
    print(f'  **塊 {len(cl)} / 行 {sum(len(v) for v in cl.values())}**'
          f'  (表を作ったときは 56 / 119)')
    return cl


def main():
    roles = list(csv.DictReader(open(ROLES, encoding='utf-8-sig')))
    by = {r['page_path']: r for r in roles}
    root = root_of(roles)
    cl = read_workbook(root)
    print(f'  札の表 {len(roles)} 頁  {ROLES}')

    print('\n== 既知セル ==')
    ok = (len(cl) == 56 and sum(len(v) for v in cl.values()) == 119)
    print(f'  塊 56 / 行 119           {"PASS" if ok else "**FAIL**"}')
    miss = [w['page_path'] for v in cl.values() for w in v if w['page_path'] not in by]
    print(f'  札の表に結べない行 {len(miss)}  {"PASS" if not miss else "**FAIL**"}')
    for m in miss[:5]:
        print(f'    {m}')
    stalled = [r['page_path'] for r in roles if r['use'] == STALLED]
    inwb = {w['page_path'] for v in cl.values() for w in v}
    st_in = [p for p in stalled if p in inwb]
    print(f'  裁定3 で止まっていた頁 {len(stalled)} / この表に在る {len(st_in)}  '
          f'{"PASS" if len(st_in) == len(stalled) else "**FAIL**"}')
    ok &= (not miss) and len(st_in) == len(stalled)
    print(f'  **既知セル {"すべて PASS" if ok else "FAIL。先を読まない"}**')
    if not ok:
        return 1

    # ---- 塊ごとに読む ----
    acts, kinds = [], collections.Counter()
    for cid, rows in cl.items():
        marked = [w for w in rows if w['mark']]
        weak = any(WEAK in w['note'] for w in rows)
        if len(marked) == 1:
            kinds['○が1つ(代表が決まった)'] += 1
            rep = marked[0]
            grade = (f'作者が「{WEAK}」と書いた上で選んだ' if weak else '作者が選んだ')
            for w in rows:
                acts.append(dict(
                    cluster=cid, page_path=w['page_path'], rel=w['rel'],
                    読み=('代表' if w is rep else '落とす'),
                    根拠の強さ=grade, 作者の備考=w['note'],
                    代表=rep['rel'] if w is not rep else ''))
        elif len(marked) >= 2:
            kinds['○が2つ以上(塊を解く)'] += 1
            for w in rows:
                acts.append(dict(cluster=cid, page_path=w['page_path'], rel=w['rel'],
                                 読み='塊を解く', 根拠の強さ='作者が選んだ',
                                 作者の備考=w['note'], 代表=''))
        else:
            kinds['○が0個(未決。札を動かさない)'] += 1
            for w in rows:
                acts.append(dict(cluster=cid, page_path=w['page_path'], rel=w['rel'],
                                 読み='未決', 根拠の強さ='',
                                 作者の備考=w['note'], 代表=''))

    # ---- 版の答を、内容の軸を壊さずに合成する ----
    for a in acts:
        old = by[a['page_path']]['use']
        a['old_use'] = old
        a['軸'] = ('内容' if old.startswith(CONTENT_EXCLUDE)
                 else '版' if old.startswith('採点しない') else '採点側')
        if a['読み'] == '落とす':
            a['new_use'] = '採点しない(同年代。作者が別の方を選んだ。裁定3)'
            if a['軸'] == '内容':
                a['new_use'] += f' / 元の理由も残る: {old}'
            a['合成'] = '落とした'
        elif a['読み'] == '塊を解く':
            a['new_use'] = '採点する(塊を解いた。独立した頁。作者「全然別物」)'
            a['合成'] = '塊を解いた'
        elif a['読み'] == '未決':
            a['new_use'] = ''
            a['合成'] = '動かさない'
        elif a['軸'] == '内容':
            # **版では選ばれたが、内容の理由での除外は生きている**
            a['new_use'] = old
            a['合成'] = '**内容の軸は動かさない**(版だけ作者が選んだ)'
        elif a['軸'] == '版':
            # 作者が、こちらの版の読みを覆した
            a['new_use'] = '採点する(同年代。作者が選んだ。裁定3)'
            a['合成'] = '**以前の版の読みを作者が覆した**'
        elif old == STALLED:
            a['new_use'] = '採点する(同年代。作者が選んだ。裁定3)'
            a['合成'] = '暫定の札が確定した'
        elif '両方' in old:
            # 以前の読みが「**両方が成果物**」と書いてある行で、相手が落ちた。
            # 札だけ直して古い文を残すと、台帳の中で食い違う(規律3)
            a['new_use'] = (f'採点する(同年代。作者が選んだ。裁定3)'
                            f' / 前の読み「{old}」は作者が覆した')
            a['合成'] = '**前の読み「両方が成果物」を作者が覆した**'
        else:
            # 既に採点側。**修飾(別行で採点・1枚イラスト)をそのまま残す**
            a['new_use'] = old
            a['合成'] = '採点側のまま(修飾を残す)'

    print('\n== 塊の読み(理由別)==')
    for k, v in kinds.most_common():
        print(f'  {k:34s} {v:>3}')

    print('\n== 合成(版の軸と内容の軸)==')
    for k, v in collections.Counter(
            (a['読み'], a['軸'], a['合成']) for a in acts).most_common():
        print(f'  {k[0]:6s} 元の軸={k[1]:4s} {k[2][:44]:46s} {v:>3}')

    print('\n== 根拠の強さ(代表が決まった行だけ)==')
    for k, v in collections.Counter(
            a['根拠の強さ'] for a in acts if a['読み'] == '代表').most_common():
        print(f'  {k:46s} {v:>3}')

    print('\n== ○が付かなかった塊(**作者に戻す**)==')
    for cid, rows in cl.items():
        if not any(w['mark'] for w in rows):
            why = ' / '.join(sorted({w['note'] for w in rows if w['note']})) or '(備考なし)'
            print(f'  塊 {cid:>4}  {why}')
            for w in rows:
                print(f'        {w["rel"]}')

    # ---- 札に反映 ----
    # **採点側の数え方を2つ併記する**(追補108 §D。`別行で採点` は採点しない側ではない)
    def cnt(rs):
        a = sum(1 for r in rs if r['use'].startswith('採点する'))
        b = a + sum(1 for r in rs if r['use'].startswith('別行で採点'))
        return a, b

    before = cnt(roles)
    moved = []
    for a in acts:
        if not a['new_use']:
            continue
        r = by[a['page_path']]
        if r['use'] != a['new_use']:
            moved.append(a)
        r['use'] = a['new_use']
        r['decided_by'] = 'author_workbook_20261027'
        if a['読み'] == '落とす':
            r['version_role'] = 'superseded'
            r['superseded_by'] = root + a['代表']
        elif a['読み'] == '塊を解く':
            r['version_role'] = 'unrelated'; r['superseded_by'] = ''
        else:
            r['version_role'] = 'final'; r['superseded_by'] = ''
    after = cnt(roles)

    print('\n== 主値(採点側と数える頁。**数え方を2つ併記**)==')
    print(f'  A(`採点する` で始まる札)      前 {before[0]} -> 後 **{after[0]}**'
          f'  差 {after[0] - before[0]}')
    print(f'  B(A + `別行で採点`)          前 {before[1]} -> 後 **{after[1]}**'
          f'  差 {after[1] - before[1]}')
    print(f'  採点しない(B の外)            {len(roles) - before[1]} -> {len(roles) - after[1]}')
    print(f'  **帰無値は「動かない」。{"動いた" if after[0] != before[0] else "動かなかった"}**')
    left = [r for r in roles if r['use'] == STALLED]
    print(f'  裁定3 の暫定の札 {len(stalled)} -> **{len(left)}**')

    print(f'\n== 札が動いた頁 {len(moved)} ==')
    for k, v in collections.Counter((m['old_use'], m['new_use']) for m in moved).most_common():
        print(f'  {v:>3}  {k[0][:40]:42s} -> {k[1][:46]}')

    with open(OUT / 'version_roles_current.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(roles[0])); w.writeheader(); w.writerows(roles)
    for name, data in (('workbook_actions.csv', acts), ('moved.csv', moved)):
        if data:
            with open(OUT / name, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(f'\nwrote {OUT}/ (version_roles_current.csv / workbook_actions.csv / moved.csv)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
