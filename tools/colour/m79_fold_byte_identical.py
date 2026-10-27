# -*- coding: utf-8 -*-
"""m79: **全数のバイト同一の組を母集団から畳む**(追補108 §A の実装。登録は走る前に書いた)。

**裁定1**(作者): 頁でハッシュ値が同じなら同じファイル。
**代表は道の辞書順の先頭 —— 機械的な選び方で、内容の判断ではない**(追補104 と同じ)。

**検算(既知セル)**: 畳んだ後 **7,418** / 署名の塊で裁定1 が当たった **58塊** /
追補104 で落とした **6頁**がここでも落ちる。

**主値**: **代表の選び方で母集団の中身が変わる組の数**(辞書順 / 年代が古い方 / 新しい方)。
**帰無値 0組。**
**第二の主値**: 落とす側に在る頁のうち **`採点する` と数えていた頁**。**帰無値 0頁。**

**札の表は書き換えない**(母集団を動かす便と札を書き換える便を混ぜない)。
"""
import collections
import csv
import re
import sys
from pathlib import Path

ALL = Path('../lineart/inbox/request9_file_sha256_all_20261026.csv')
MINE = Path('../lineart/inbox/sha256_dup_clusters_20261026.csv')
ROLES = Path('results/m78_workbook_applied/version_roles_current.csv')

# **「採点側」の数え方を2つ併記する。**
# 主値 701 -> 694 -> 664 は **A**(札が `採点する` で始まる行)で数えてきたが、
# **`別行で採点(…)` は採点しない側ではない** —— 名前が量を取り違えている形。
# **どちらかに倒さず、両方印字する**(定数は役割ごとに引き直す)
def adopt_a(u):
    return u.startswith('採点する')


def adopt_b(u):
    return u.startswith('採点する') or u.startswith('別行で採点')
OUT = Path('results/m79_folded_population'); OUT.mkdir(parents=True, exist_ok=True)
YEAR = re.compile(r'\\(\d{4}[a-z]?)\\')


def year_of(p):
    m = YEAR.search(p)
    return m.group(1) if m else ''


def folder_of(p):
    return p[:p.rfind('\\')]


def main():
    rows = list(csv.DictReader(open(ALL, encoding='utf-8-sig')))
    print('== 材料 ==')
    print(f'  {ALL}  {len(rows)} 行')
    drop = collections.Counter()
    pop = []
    for r in rows:
        if r['f_read_error']:
            drop['読めなかった'] += 1; continue
        if not r['f_sha256']:
            drop['ハッシュが空'] += 1; continue
        pop.append(r)
    print('== 母集団の差分(理由別。規律9)==')
    for k, v in drop.most_common():
        print(f'  {k:24s} -{v}')
    print(f'  **母集団 {len(pop)}**'
          f'  (形式: ' + ' / '.join(f'{k} {v}' for k, v in
                                   collections.Counter(r['format'] for r in pop).most_common()) + ')')

    by = collections.defaultdict(list)
    by_sha = {r['page_path']: r['f_sha256'] for r in pop}
    for r in pop:
        by[r['f_sha256']].append(r)
    groups = {h: sorted(v, key=lambda r: r['page_path'])
              for h, v in by.items() if len(v) > 1}
    nfile = sum(len(v) for v in groups.values())
    print(f'\n== バイト同一の組 **{len(groups)}** / ファイル **{nfile}** ==')
    for k, v in sorted(collections.Counter(len(v) for v in groups.values()).items()):
        print(f'  大きさ {k:>3} の組 {v:>5}')

    # ---- 代表を3つの規則で選ぶ(主値)----
    def pick(v, how):
        if how == '辞書順':
            return v[0]
        key = (lambda r: (year_of(r['page_path']), r['page_path']))
        return sorted(v, key=key)[0 if how == '年代が古い方' else -1]

    disagree, spans = [], collections.Counter()
    for h, v in groups.items():
        ys = {year_of(r['page_path']) for r in v}
        fs = {folder_of(r['page_path']) for r in v}
        span = ('同じフォルダーの中' if len(fs) == 1
                else '同年代(フォルダーは別)' if len(ys) == 1 else '年代を跨ぐ')
        spans[span] += 1
        # **主値は「年代が1つに決まらない組」。**
        # 「辞書順の先頭 vs 末尾」で数えると、年代が同じ組でも必ず違う道になるので
        # **100% が出る —— それは規則の縮退で、測っていない**(T-M9)
        if len(ys) > 1:
            disagree.append(dict(sha12=h[:12], n=len(v), 跨ぎ=span,
                                 年代=' / '.join(sorted(ys)),
                                 **{f'代表({k})': pick(v, k)['page_path']
                                    for k in ('辞書順', '年代が古い方', '年代が新しい方')}))

    print('\n== 組の跨ぎ方(理由別)==')
    for k, v in spans.most_common():
        print(f'  {k:22s} {v:>5}')

    print('\n== **主値: 年代が1つに決まらない組**(代表の選び方で、頁が入る年代が変わる)==')
    print(f'  **{len(disagree)} 組 / {len(groups)}**  '
          f'({len(disagree) / len(groups) * 100:.1f}%)')
    print(f'  **帰無値は 0組。{"動いた" if disagree else "動かなかった"}**')
    print(f'  年代が1つに決まる組 {len(groups) - len(disagree)}'
          f'(= 代表の選び方は年代の池を動かさない)')

    print('\n== **年代の池が、代表の選び方でどれだけ動くか**(世代もプールである)==')
    pools = {}
    for how in ('辞書順', '年代が古い方', '年代が新しい方'):
        kept = {pick(v, how)['page_path'] for v in groups.values()}
        c = collections.Counter(
            year_of(r['page_path']) for r in pop
            if r['f_sha256'] not in groups or r['page_path'] in kept)
        pools[how] = c
    ys = sorted(set().union(*[set(c) for c in pools.values()]))
    print(f'  {"年代":8s} ' + ' '.join(f'{h:>12s}' for h in pools) + '   振れ幅')
    for y in ys:
        vals = [pools[h][y] for h in pools]
        mark = ' **' + str(max(vals) - min(vals)) + '**' if max(vals) != min(vals) else '   0'
        print(f'  {y:8s} ' + ' '.join(f'{v:>12d}' for v in vals) + mark)
    print(f'  {"合計":8s} ' + ' '.join(f'{sum(pools[h].values()):>12d}' for h in pools))

    # ---- 畳む ----
    keep = {pick(v, '辞書順')['page_path'] for v in groups.values()}
    dropped = [r for v in groups.values() for r in v
               if r['page_path'] not in keep]
    folded = len(pop) - len(dropped)
    print('\n== 畳んだ後の母集団 ==')
    print(f'  {len(pop)} - {len(dropped)} = **{folded}**  (予測 7,418  '
          f'{"PASS" if folded == 7418 else "**FAIL**"})')
    print('  落とした行(理由別。規律9)')
    for k, v in collections.Counter(r['format'] for r in dropped).most_common():
        print(f'    形式 {k:6s} -{v}')
    for k, v in collections.Counter(len(by[r['f_sha256']]) for r in dropped).most_common():
        print(f'    組の大きさ {k:>3} から -{v}')

    # ---- 既知セル: 署名の塊と追補104 の6頁 ----
    print('\n== 既知セル ==')
    ok = (folded == 7418)
    mine = list(csv.DictReader(open(MINE, encoding='utf-8-sig')))
    cols = list(mine[0])
    cid = next(c for c in cols if 'cluster' in c.lower())
    pp = next(c for c in cols if 'path' in c.lower())
    mc = collections.defaultdict(list)
    for r in mine:
        mc[r[cid]].append(r[pp])
    same = sum(1 for v in mc.values()
               if len(v) > 1 and len({by_sha.get(p, '') for p in v}) == 1
               and all(p in by_sha for p in v))
    print(f'  署名の塊 {len(mc)} のうち、全員バイト同一の塊 **{same}**  '
          f'(予測 58  {"PASS" if same == 58 else "**FAIL**"})')
    ok &= (same == 58)

    roles = list(csv.DictReader(open(ROLES, encoding='utf-8-sig')))
    r104 = [r['page_path'] for r in roles if '裁定1' in r['use'] and '採点しない' in r['use']]
    hit = [p for p in r104 if p not in keep]
    print(f'  追補104 で落とした {len(r104)}頁 が、ここでも落ちる **{len(hit)}**  '
          f'{"PASS" if len(hit) == len(r104) else "**FAIL**"}')
    ok &= (len(hit) == len(r104))
    print(f'  **既知セル {"すべて PASS" if ok else "**FAIL**。主値を読まない"}**')

    # ---- 第二の主値: 札の表に効く数 ----
    byrole = {r['page_path']: r for r in roles}
    eff = [byrole[r['page_path']] for r in dropped if r['page_path'] in byrole]
    adopt = [r for r in eff if adopt_a(r['use'])]
    adopt_wide = [r for r in eff if adopt_b(r['use'])]
    print('\n== **第二の主値: 872頁の札の表に効く数** ==')
    print(f'  落とす側に在る頁のうち、札の表に在る **{len(eff)}**')
    print(f'  うち **`採点する` と数えていた頁 {len(adopt)}**(数え方 A)'
          f' / **`別行で採点` も入れると {len(adopt_wide)}**(数え方 B)')
    print(f'  **帰無値は 0頁。{"動いた" if adopt else "動かなかった"}**')
    for k, v in collections.Counter(r['use'] for r in eff).most_common():
        print(f'    {k[:52]:54s} {v:>3}')
    # **落とす 38頁の代表が、札の表の中に在るか** —— 外に在るなら、
    # 畳むと**画素の在る側を捨てる**ことになる
    rep_of = {r['page_path']: pick(v, '辞書順')['page_path']
              for v in groups.values() for r in v}
    inside = [r for r in adopt if rep_of[r['page_path']] in byrole]
    print(f'\n  **落とす `採点する` {len(adopt)}頁 のうち、代表も札の表の中に在る '
          f'{len(inside)} / 外に在る {len(adopt) - len(inside)}**')
    print('  **外に在る分は、畳むと画素の在る側を捨てることになる** ——'
          ' 代表の選び方を「札の表に在る方」にする規則は**まだ登録していない**')
    print('\n  落とす 38頁の**代表の札**(代表が `採点する` でなければ、札の中で食い違っている)')
    for k, v in collections.Counter(
            byrole[rep_of[r['page_path']]]['use'] for r in adopt).most_common():
        print(f'    {k[:52]:54s} {v:>3}')
    surv = [r for r in roles
            if by_sha.get(r['page_path'], '') not in groups or r['page_path'] in keep]
    n_ad = sum(1 for r in surv if adopt_a(r['use']))
    n_bd = sum(1 for r in surv if adopt_b(r['use']))
    now_a = sum(1 for r in roles if adopt_a(r['use']))
    now_b = sum(1 for r in roles if adopt_b(r['use']))
    print(f'\n  **畳んだ後の札の表 {len(surv)} 行**'
          f'  / 採点側 A **{n_ad}**(いま {now_a}) / 採点側 B **{n_bd}**(いま {now_b})')
    print('  **札の表は書き換えない**(母集団を動かす便と、札を書き換える便を混ぜない)。'
          '上の数は印字だけ')

    # **同じバイト列の組を、札の表がどう扱っているか。**
    # **「採点する 1つ + 採点しない 残り」が正しい形**(重複が畳まれている)。
    # 片側に揃っている組だけが欠陥である:
    #   全部 `採点する`   -> **同じ絵を2回数えている**
    #   全部 `採点しない` -> **その絵が採点側から丸ごと消えている**
    print('\n== 同じバイト列の組を、札の表がどう扱っているか'
          '(**数え方 B。`別行で採点` は採点側**)==')
    cats, bad = collections.Counter(), []
    for h, v in groups.items():
        ins = [byrole[r['page_path']] for r in v if r['page_path'] in byrole]
        if len(ins) < 2:
            cats[f'札の表に在る行が {len(ins)} 行(組として映らない)'] += 1
            continue
        n_ad = sum(1 for x in ins if adopt_b(x['use']))   # **数え方 B で判定する**
        if n_ad == 1:
            cats['**採点する 1つ + 残りは採点しない(正しい形)**'] += 1
        elif n_ad == 0:
            # **`superseded_by` が採点側の行を指していれば、その絵は別の行で採点される。**
            # **欠陥ではない** —— 指す先が表の外か、指していない組だけが欠陥
            tgt = [byrole.get(x['superseded_by']) for x in ins if x['superseded_by']]
            if tgt and all(t is not None and adopt_b(t['use']) for t in tgt):
                cats['採点しないが、`superseded_by` が採点側を指している(正しい形)'] += 1
            else:
                cats['**全部 採点しない。指す先も採点側でない(絵が消えている)**'] += 1
                bad.append((h[:12], '全部 採点しない', ins))
        else:
            cats[f'**全部/複数が 採点する({n_ad}つ。同じ絵を2回数えている)**'] += 1
            bad.append((h[:12], f'採点する が {n_ad}つ', ins))
    for k, v in cats.most_common():
        print(f'  {k:52s} {v:>5}')
    print(f'  **欠陥の組 {len(bad)}** —— **代表の選び方では直らない。札の方の問題**')
    if bad:
        with open(OUT / 'group_label_defects.csv', 'w', newline='',
                  encoding='utf-8-sig') as f:
            w = csv.writer(f); w.writerow(['sha12', '欠陥', 'page_path', 'use'])
            for h, why, ins in bad:
                for x in ins:
                    w.writerow([h, why, x['page_path'], x['use']])

    print(f'\nwrote {OUT}/')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
