# -*- coding: utf-8 -*-
"""m80: **作者が決めた代表の規則(年代が新しい方)を母集団と札の表に当てる**(追補110 §B)。

**代表 = 年代が新しい方**(作者 2026-10-28)。**同年代の組では年代の規則が何も言わない**ので、
**同順は道の辞書順の先頭 —— 機械的で、同年代の中では階層の浅い方・名前の早い方を選ぶ**
(追補108 C1: 「辞書順」と書くときは、それが何の軸を選んでいるかも書く)。

**主値**: 畳んだ後の採点側 A(いま 661)/ B(いま 745)。**帰無値 = 動かない。**
**第二の主値**: 同じ絵を2回数えている組(いま 7)。**帰無値 = 7のまま。**
**第三の主値**: **新しい側の複製が 872頁の表の外に在る組**。**帰無値 = 0組**
  —— 0 でなければ、年代の規則と「測れること」が衝突する。**こちらでは解かない。**
"""
import collections
import csv
import re
import sys
from pathlib import Path

ALL = Path('../lineart/inbox/request9_file_sha256_all_20261026.csv')
MINE = Path('../lineart/inbox/sha256_dup_clusters_20261026.csv')
ROLES = Path('results/m78_workbook_applied/version_roles_current.csv')
OUT = Path('results/m80_rep_rule_newest'); OUT.mkdir(parents=True, exist_ok=True)
YEAR = re.compile(r'\\(\d{4}[a-z]?)\\')


def year_of(p):
    m = YEAR.search(p)
    return m.group(1) if m else ''


def adopt_a(u):
    return u.startswith('採点する')


def adopt_b(u):
    return u.startswith('採点する') or u.startswith('別行で採点')


def main():
    rows = [r for r in csv.DictReader(open(ALL, encoding='utf-8-sig'))
            if r['f_sha256'] and not r['f_read_error']]
    roles = list(csv.DictReader(open(ROLES, encoding='utf-8-sig')))
    byrole = {r['page_path']: r for r in roles}
    by = collections.defaultdict(list)
    for r in rows:
        by[r['f_sha256']].append(r)
    groups = {h: v for h, v in by.items() if len(v) > 1}
    print('== 材料 ==')
    print(f'  {ALL}  母集団 {len(rows)}  /  バイト同一の組 {len(groups)}')
    print(f'  {ROLES}  {len(roles)} 頁')

    # **代表 = 年代が新しい方。同順は道の辞書順の先頭**
    def pick(v):
        # 年代を降順、同年代では道を昇順
        return sorted(v, key=lambda r: (_negyear(year_of(r['page_path'])),
                                        r['page_path']))[0]

    keep = {pick(v)['page_path'] for v in groups.values()}
    dropped = [r for v in groups.values() for r in v if r['page_path'] not in keep]
    folded = len(rows) - len(dropped)

    print('\n== 既知セル ==')
    ok = (folded == 7418)
    print(f'  畳んだ後の母集団 {folded}  (予測 7,418  {"PASS" if ok else "**FAIL**"})')
    mine = list(csv.DictReader(open(MINE, encoding='utf-8-sig')))
    cols = list(mine[0])
    cid = next(c for c in cols if 'cluster' in c.lower())
    pp = next(c for c in cols if 'path' in c.lower())
    mc = collections.defaultdict(list)
    for r in mine:
        mc[r[cid]].append(r[pp])
    sha = {r['page_path']: r['f_sha256'] for r in rows}
    same = sum(1 for v in mc.values() if len(v) > 1
               and all(p in sha for p in v) and len({sha[p] for p in v}) == 1)
    print(f'  署名の塊で全員バイト同一 {same}  (予測 58  {"PASS" if same == 58 else "**FAIL**"})')
    r104 = [r['page_path'] for r in roles if '裁定1' in r['use'] and '採点しない' in r['use']]
    hit = [p for p in r104 if p not in keep]
    print(f'  追補104 で落とした {len(r104)}頁 がここでも落ちる {len(hit)}  '
          f'{"PASS" if len(hit) == len(r104) else "**FAIL**"}')
    ok &= (same == 58) and (len(hit) == len(r104))
    print(f'  **既知セル {"すべて PASS" if ok else "FAIL。先を読まない"}**')
    if not ok:
        return 1

    print('\n== 第三の主値: **新しい側が札の表の外に在る組**(画素を捨てる形)==')
    risk = []
    for h, v in groups.items():
        ins = [r for r in v if r['page_path'] in byrole]
        if not ins:
            continue
        rep = pick(v)['page_path']
        if rep not in byrole:
            risk.append((h[:12], rep, [r['page_path'] for r in ins]))
    print(f'  札の表に1行以上在る組 {sum(1 for v in groups.values() if any(r["page_path"] in byrole for r in v))}')
    print(f'  **うち代表(新しい側)が表の外に在る組 {len(risk)}**  '
          f'(帰無値 0。{"**動いた**" if risk else "動かなかった"})')
    for h, rep, ins in risk[:10]:
        print(f'    {h}  代表(表の外) {rep[14:]}')
        for p in ins:
            print(f'        表の中: {p[14:]}  {byrole[p]["use"][:34]}')

    print('\n== 主値: 畳んだ後の札の表 ==')
    surv = [r for r in roles
            if sha.get(r['page_path'], '') not in groups or r['page_path'] in keep]
    a0, b0 = (sum(1 for r in roles if adopt_a(r['use'])),
              sum(1 for r in roles if adopt_b(r['use'])))
    a1, b1 = (sum(1 for r in surv if adopt_a(r['use'])),
              sum(1 for r in surv if adopt_b(r['use'])))
    print(f'  行        {len(roles)} -> **{len(surv)}**')
    print(f'  採点側 A  {a0} -> **{a1}**  差 {a1 - a0}')
    print(f'  採点側 B  {b0} -> **{b1}**  差 {b1 - b0}')
    print(f'  **帰無値は「動かない」。{"動いた" if a1 != a0 else "動かなかった"}**')

    print('\n== 第二の主値: 同じ絵を2回数えている組 ==')
    cats, bad = collections.Counter(), []
    for h, v in groups.items():
        ins = [byrole[r['page_path']] for r in v
               if r['page_path'] in byrole and r['page_path'] in keep]
        if len(ins) < 2:
            cats[f'畳んだ後に表に残る行が {len(ins)} 行'] += 1
            continue
        n = sum(1 for x in ins if adopt_b(x['use']))
        if n >= 2:
            cats['**採点側が2つ以上(まだ2回数えている)**'] += 1
            bad.append((h[:12], ins))
        else:
            cats['採点側が1つ以下(解けた)'] += 1
    for k, v in cats.most_common():
        print(f'  {k:46s} {v:>5}')
    print(f'  **まだ2回数えている組 {len(bad)}**  (前 7。帰無値「7のまま」。'
          f'{"動いた" if len(bad) != 7 else "動かなかった"})')
    for h, ins in bad:
        print(f'    {h}')
        for x in ins:
            print(f'        {x["page_path"][14:]:46s} {x["use"][:34]}')

    print('\n== 年ごとの池(畳んだ後。**b/c は別の池のまま**。作者の答)==')
    pool = collections.Counter(year_of(r['page_path']) for r in rows
                               if sha[r['page_path']] not in groups
                               or r['page_path'] in keep)
    old = collections.Counter(year_of(r['page_path']) for r in rows)
    for y in sorted(pool):
        print(f'  {y:7s} 畳む前 {old[y]:>5} -> 畳んだ後 **{pool[y]:>5}**  '
              f'{pool[y] - old[y]:+d}')
    print(f'  {"合計":7s} 畳む前 {sum(old.values()):>5} -> 畳んだ後 **{sum(pool.values()):>5}**')

    with open(OUT / 'folded_population_newest.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['page_path', 'format', 'f_sha256', 'f_bytes', '年代'])
        for r in rows:
            if sha[r['page_path']] not in groups or r['page_path'] in keep:
                w.writerow([r['page_path'], r['format'], r['f_sha256'], r['f_bytes'],
                            year_of(r['page_path'])])
    with open(OUT / 'dropped_newest.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['page_path', '代表', '年代', '代表の年代', '組の大きさ'])
        for v in groups.values():
            rep = pick(v)['page_path']
            for r in v:
                if r['page_path'] != rep:
                    w.writerow([r['page_path'], rep, year_of(r['page_path']),
                                year_of(rep), len(v)])
    with open(OUT / 'version_roles_folded.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(roles[0])); w.writeheader(); w.writerows(surv)
    if risk:
        with open(OUT / 'newest_outside_table.csv', 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f); w.writerow(['sha12', '代表(表の外)', '表の中の頁', 'use'])
            for h, rep, ins in risk:
                for p in ins:
                    w.writerow([h, rep, p, byrole[p]['use']])
    print(f'\nwrote {OUT}/')
    return 0


def _negyear(y):
    """年代札を降順に並べるための鍵(\`2019b\` は \`2019\` より後)。"""
    return tuple(-ord(c) for c in f'{y:<6}')


if __name__ == '__main__':
    sys.exit(main())
