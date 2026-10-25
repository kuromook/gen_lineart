"""M25: **作者の回答で版の札を直す**。名前順・時刻・`superset` の錨はすべて落ちたので、
残る錨は**作者の言葉だけ**(通達 `notice_version_identity_ask_the_author_20261026.md`)。

入力は `m19` の出力(`results/m19_versions/version_roles.csv`)。**m19 の規則は書き換えない**
(走った記録なので残す)。この器具は**上書きの層**として作者の回答を当てる。

当てる規則(**作者の言葉。これだけが錨**):

  A1 「**b付きが新しい　どちらも**」
     → 年フォルダーが `YYYY` と `YYYYb` の対になっている塊は、**b 側が `final`**
  A2 「**`F:\manuscripts\2016\gal\gal` は古い　`2016\gal\` 直下にある clip が正**」
     → `2016\gal` の塊は**直下側が `final`**

基盤の裁定で当てる規則(作者の回答ではないので区別して書く):

  B1 **塊が作品を跨ぐなら版の連鎖として解決しない**(`ruling_m_name_order_no_anchor` §4)
     → 構成員全員を `duplicate_content` にし、**両方を母集団に残す**。
       `superseded` と書くと「内容の同一性」が「頁の置き換え」に化ける(T94)
  B2 **`superset`(層の包含)は作者に 1/3。根拠として引用しない**
     → A1/A2/B1 で決まらなかった `superset` の塊は **`undecided`** に戻す

**期待(走らせる前に書く。合わなければ表を出さない)**:

| | 内容 | 出所 |
|---|---|---|
| **KC-V1** | A1 が当たる塊が **25**(2019/2019b 18 + 2022/2022b 5 + 塊83・85) | 基盤の裁定(10-25 23:39 + 10-26 23:49 の訂正) |
| **KC-V2** | A2 が当たる塊が **3**(31・32 は反転、33 は現状で正しい)、反転は **2** | 同上 |
| **KC-V3** | **反転の合計 27塊** | `ruling_m_author_gal_shallow_20261026.md` §3 |
| **KC-V4** | `undecided` が **11塊 → 9塊**(塊87 と 106 が解ける) | `notice_version_identity_ask_the_author` §3 |
| **KC-V5** | B1 が当たる塊が **3**(24・29・58) | `ruling_m_name_order_no_anchor` §4 |

**どれか1つでも外れたら、表を書かずに止める。** 作者の言葉を当てる作業で数が合わないなら、
当てている対象が作者の述べた対象ではない。
"""
import collections
import csv
import re
import sys
from pathlib import Path

SRC = Path('results/m19_versions/version_roles.csv')
OUT = Path('results/m25_author_versions'); OUT.mkdir(parents=True, exist_ok=True)
PAGELIST = Path('results/m21_request4/request4_page_list.csv')
EXPECT = {'a1': 25, 'a2': 3, 'flips': 27, 'undecided': 9, 'crosswork': 3}
YEARB = re.compile(r'^(\d{4})([a-z])$')


def seg(p):
    q = p.replace('\\', '/').split('/')
    return q[2] if len(q) > 2 else '', q[3] if len(q) > 3 else '', q[2:]


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    byc = collections.defaultdict(list)
    for r in rows:
        if r['dup_cluster_id']:
            byc[r['dup_cluster_id']].append(r)
    print(f'塊 {len(byc)} / 行 {len(rows)}')

    a1, a2, crosswork, b2 = [], [], [], []
    for ci, g in byc.items():
        years = {seg(r['page_path'])[0] for r in g}
        works = {seg(r['page_path'])[1] for r in g}
        base = {YEARB.match(y).group(1) if YEARB.match(y) else y for y in years}
        if len(works) > 1:
            crosswork.append(ci)
            continue
        if len(years) > 1 and len(base) == 1 and any(YEARB.match(y) for y in years):
            a1.append(ci)
            continue
        if all(seg(r['page_path'])[0] == '2016'
               and seg(r['page_path'])[1] == 'gal' for r in g):
            a2.append(ci)
            continue
        if any(r['decided_by'] == 'superset' for r in g):
            b2.append(ci)

    print(f'\n== 規則が当たる塊 ==')
    print(f'  A1 b 付きの年 : {len(a1)} 塊(期待 {EXPECT["a1"]})  {sorted(a1, key=int)}')
    print(f'  A2 2016\\gal   : {len(a2)} 塊(期待 {EXPECT["a2"]})  {sorted(a2, key=int)}')
    print(f'  B1 作品を跨ぐ : {len(crosswork)} 塊(期待 {EXPECT["crosswork"]})  '
          f'{sorted(crosswork, key=int)}')
    print(f'  B2 superset 残: {len(b2)} 塊  {sorted(b2, key=int)}')

    new = {r['page_path']: dict(r) for r in rows}
    flips, resolved = [], []

    def setrole(r, role, why, sup=''):
        n = new[r['page_path']]
        if n['version_role'] != role:
            flips.append((r['dup_cluster_id'], r['page_path'],
                          n['version_role'], role))
        n['version_role'] = role
        n['decided_by'] = why
        n['superseded_by'] = sup

    for ci in a1:
        g = byc[ci]
        win = [r for r in g if YEARB.match(seg(r['page_path'])[0])]
        if len(win) != 1:
            print(f'  !! 塊{ci} の b 側が {len(win)} 件。止める')
            return 1
        w = win[0]
        for r in g:
            setrole(r, 'final' if r is w else 'superseded',
                    '作者の回答(b 付きが新しい　どちらも)',
                    '' if r is w else w['page_path'])
        resolved.append(ci)
    for ci in a2:
        g = byc[ci]
        win = [r for r in g if len(seg(r['page_path'])[2]) == 3]
        if len(win) != 1:
            print(f'  !! 塊{ci} の直下側が {len(win)} 件。止める')
            return 1
        w = win[0]
        for r in g:
            setrole(r, 'final' if r is w else 'superseded',
                    '作者の回答(2016\\gal 直下が正)',
                    '' if r is w else w['page_path'])
        resolved.append(ci)
    for ci in crosswork:
        for r in byc[ci]:
            setrole(r, 'duplicate_content',
                    '塊が作品を跨ぐ。版の連鎖として解決しない(T94/T95)')
    for ci in b2:
        for r in byc[ci]:
            setrole(r, 'undecided',
                    'superset は作者に 1/3。根拠として引用しない')

    und = {ci for ci, g in byc.items()
           if any(new[r['page_path']]['version_role'] == 'undecided' for r in g)}
    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-V1 A1 の塊', len(a1), EXPECT['a1']),
                         ('KC-V2 A2 の塊', len(a2), EXPECT['a2']),
                         ('KC-V3 反転した頁を持つ塊', len({f[0] for f in flips
                                                   if f[0] in set(a1) | set(a2)}),
                          EXPECT['flips']),
                         ('KC-V4 undecided の塊', len(und), EXPECT['undecided']),
                         ('KC-V5 作品を跨ぐ塊', len(crosswork), EXPECT['crosswork'])):
        good = got == want
        ok &= good
        print(f'  {k:28s} {got:>4} / 期待 {want:>4}  '
              f'{"PASS" if good else "**FAIL**"}')
    if not ok:
        print('\n**既知セルが合わない。表を書かない。**')
        print('  反転した塊:', sorted({f[0] for f in flips}, key=int))
        return 1

    role = collections.Counter(v['version_role'] for v in new.values())
    print(f'\n== 新しい役割の分布 ==')
    for k, c in role.most_common():
        print(f'  {k:20s} {c:>4}')
    keep = [p for p, v in new.values().__class__ and new.items()
            if v['version_role'] not in ('superseded',)]
    scoring = [p for p, v in new.items()
               if v['version_role'] not in ('superseded', 'duplicate_content')]
    print(f'\n  母集団(superseded だけ落とす)  : **{len(keep)}** ページ')
    print(f'  採点の母集団(内容の重複も1つに): **{len(scoring)}** ページ'
          f'  ← T95(プールの単位は塊)')

    print(f'\n== 札が変わった頁 {len(flips)} ==')
    for ci, p, o, n in sorted(flips, key=lambda x: int(x[0])):
        print(f'  塊{ci:>4}  {o:>18} -> {n:<18} {p[14:]}')

    inrun = set()
    if PAGELIST.exists():
        pl = list(csv.DictReader(open(PAGELIST, encoding='utf-8')))
        inrun = {r['page_path'] for r in pl}
        changed_in = [f for f in flips if f[1] in inrun]
        added = [p for p, v in new.items()
                 if p not in inrun and v['version_role'] not in
                 ('superseded', 'duplicate_content')]
        dropped = [p for p in inrun
                   if new.get(p, {}).get('version_role') == 'superseded']
        print(f'\n== 本走(779頁)への影響 ==')
        print(f'  一覧 {len(inrun)} 頁のうち札が変わった頁 **{len(changed_in)}**')
        print(f'  一覧に無いが現行になった頁(**追加が要る**) **{len(added)}**')
        print(f'  一覧に有るが現行でなくなった頁(**描き直し不要・札だけ**) **{len(dropped)}**')
        for p in sorted(added):
            print(f'    追加 {p[14:]}')

    cols = list(rows[0].keys())
    with open(OUT / 'version_roles_author.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(new[r['page_path']])
    with open(OUT / 'version_role_changes.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['dup_cluster_id', 'page_path', '旧 version_role',
                    '新 version_role', '新 decided_by', '本走の一覧に在るか'])
        for ci, p, o, n in sorted(flips, key=lambda x: int(x[0])):
            w.writerow([ci, p, o, n, new[p]['decided_by'],
                        'yes' if p in inrun else ''])
    print(f'\nwrote {OUT/"version_roles_author.csv"} / '
          f'{OUT/"version_role_changes.csv"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
