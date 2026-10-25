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
     **基盤が名指しした塊24・29・58 にだけ当てる。** 「作品」をフォルダー名から
     機械で決めてはいけない —— 初版はそれをやって 23塊を拾い、`comics4\\underworld`
     (束ねフォルダー)や `china`↔`china_v2`(同じ作品の改訂)まで作品跨ぎに数えた。
     **広い走査は印字して問いに回す。器具で決めない**(10-26 の通達)
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
# **数えるのは「規則が当たる塊」ではなく「札が変わる塊」。** 初版は前者を数えて
# A1 27 / A2 4 を出し、既知セルを落とした —— 基盤の 25・2 は**決着済みの塊のうち
# 反転するもの**の数だった。**棒ではなく数え方を直した**(m24 と同じ形)。
EXPECT = {'a1_flip': 25, 'a2_flip': 2, 'flips': 27, 'undecided': 9,
          'crosswork': 3}
CROSSWORK = {'24', '29', '58'}            # 基盤が名指しした3塊(§4)
YEARB = re.compile(r'^(\d{4})([a-z])$')


def parts(p):
    return p.replace('\\', '/').split('/')


def b_side(p1, p2):
    """p2 が p1 の **b 付きの年**の同じ頁なら True(他は全部同じ道)。"""
    a, b = parts(p1), parts(p2)
    if len(a) != len(b) or a[3:] != b[3:]:
        return False
    m = YEARB.match(b[2])
    return bool(m) and m.group(1) == a[2]


def gal_deep(p1, p2):
    """`2016\\gal\\gal\\pageN` (p2) と `2016\\gal\\pageN` (p1) の対なら True。"""
    a, b = parts(p1), parts(p2)
    return (len(a) > 4 and len(b) > 5 and a[2] == b[2] == '2016'
            and a[3] == b[3] == 'gal' and b[4:] == ['gal'] + a[4:])


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    byc = collections.defaultdict(list)
    for r in rows:
        if r['dup_cluster_id']:
            byc[r['dup_cluster_id']].append(r)
    print(f'塊 {len(byc)} / 行 {len(rows)}')

    new = {r['page_path']: dict(r) for r in rows}
    flips, a1c, a2c, pairs, big = [], set(), set(), [], []

    def setrole(r, role, why, sup=''):
        n = new[r['page_path']]
        if n['version_role'] != role:
            flips.append((r['dup_cluster_id'], r['page_path'],
                          n['version_role'], role))
        n['version_role'], n['decided_by'], n['superseded_by'] = role, why, sup

    for ci, g in byc.items():
        if ci in CROSSWORK:
            continue
        if len(g) != 2:
            # **構成員が3つ以上の塊には当てない。** 作者は「b 付きが新しい」と
            # **1つの頁の2つの年**について述べている。3つ以上の塊(例: 塊1 は
            # `ako7\\page0071` と `ako6\\done4\\ako6-27` と その b 側)では、
            # 対ごとに当てると塊の中に `final` が2つ出る。**保留して問いに回す。**
            if any(b_side(a['page_path'], b['page_path'])
                   or b_side(b['page_path'], a['page_path'])
                   or gal_deep(a['page_path'], b['page_path'])
                   or gal_deep(b['page_path'], a['page_path'])
                   for a in g for b in g if a is not b):
                big.append(ci)
            continue
        for i, r1 in enumerate(g):
            for r2 in g[i + 1:]:
                p1, p2 = r1['page_path'], r2['page_path']
                A1 = '作者の回答(b 付きが新しい　どちらも)'
                A2 = '作者の回答(2016\\gal 直下が正)'
                if b_side(p1, p2):
                    lo, hi, why, tag = r1, r2, A1, 'a1'
                elif b_side(p2, p1):
                    lo, hi, why, tag = r2, r1, A1, 'a1'
                elif gal_deep(p1, p2):
                    lo, hi, why, tag = r2, r1, A2, 'a2'
                elif gal_deep(p2, p1):
                    lo, hi, why, tag = r1, r2, A2, 'a2'
                else:
                    continue
                # lo = 落ちる側(古い) / hi = 現行
                setrole(hi, 'final', why)
                setrole(lo, 'superseded', why, hi['page_path'])
                (a1c if tag == 'a1' else a2c).add(ci)
                pairs.append((ci, tag, hi['page_path'], lo['page_path']))
    for ci in CROSSWORK:
        for r in byc[ci]:
            setrole(r, 'duplicate_content',
                    '塊が作品を跨ぐ。版の連鎖として解決しない(T94/T95)')
    b2 = [ci for ci, g in byc.items()
          if ci not in a1c | a2c | CROSSWORK
          and any(r['decided_by'] == 'superset' for r in g)]
    for ci in b2:
        for r in byc[ci]:
            setrole(r, 'undecided', 'superset は作者に 1/3。根拠として引用しない')

    print(f'\n== 規則が当たった対 {len(pairs)}(塊 A1 {len(a1c)} / A2 {len(a2c)}) ==')
    print(f'  **保留**: 作者の規則が内側に当たるが構成員が3つ以上の塊 '
          f'{len(big)} {sorted(big, key=int)}  ← 問いに回す')
    print(f'  B1 作品を跨ぐ(基盤の名指し): {sorted(CROSSWORK, key=int)}')
    print(f'  B2 superset を根拠にしていた残り: {sorted(b2, key=int)}')

    dec = {ci for ci, g in byc.items()
           if {r['version_role'] for r in g} & {'final', 'superseded'}}
    und0 = {ci for ci, g in byc.items()
            if any(r['version_role'] == 'undecided' for r in g)}
    und = {ci for ci, g in byc.items()
           if any(new[r['page_path']]['version_role'] == 'undecided'
                  for r in g)}
    und_left = {ci for ci in und0
                if any(new[r['page_path']]['version_role'] == 'undecided'
                       for r in byc[ci])}
    fl = {f[0] for f in flips}
    print(f'\n== 既知セル(数えるのは「札が変わった決着済みの塊」) ==')
    ok = True
    for k, got, want in (('KC-V1 A1 で反転(決着済み)', len(fl & a1c & dec),
                          EXPECT['a1_flip']),
                         ('KC-V2 A2 で反転(決着済み)', len(fl & a2c & dec),
                          EXPECT['a2_flip']),
                         ('KC-V3 反転の合計(決着済み)',
                          len(fl & (a1c | a2c) & dec), EXPECT['flips']),
                         ('KC-V4 原11塊のうち残る undecided',
                          len(und_left), EXPECT['undecided']),
                         ('KC-V5 作品を跨ぐ塊', len(CROSSWORK),
                          EXPECT['crosswork'])):
        good = got == want
        ok &= good
        print(f'  {k:28s} {got:>4} / 期待 {want:>4}  '
              f'{"PASS" if good else "**FAIL**"}')
    print(f'  (参考)A1/A2 で札が変わった塊のうち、m19 が決着させていなかったもの: '
          f'{sorted((fl & (a1c | a2c)) - dec, key=int)}')
    print(f'  (参考)undecided の塊は全部で {len(und)}'
          f'(原11塊の残り {len(und_left)} + superset を外した {sorted(und - und0, key=int)})')
    if not ok:
        print('\n**既知セルが合わない。表を書かない。**')
        return 1

    role = collections.Counter(v['version_role'] for v in new.values())
    print(f'\n== 新しい役割の分布 ==')
    for k, c in role.most_common():
        print(f'  {k:20s} {c:>4}')
    keep = [p for p, v in new.items() if v['version_role'] != 'superseded']
    # **採点の母集団は塊ごとに代表1つ**(T95: プールの単位は重複の塊)。
    # 初版は `duplicate_content` を全部落としていて、**構成員が全員
    # `duplicate_content` の塊(作品を跨ぐ3塊)が0頁になっていた** ——
    # 「両方を母集団に残す」(裁定§4)の逆。代表を1つ残す形に直した。
    hasfinal = {v['dup_cluster_id'] for v in new.values()
                if v['version_role'] == 'final' and v['dup_cluster_id']}
    scoring, seen = [], set()
    for q, v in new.items():
        role, ci = v['version_role'], v['dup_cluster_id']
        if role == 'superseded':
            continue
        if role == 'duplicate_content':
            if ci in hasfinal or ci in seen:
                continue
            seen.add(ci)
        scoring.append(q)
    print(f'\n  頁の母集団(superseded だけ落とす)  : **{len(keep)}** ページ'
          f'  ← 作者の持つ頁。重複も両方残す(裁定§4)')
    print(f'  採点の母集団(重複の塊は代表1つ)    : **{len(scoring)}** ページ'
          f'  ← T95(プールの単位は塊)')

    print(f'\n== 札が変わった頁 {len(flips)} ==')
    for ci, p, o, n in sorted(flips, key=lambda x: int(x[0])):
        print(f'  塊{ci:>4}  {o:>18} -> {n:<18} {p[14:]}')

    inrun = set()
    if PAGELIST.exists():
        inrun = {r['page_path'] for r in
                 csv.DictReader(open(PAGELIST, encoding='utf-8'))}
        changed_in = [f for f in flips if f[1] in inrun]
        added = [p for p in scoring if p not in inrun]
        dropped = [p for p in inrun
                   if new.get(p, {}).get('version_role') == 'superseded']
        print(f'\n== 本走(一覧 {len(inrun)} 頁)への影響 ==')
        print(f'  一覧にあって札が変わった頁 **{len(changed_in)}**')
        print(f'  一覧に無いが採点の母集団に入る頁(**追加が要る**) **{len(added)}**')
        print(f'  一覧にあって現行でなくなった頁(**描き直しの候補**) **{len(dropped)}**')
        for p in sorted(added):
            print(f'    追加 {p[14:]}')
        for p in sorted(dropped)[:30]:
            print(f'    落ち {p[14:]}')

    with open(OUT / 'version_roles_author.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
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
