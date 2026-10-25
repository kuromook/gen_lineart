"""M26: 作者の回答で母集団が動いたので、**依頼4 の一覧の訂正(追加分)**を出す。

**本走(779頁)は止めない。** 出すのは3つ:

  1. **追加が要る頁**(採点の母集団に入ったが、走っている一覧に無い)
  2. **現行でなくなった頁**(一覧に在るが `superseded` になった) —— 描き直しではなく**札**
  3. 追加分の**層の一覧**(アルファを要求する層)

選別の規則は `m21.select`(1か所から借りる)、母集団の規則は
`m25.scoring_population`(同じく1か所から)。**どちらも書き直さない。**

**既知セル(走らせる前に書く)**:

| | 内容 | 出所 |
|---|---|---|
| **KC-A1** | 旧の母集団786で選ぶと **8,076層 / 512ページ**、先頭8頁は **303層** | 依頼4 の発注(10-25 19:55) |
| **KC-A2** | 追加が要る頁は **30**、現行でなくなる頁は **29** | `m25`(追補18 §F の訂正) |

**KC-A2 は最初 28 で登録して落ちた(実測30)。28 は `m25` の
`scoring_population` を直す前の走行の数**で、直す前は `duplicate_content` を
全部落としていた。**増えた2頁は作品跨ぎの塊の代表**
(`2018\\skima11\\skima12.clip` 塊24 / `2018\\ako6\\done4\\ako6-18.clip` 塊58)で、
**裁定§4「両方を母集団に残す」で戻ってきた頁**である。
**数で合わせたのではなく、増えた2頁を名前で確かめて 30 にした。**
28 は追補18 §F と `note_m_author_answers_applied_20261026.md` §3 で**撤回**(規律3)。

**落ちたら一覧を書かない。**
"""
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m21_request4_layer_list as m21                   # noqa: E402
import m25_author_version_answers as m25                # noqa: E402

NEWROLES = Path('results/m25_author_versions/version_roles_author.csv')
OLDPAGES = Path('results/m21_request4/request4_page_list.csv')
OUT = Path('results/m26_addendum'); OUT.mkdir(parents=True, exist_ok=True)
EXPECT = {'layers_old': 8076, 'pages_old': 512, 'known': 303,
          'added': 30, 'dropped': 29}


def main():
    rows = list(csv.DictReader(open(m21.CSV872, encoding='utf-8')))
    new = {r['page_path']: r for r in csv.DictReader(
        open(NEWROLES, encoding='utf-8'))}
    pages = list(dict.fromkeys(r['page_path'] for r in rows))
    reg = pages[:m21.REG8]
    old = [r['page_path'] for r in csv.DictReader(
        open(OLDPAGES, encoding='utf-8'))]
    scoring = m25.scoring_population(new)
    order_old = reg + [p for p in old if p not in set(reg)]
    uniq_old, stat_old = m21.select(rows, order_old)
    known = sum(1 for r in uniq_old if r[0] in set(reg))

    added = [p for p in scoring if p not in set(old)]
    dropped = [p for p in old if new[p]['version_role'] == 'superseded']

    print('== 既知セル ==')
    ok = True
    for k, got, want in (('KC-A1 旧の層数', len(uniq_old), EXPECT['layers_old']),
                         ('KC-A1 旧のページ数', len({r[0] for r in uniq_old}),
                          EXPECT['pages_old']),
                         ('KC-A1 先頭8頁の層数', known, EXPECT['known']),
                         ('KC-A2 追加が要る頁', len(added), EXPECT['added']),
                         ('KC-A2 現行でなくなる頁', len(dropped),
                          EXPECT['dropped'])):
        good = got == want
        ok &= good
        print(f'  {k:24s} {got:>6} / 期待 {want:>6}  '
              f'{"PASS" if good else "**FAIL**"}')
    if not ok:
        print('\n**既知セルが合わない。一覧を書かない。**')
        return 1

    uniq_add, stat_add = m21.select(rows, added)
    cnt = collections.Counter(r[0] for r in uniq_add)
    print(f'\n== 追加分 ==')
    print(f'  頁 **{len(added)}** / アルファを要求する層 **{len(uniq_add)}**'
          f'  (ベース {stat_add["base"]} + フォルダーのベースの子孫 '
          f'{stat_add["child"]})')
    print(f'  ベースが1枚も無い頁(プレビューだけ): '
          f'{sum(1 for p in added if not cnt[p])}')
    print(f'  ラスタを持たないベース: {stat_add["base_no_raster"]}')
    for p in sorted(added):
        print(f'    {cnt[p]:>4} 層  {p[14:]}')
    print(f'\n== 現行でなくなった頁 {len(dropped)}(札だけ。描き直しではない) ==')
    for p in sorted(dropped):
        print(f'    {p[14:]}  -> {new[p]["superseded_by"][14:]}')

    with open(OUT / 'request4_addendum_layer_list.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['順', 'page_path', 'layer_id', '役割',
                    '親フォルダーの layer_id', 'name'])
        for i, (p, lid, role, par, nm) in enumerate(uniq_add, start=1):
            w.writerow([i, p, lid, role, par, nm])
    with open(OUT / 'request4_addendum_page_list.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['順', 'page_path', 'version_role', 'decided_by',
                    'アルファ要求層数'])
        for i, p in enumerate(sorted(added), start=1):
            w.writerow([i, p, new[p]['version_role'], new[p]['decided_by'],
                        cnt[p]])
    with open(OUT / 'request4_no_longer_current.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['page_path', 'superseded_by', 'decided_by'])
        for p in sorted(dropped):
            w.writerow([p, new[p]['superseded_by'], new[p]['decided_by']])
    print(f'\nwrote {OUT}/request4_addendum_layer_list.csv ほか2本')
    return 0


if __name__ == '__main__':
    sys.exit(main())
