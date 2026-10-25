"""M32: 作者の問い E の回答を当てる。**D と同じく「おそらく」なので札を埋めない**。

作者の回答(逐語。`decided_by` にこのまま):

> E これも全くおなじに見える おそらくバックアップ

(`2018\\skima11\\skima12chara.clip` ↔ `skima12charaDone.clip`、塊18)

**「おそらく」は断定ではない**(基盤)。→ **役割は `undecided`**、
`decided_by` に逐語 + **「作者も判定できない(推定の語)」**。
**接尾辞 `Done` を根拠にしない** —— 作者は `Done` を「仕上げた方」と読んでいない。

**期待(走らせる前に書く)**:

| | 内容 |
|---|---|
| **KC-E1** | 当たる塊は **1**(塊18) |
| **KC-E2** | 頁の母集団 **822 → 823** / 採点 **800 → 801** / 主値 **767 → 768** |
| **KC-E3** | **sha256 待ちの暫定が 6 → 7頁**(17・22・30・103・104・105・18)。
             **一致なら主値は 768 − 7 = 761** |
"""
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m25_author_version_answers as m25                 # noqa: E402

SRC = Path('results/m31_avbu/version_roles_current.csv')
OUT = Path('results/m32_charadone'); OUT.mkdir(parents=True, exist_ok=True)
VERBATIM = ('作者の回答(2026-10-26): E これも全くおなじに見える おそらくバックアップ'
            ' / **作者も判定できない(推定の語)。** 画素の照合(sha256)待ち')
PROVISIONAL = {'17', '22', '30', '103', '104', '105', '18'}
EXPECT = {'pages': 823, 'scoring': 801, 'done': 768}


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    byc = collections.defaultdict(list)
    for r in rows:
        if r['dup_cluster_id']:
            byc[r['dup_cluster_id']].append(r)
    for r in byc['18']:
        q = r['page_path']
        print(f'  塊18 {new[q]["version_role"]:>12} -> undecided  {q[14:]}')
        new[q]['version_role'] = 'undecided'
        new[q]['decided_by'] = VERBATIM
        new[q]['superseded_by'] = ''
    pages = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    sc = m25.scoring_population(new)
    wip = sum(1 for p in sc if new[p]['version_role'] == 'wip')
    prov = [p for p in sc if new[p]['dup_cluster_id'] in PROVISIONAL]
    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-E2 頁の母集団', pages, EXPECT['pages']),
                         ('KC-E2 採点の母集団', len(sc), EXPECT['scoring']),
                         ('KC-E2 主値(仕上がった頁)', len(sc) - wip,
                          EXPECT['done']),
                         ('KC-E3 暫定の頁(塊7つ)', len(prov) - 7, 7)):
        good = got == want
        ok &= good
        print(f'  {k:26s} {got:>5} / 期待 {want:>5}  '
              f'{"PASS" if good else "**FAIL**"}')
    print(f'  (暫定の塊の構成員 {len(prov)}頁 = 7塊 × 2頁)')
    if not ok:
        print('\n**既知セルが合わない。表を書かない。**')
        return 1
    print(f'\n== 役割 ==')
    for k, c in collections.Counter(
            v['version_role'] for v in new.values()).most_common():
        print(f'  {k:20s} {c:>4}')
    print(f'\n**主値 {len(sc) - wip}。sha256 で7塊が一致すれば '
          f'{len(sc) - wip - 7} に下がる**(同じ画素を2回数えないため)。')
    with open(OUT / 'version_roles_current.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(new[r['page_path']])
    print(f'wrote {OUT}/version_roles_current.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
