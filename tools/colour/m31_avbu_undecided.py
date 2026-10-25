"""M31: 作者の問い D の回答を当てる。**「わからない」なので札を埋めない**。

作者の回答(逐語。`decided_by` にこのまま):

> D わからない 同じにみえる ただのバックアップのようだ

(`2016\\avbu\\avb` ↔ `2016\\avbu\\avb_bk`、4塊)

**「わからない」が先に来ていて、「〜ようだ」も断定ではない**(基盤)。
→ **役割は `undecided`。`duplicate_content` にも `wip` にも `branch` にもしない。**
**`_bk` という名前も根拠にしない**(名前は点検ではない)。

**いまの台帳では4塊とも `final`/`superseded`** で、**根拠は `identical_metadata`**
(= 統計量の署名が一致 → 年と辞書順で機械的に選んだ)。**作者が判定できない以上、
その機械の選択を「決着」として残せない。** → **`undecided` に戻す**(落とさない)。

**記録(基盤)**: *人に判定を頼む前に計算可能な軸を測る* の裏返しで、
**人が判定できなかったとき、計算可能な軸が唯一の道になる。** D がその一例目。
→ 画素の照合(sha256)が主経路。

**期待(走らせる前に書く)**:

| | 内容 |
|---|---|
| **KC-A1** | 当たる塊は **4**(30・103・104・105) |
| **KC-A2** | 頁の母集団 **818 → 822**(`superseded` 4頁が戻る) |
| **KC-A3** | 採点 **796 → 800** / 主値(仕上がった頁) **763 → 767** |
| **KC-A4** | **他の頁の役割は1つも変わらない** |
"""
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m25_author_version_answers as m25                 # noqa: E402

SRC = Path('results/m30_branches/version_roles_current.csv')
OUT = Path('results/m31_avbu'); OUT.mkdir(parents=True, exist_ok=True)
VERBATIM = ('作者の回答(2026-10-26): D わからない 同じにみえる ただのバックアップのようだ'
            ' / **作者も判定できない。** 画素の照合(sha256)待ち')
PRIOR = {'30', '103', '104', '105'}
EXPECT = {'pages': 822, 'scoring': 800, 'done': 767}


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    byc = collections.defaultdict(list)
    for r in rows:
        if r['dup_cluster_id']:
            byc[r['dup_cluster_id']].append(r)
    hit = {ci for ci, g in byc.items()
           if all('\\avbu\\' in r['page_path'] for r in g)}
    before_p = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    before_s = m25.scoring_population(new)
    before_done = len(before_s) - sum(
        1 for p in before_s if new[p]['version_role'] == 'wip')
    print(f'== 当たる塊 {len(hit)}  {sorted(hit, key=int)} ==')
    moved = []
    for ci in sorted(hit, key=int):
        for r in byc[ci]:
            q = r['page_path']
            moved.append((ci, q, new[q]['version_role']))
            new[q]['version_role'] = 'undecided'
            new[q]['decided_by'] = VERBATIM
            new[q]['superseded_by'] = ''
    for ci, q, old in moved:
        print(f'  塊{ci:>4} {old:>12} -> undecided  {q[14:]}')
    pages = sum(1 for v in new.values() if v['version_role'] != 'superseded')
    sc = m25.scoring_population(new)
    wip = sum(1 for p in sc if new[p]['version_role'] == 'wip')
    print(f'\n== 既知セル ==')
    ok = True
    for k, got, want in (('KC-A1 塊の数', len(hit), len(PRIOR)),
                         ('KC-A1 塊の顔ぶれ', int(hit == PRIOR), 1),
                         ('KC-A2 頁の母集団', pages, EXPECT['pages']),
                         ('KC-A3 採点の母集団', len(sc), EXPECT['scoring']),
                         ('KC-A3 主値(仕上がった頁)', len(sc) - wip,
                          EXPECT['done']),
                         ('KC-A4 他が動いていない',
                          sum(1 for p, v in new.items()
                              if v['version_role'] != dict(
                                  (r['page_path'], r['version_role'])
                                  for r in rows)[p]
                              and v['dup_cluster_id'] not in hit), 0)):
        good = got == want
        ok &= good
        print(f'  {k:26s} {got:>5} / 期待 {want:>5}  '
              f'{"PASS" if good else "**FAIL**"}')
    print(f'  (前) 頁 {before_p} / 採点 {len(before_s)} / 主値 {before_done}')
    if not ok:
        print('\n**既知セルが合わない。表を書かない。**')
        return 1
    role = collections.Counter(v['version_role'] for v in new.values())
    print(f'\n== 役割 ==')
    for k, c in role.most_common():
        print(f'  {k:20s} {c:>4}')
    print(f'\n== 主値の母集団は **{len(sc) - wip}**。'
          f'ただし **sha256 待ちの暫定が6頁** ==')
    print('  塊17・22(`branch`、署名 Jaccard 1.000)と 塊30・103・104・105'
          '(`undecided`、作者も判定できない)')
    print('  **画素が一致したら A の規則(代表1頁)に寄せる** → 主値は '
          f'{len(sc) - wip} - 6 = **{len(sc) - wip - 6}**')
    with open(OUT / 'version_roles_current.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(new[r['page_path']])
    print(f'\nwrote {OUT}/version_roles_current.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
