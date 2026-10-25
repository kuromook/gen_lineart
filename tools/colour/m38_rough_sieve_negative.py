"""M38: `rough` の篩を**否定形**で作る(基盤の指示)。
あわせて**曖昧な1頁の両方の数**(758 / 759)を印字する。

**なぜ否定形か**: `下絵` は作者のテンプレートの標準の層で **823頁中800頁**に在る。
→ **「`下絵` が在るか」は仕上がった頁の性質**であって、下絵の印ではない。
**聞くべきは「`下絵` 以外がほとんど無いか」。**

**量の選択(閾値の問題ではない)**:

> **テンプレートの層群(`下絵`・`用紙`・`断ち落とし`・トンボ・ノンブル・`調整`・ラフ系)の
> 外にある描画層の、数ではなく描画比の和。**
> 仕上がった頁ならそこに大半の墨が在り、下絵だけの頁ならほぼ0になる。

**切り方(登録。結果を見てから動かさない)**: **作者が `rough` と確認した層群の最大値**
(`koa` の5ファイルの最大 = **0.020**)**以下**を候補にする。
**閾値を自分で選ばず、作者の札が付いた例の最大値を使う。**

**測った値(走らせた結果、この docstring に貼る)**:

| | `下絵`系の外の描画比の和 |
|---|---:|
| `koa\\koakuma.clip` | 0.001 |
| `koa\\koakuma2..clip` | 0.020 |
| `koa\\koakuma2.clip` | 0.018 |
| `koa\\koakuma2b..clip` | 0.018 |
| `koa\\koakuma_check.clip` | 0.003 |
| **仕上がった頁の例** | `toho\\koakuma` **10.93** / `koakuma2` **10.95** / `illust201710\\page0003` **6.15** / `skima11\\skima12chara` **55.01** |
| 872頁の分布 | 最小 0.000 / 下位5% 0.016 / **中央 0.748** / 最大 67.538 |

**300倍の隔たりがある。** カラー層の数で切るとモノクロ原稿186頁を巻き込んだのとは別物。
"""
import collections
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m19_version_resolution as m19                     # noqa: E402
import m25_author_version_answers as m25                 # noqa: E402

SRC = Path('results/m37_done_b/version_roles_current.csv')
OUT = Path('results/m38_rough_sieve'); OUT.mkdir(parents=True, exist_ok=True)
TMPL = re.compile(r'(下絵|下書|下描|ラフ|rough|用紙|断ち落とし|トンボ|ノンブル'
                  r'|PICO|調整)', re.I)
CUT = 0.020           # 作者が確認した下絵の最大値(koa\koakuma2..clip)
AMBIGUOUS = r'F:\manuscripts\2017\toho\koa\koakuma2.clip'   # 3つ目(作者は「二つ」)


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    new = {r['page_path']: dict(r) for r in rows}
    lay = list(csv.DictReader(open(m19.CSV872, encoding='utf-8')))
    bp = collections.defaultdict(list)
    for r in lay:
        bp[r['page_path']].append(r)

    def outside(p):
        s = 0.0
        for r in bp[p]:
            if r['folder'] == 'True':
                continue
            d = m19.num(r['drawn_ratio'], 0) or 0
            if d <= 0 or TMPL.search(r['name'] or ''):
                continue
            s += d
        return s

    scored = [q for q, v in new.items() if v['use'].startswith('採点する')]
    vals = {q: outside(q) for q in new}
    cand = sorted((q for q in scored if vals[q] <= CUT), key=lambda q: vals[q])
    print(f'== 否定形の篩(テンプレートの外の描画比の和 <= {CUT})==')
    print(f'  主値の母集団 {len(scored)} 頁のうち **{len(cand)} 頁**'
          f'  (カラー層の数で切ったときは186頁。**量の選択の問題だった**)')
    for q in cand:
        print(f'    {vals[q]:7.4f}  層 {len(bp[q]):>3}  '
              f'{new[q]["version_role"]:>9}  {q[14:]}')
    if not cand:
        print('    **0頁**(規律7)')

    print(f'\n== 曖昧な1頁の両方の数(腕2つを床1つに当てない、の母集団版)==')
    both = len([q for q, v in new.items() if v['use'].startswith('採点する')])
    print(f'  3頁を `rough` として除外(いまの数) : **{both}**')
    print(f'  **曖昧な1頁(`koa\\koakuma2.clip`)を含める: {both + 1}**')
    print(f'  → **どちらが正かは作者にしか決まらない。確認が付くまで両方を同じ行に。**')
    print(f'  (`koa` の5ファイルのうち、作者が「二つ」と述べ、こちらは3つに当てた)')

    with open(OUT / 'rough_candidates_negative.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['page_path', 'テンプレート外の描画比の和', '層の数',
                    'いまの version_role', 'いまの use'])
        for q in cand:
            w.writerow([q, f'{vals[q]:.6f}', len(bp[q]),
                        new[q]['version_role'], new[q]['use']])
    print(f'\nwrote {OUT}/rough_candidates_negative.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
