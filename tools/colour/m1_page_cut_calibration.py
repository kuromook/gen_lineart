"""M1: ページ単位選別の器具確認 —— 作品(=世代)ごとの行で。

2026-10-24 の Working Discipline の3点拡張(`inbox/notice_discipline_...`、
正本は `foundation` の `doc/CURRENT.md § Working Discipline` と lesson 6、`f112ccf`)に
合わせて書き直した。初版はこの3点すべてに反していた:

  (1) **世代はプールである。** M にとっては**作品が世代**。初版は 1,348ページを
      1本の分布にまとめて印字していた。作品ごとの行で出す
  (2) **世代を跨ぐ走行は世代ごとに既知セルを1つ。** 判定の再現は作品ごとに出す
      (289作品のうち何作品が再現したか。合算の mismatch 0 では世代ずれが映らない)
  (3) **器具と解釈を別に持ち、解釈にも登録を要求する。** `measure_*` は数を返すだけ、
      `judge_*` は**事前登録した予測**に対する PASS/FAIL を返す。予測は docstring に
      書いてから走らせる

較正できるのはページ側の棒だけである。画素側の棒(彩度 >= 24)はこの在庫からは
較正できない —— ページあたりの彩度ヒストグラムが無い。
"""
import csv
import sys
from collections import defaultdict

import numpy as np

BAR = 0.02          # 抽出側の棒。較正されていない(借りた数として使わない)
TILE = None


def load(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ 器具(測る)

def measure_known_cell(pages, works):
    """抽出側の作品単位の判定を、作品ごとに再現する。返り値は作品ごとの真偽。"""
    by = defaultdict(list)
    for r in pages:
        by[(r['年'], r['作品'])].append(float(r['色画素率']))
    out = {}
    for w in works:
        k = (w['年'], w['作品'])
        v = by[k]
        n = sum(1 for x in v if x >= BAR)
        j = '全カラー' if n == len(v) else ('全モノクロ' if n == 0 else '混在')
        out[k] = ((len(v), n, j)
                  == (int(w['測定ページ']), int(w['カラーページ']), w['判定']))
    return out, by


def measure_zero_share(by, jm):
    """判定ごとに「ちょうど 0」のページの割合。プールの正体を数える。"""
    out = {}
    for j in ('全モノクロ', '混在', '全カラー'):
        v = [x for k in by for x in by[k] if jm[k] == j]
        out[j] = (len(v), sum(1 for x in v if x == 0.0))
    return out


def measure_per_work(by, jm, judgement):
    """作品ごとの行。n, min, 中央, max, 作品内の幅。プールしない。"""
    rows = []
    for k in sorted(by):
        if jm[k] != judgement:
            continue
        v = sorted(by[k])
        rows.append({'work': k, 'n': len(v), 'min': v[0],
                     'med': float(np.median(v)), 'max': v[-1],
                     'span': v[-1] - v[0]})
    return rows


def measure_cut(by, jm, cut):
    """棒を判定ごとに当てたときに残るページ数。作品を跨いで平均しない。"""
    out = {}
    for j in ('全モノクロ', '混在', '全カラー'):
        v = [x for k in by for x in by[k] if jm[k] == j]
        out[j] = (len(v), sum(1 for x in v if x > cut))
    return out


# -------------------------------------------------------------- 解釈(判定する)
# 予測は走らせる前に書く。外れたら「解釈が死んだ」と読む(器具ではない)。

def judge_known_cell(ok):
    """予測 P1: 289作品すべてで判定が再現する(作品ごと、合算ではなく)。
    外れた作品が1つでもあれば、その作品は既知セルにならない。"""
    bad = [k for k, v in ok.items() if not v]
    return len(bad) == 0, f'再現しなかった作品 {len(bad)} / {len(ok)}', bad


def judge_zero_is_monochrome(zs):
    """予測 P2: 「ちょうど 0」のページは全モノクロ作品が作っている。
    具体的に: 全モノクロ作品で 95% 以上、全カラー作品で 5% 以下。
    外れたら「0 の山は世代の性質ではない」。"""
    mono = zs['全モノクロ'][1] / zs['全モノクロ'][0]
    colr = zs['全カラー'][1] / zs['全カラー'][0]
    return (mono >= 0.95 and colr <= 0.05), f'全モノクロ {mono:.3f} / 全カラー {colr:.3f}'


def judge_between_dominates(rows):
    """予測 P3: 色画素率の広がりは**作品間**が主で、作品内は小さい。
    具体的に: 測定5ページある作品の「作品内の幅」の中央が、
    作品の中央値の作品間の幅の **1/3 以下**。
    外れたら、プールした1本の幅(36倍)は作品間の効果として読めない ——
    作品内でも大きく動いているので、**どちらの項も大きい**ことになる。"""
    n5 = [r for r in rows if r['n'] >= 5]
    if len(n5) < 3:
        return None, '測定5ページある作品が3作品未満で判定しない'
    within = float(np.median([r['span'] for r in n5]))
    meds = [r['med'] for r in n5]
    between = max(meds) - min(meds)
    return (within <= between / 3.0,
            f'作品内の幅 中央 {within:.4f} / 作品間の幅 {between:.4f} '
            f'(比 {within/between:.2f}、棒は 0.33)')


# --------------------------------------------------------------------- 走行

def main():
    pages = load('inbox/work_color_mode_pages.csv')
    works = load('inbox/work_color_mode.csv')
    jm = {(w['年'], w['作品']): w['判定'] for w in works}

    ok, by = measure_known_cell(pages, works)
    good, msg, bad = judge_known_cell(ok)
    print(f'P1 既知セル(作品ごと): {msg} -> {"PASS" if good else "FAIL"}')
    if not good:
        print('  ', bad[:10])
        print('再現しないので他を読まない(規律2)')
        return 1
    print(f'   pages {len(pages)} / works {len(works)}')

    zs = measure_zero_share(by, jm)
    print('\nちょうど 0 のページ(判定ごと。プールしない):')
    for j, (n, z) in zs.items():
        print(f'  {j:<6} {n:5d}ページ中 {z:5d} が 0  ({100*z/n:5.1f}%)')
    good2, msg2 = judge_zero_is_monochrome(zs)
    print(f'P2 0 の山は全モノクロ作品が作っている: {msg2} '
          f'-> {"PASS" if good2 else "FAIL"}')

    rows = measure_per_work(by, jm, '全カラー')
    print(f'\n全カラー {len(rows)}作品 —— 作品(世代)ごとの行:')
    print(f'  {"作品":<24} {"n":>2}  {"min":>7} {"中央":>7} {"max":>7} {"幅":>7}')
    for r in rows:
        print(f'  {r["work"][0]+"/"+r["work"][1]:<24} {r["n"]:2d}  '
              f'{r["min"]:7.4f} {r["med"]:7.4f} {r["max"]:7.4f} {r["span"]:7.4f}')
    n1 = sum(1 for r in rows if r['n'] == 1)
    print(f'  測定5ページある作品 {sum(1 for r in rows if r["n"]>=5)} / '
          f'1ページだけ {n1} —— **1ページの作品は作品内の幅が読めない**')
    good3, msg3 = judge_between_dominates(rows)
    print(f'P3 広がりは作品間が主: {msg3} -> '
          f'{"PASS" if good3 else "FAIL" if good3 is False else "判定せず"}')

    print('\n棒を当てたときに残るページ(判定ごと。作品を跨いで平均しない):')
    for cut, lab in ((0.0, '> 0(この木が登録した棒)'), (BAR, f'> {BAR}(抽出側の棒)')):
        c = measure_cut(by, jm, cut)
        print(f'  {lab}')
        for j, (n, k) in c.items():
            print(f'    {j:<6} {k:5d} / {n:5d}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
