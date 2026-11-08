# -*- coding: utf-8 -*-
"""L38 **流れ込みの答えを読む**(2026-11-08)。登録: `doc/l38_bleed_registration_20261108.md`。

**この読み手は答えが来た後に書いた。棒は `79eb8c4` と `b2db974` で固定済み**(答えが0行の時点)。
**書くときに答えの分布を見ていない** —— `wc -l` も `Counter` も通していない。
**それでも「先に書いた」より弱い。記録する。** ここに在るのは**登録の文の写し**だけである:

  C1 一様 >= 25% / C3 <= 5% / 5〜25% は保留(詰めるのに要る数を印字)
  C2' 一様の中で「覆い>0」−「覆い 0」 >= 30 ポイント(追補1 で差し替えた)
  C4 腕B の精度 − 基準率 >= 15 ポイント かつ 感度 >= 0.5
  C5 IC(f09 在る / f45 無い)が外れたら**何も読まない**
  C6 答えられなかった率 >= 35% で他の棒を読まない

**腕W は 11-08 に落ちた定義のまま載せる**(`l38_score_armw_20261108.armw`。実体は1つ)。
"""
import collections
import json
import math
import os
import sys
from pathlib import Path

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from l38_score_armw_20261108 import T, armw                         # noqa: E402

ROOT = Path('/home/sh1/deepl/lineart-fill-region')
D = ROOT / 'data' / 'l38_bleed_20261108'
ANS = D / 'l38_bleed_20261108_answers.jsonl'
CROPS = ROOT / 'results' / 'bleed20261108'
YES, NO = 'bleed', 'clean'
IC = {'f09': YES, 'f45': NO}            # **期待値は作者の 11-08 の区分けから**(登録 §4)


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def rate(name, rows, pad=28):
    k = sum(1 for r in rows if r['value'] == YES)
    n = sum(1 for r in rows if r['value'] in (YES, NO))
    p, lo, hi = wilson(k, n)
    print(f'  {name:<{pad}} {k}/{n} = **{p:.1%}**(Wilson {lo:.1%}〜{hi:.1%})')
    return k, n, p, lo, hi


def need(p, width=0.10):
    for n in range(10, 20001):
        _, lo, hi = wilson(round(p * n), n)
        if hi - lo <= width:
            return n
    return None


def arm(name, rows, fires):
    """腕の**精度・感度・覆い**を出す。**基準率と並べて印字する**(条例2)。"""
    ans = [r for r in rows if r['value'] in (YES, NO)]
    f = [r for r in ans if fires(r)]
    tp = sum(1 for r in f if r['value'] == YES)
    pos = sum(1 for r in ans if r['value'] == YES)
    base = pos / len(ans) if ans else 0
    prec = tp / len(f) if f else None
    sens = tp / pos if pos else None
    print(f'  {name:<22} 発火 {len(f):>3}/{len(ans)}(覆い {len(f) / max(1, len(ans)):5.1%})  '
          f'精度 {"—" if prec is None else f"{prec:.1%}"}  '
          f'感度 {"—" if sens is None else f"{sens:.1%}"}  '
          f'(基準率 {base:.1%})')
    return prec, sens, base, len(f)


def main():
    kmap = json.load(open(D / 'map.json', encoding='utf-8'))
    last, raw = {}, 0
    for line in open(ANS, encoding='utf-8'):
        if line.strip():
            r = json.loads(line)
            raw += 1
            last[r['item_id']] = r                 # **最後の行が有効**(規律7)
    rows = []
    for fid, m in kmap.items():
        a = last.get(fid)
        if a:
            rows.append(dict(m, fid=fid, value=a.get('value'),
                             comment=a.get('comment') or ''))
    print(f'答えの在る単位 {len(rows)}/{len(kmap)}(行 {raw}。**追記だけ**)')

    print('\n--- 器具の点検(IC。**期待値は作者の 11-08 の区分けから。自分の軸からは作っていない**)---')
    bad = []
    for fid, want in IC.items():
        hit = [r for r in rows if r['group'] == f'ic_{fid}']
        if not hit:
            print(f'  IC {fid} が答えに無い(**黙って飛ばさない**)')
            bad.append(fid)
            continue
        got = hit[0]['value']
        ok = got == want
        print(f'  IC {fid}  面積 {hit[0]["area"]:>9,}  期待 {want} / 実際 {got}  '
              f'{"一致" if ok else "**外れた**"}')
        if hit[0]['comment']:
            print(f'      一言: {hit[0]["comment"]}')
        if not ok:
            bad.append(fid)
    if bad:
        print(f'\n**C5 IC が外れた({" ".join(bad)})。登録どおり、この走行は何も読まない。**')
        return

    uni = [r for r in rows if r['group'] == 'uniform']
    held = [r for r in uni if r['value'] not in (YES, NO)]
    r6 = len(held) / len(uni) if uni else 0
    print(f'\n--- 答えられなかった率(C6)---\n  {len(held)}/{len(uni)} = {r6:.1%}')
    if r6 >= 0.35:
        print('  **C6 35% 以上。他の棒を読まない。**')
        return
    if r6 >= 0.25:
        print('  **25〜35%。問いの欠陥として記録する**(棒は読む)')

    print('\n--- 主値(一様標本 60 面)---')
    ku, nu, pu, lou, hiu = rate('一様標本', uni)
    print('  **腕0 全部「無い」= 0%、腕1 全部「在る」= 率そのもの(恒等)**(条例2)')
    print('\n  作品ごと(**畳まない**):')
    for w in sorted({r['work'] for r in uni}):
        rate(f'一様 / {w}', [r for r in uni if r['work'] == w])
    print('\n  一様標本の中の小行(**狙い撃ちしていない。この標本の中の数**):')
    rate('帯の覆い > 0', [r for r in uni if r['band_cov'] > 0])
    rate('帯の覆い 0', [r for r in uni if r['band_cov'] == 0])
    rate('帯の覆い >= 0.05', [r for r in uni if r['band_cov'] >= 0.05])
    rate('首あり', [r for r in uni if r['neck']])
    rate('首なし', [r for r in uni if not r['neck']])
    rate('割った片', [r for r in uni if r['origin'] == 'piece'])
    rate('元のままの面', [r for r in uni if r['origin'] == 'kept'])
    ar = sorted(uni, key=lambda r: r['area'])
    for i, nm in ((0, '面積 下 1/3'), (1, '面積 中 1/3'), (2, '面積 上 1/3')):
        rate(nm, ar[i * len(ar) // 3:(i + 1) * len(ar) // 3])

    print('\n--- 腕(**学習なし**。精度・感度・覆いを基準率と並べる)---')
    pb, sb, base, nfb = arm('腕B 帯の覆い > 0', uni, lambda r: r['band_cov'] > 0)
    arm('腕B 帯 >= 0.05', uni, lambda r: r['band_cov'] >= 0.05)
    # 腕W —— **11-08 に落ちた定義のまま**(r = 2、t = 0.5)。マスクから出す
    wfire, miss = {}, 0
    for r in uni:
        mp = CROPS / f'{r["fid"]}_mask.png'
        if not mp.is_file():
            miss += 1
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
        w = armw(m)
        wfire[r['fid']] = (w['n_core'] >= 2) or (w['thin'] >= T)
    arm('腕W 幅(r=2, t=0.5)', [r for r in uni if r['fid'] in wfire],
        lambda r: wfire[r['fid']])
    if miss:
        print(f'    **マスクが無くて腕W に掛けられなかった面 {miss}**(規律9)')
    print('\n  腕F 細い残りの四分位ごとの「在る」率(**学習なしの頻度表**):')
    tq = sorted(uni, key=lambda r: r['thin_frac'])
    for i in range(4):
        part = tq[i * len(tq) // 4:(i + 1) * len(tq) // 4]
        if part:
            rate(f'細い残り 第{i + 1}四分位 '
                 f'({part[0]["thin_frac"]:.3f}〜{part[-1]["thin_frac"]:.3f})', part)

    print('\n--- 棒の判定(登録のとおり)---')
    if pu >= 0.25:
        print(f'  **C1 達成**(一様 {pu:.1%} >= 25%)—— **流れ込みは広い。機械で見つけに行く**')
    elif pu <= 0.05:
        print(f'  **C3 達成**(一様 {pu:.1%} <= 5%)—— **稀。個別の修理で済む。'
              f'面プールの版は差し替えない**')
    else:
        print(f'  **判定保留**(一様 {pu:.1%} は 5〜25% の間)。Wilson の幅 {hiu - lou:.1%}')
        print(f'  **幅を 10 ポイント以下に詰めるのに要る面の数 {need(pu)}**(いま {nu})')
    g1 = [r for r in uni if r['band_cov'] > 0]
    g0 = [r for r in uni if r['band_cov'] == 0]
    k1 = sum(1 for r in g1 if r['value'] == YES)
    n1 = sum(1 for r in g1 if r['value'] in (YES, NO))
    k0 = sum(1 for r in g0 if r['value'] == YES)
    n0 = sum(1 for r in g0 if r['value'] in (YES, NO))
    if n1 and n0:
        d = k1 / n1 - k0 / n0
        print(f"  C2' 覆い>0 {k1}/{n1} = {k1 / n1:.1%} − 覆い 0 {k0}/{n0} = {k0 / n0:.1%}"
              f" = **{d:+.1%}**  "
              f"{'**達成。位置の軸は効く**' if d >= 0.30 else '**未達(30 ポイントに届かない)**'}")
    else:
        print(f"  **C2' は読めない**(覆い>0 {n1} 面 / 覆い 0 {n0} 面)")
    if pb is not None and sb is not None:
        d = pb - base
        ok = d >= 0.15 and sb >= 0.5
        print(f'  C4 腕B 精度 {pb:.1%} − 基準率 {base:.1%} = {d:+.1%} / 感度 {sb:.1%}  '
              f'{"**達成。帯の軸を器具として使う**" if ok else "**未達**"}')
        if not ok:
            print('      **未達の意味(登録に先に書いた)**: 形(腕W)も位置(腕B)も落ちた ——'
                  ' **この定式化ではどの腕も流れ込みを見つけない**')
    else:
        print('  **C4 は読めない**(腕B が1面も発火しない、または「在る」が0)')

    print('\n--- 一言の欄(**面積は出さない**)---')
    for r in rows:
        if r['comment']:
            print(f'    {r["value"]}: {r["comment"]}')
    yes = [r['fid'] for r in rows if r['value'] == YES]
    print(f'\n**「在る」と答えられた単位 {len(yes)}**: {" ".join(sorted(yes))}')
    print('**この走行が決められないこと**: (b) 読みとしての分割の率 / 流れ込みの面積 /'
          ' 小片の側 / 重ね塗りの疑い4面(ユーザーに伺ってから)')


if __name__ == '__main__':
    main()
