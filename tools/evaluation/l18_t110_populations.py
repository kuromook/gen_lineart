# -*- coding: utf-8 -*-
"""T110: 全面版と除外版で、**どの面が採点されたか**を腕ごとに数える。

2026-10-26 (基盤の指摘): この数は便の表にしか無く、走行ログを引いても出てこなかった。
**消える場所を指す引用は、いま歩けても根拠にならない。** だから歩ける形に落とす。

見つけたこと: 除外の門 `if EXCLUDE is not None and gi not in cname: continue`
(`l18_propagate.py:116`) は **`--exclude-gapfill` のときだけ働く。** `cname` には
名前が付いた面しか入らず、名前は**元の位置で `adj != 0`** の面にだけ付く。だから

  ・全面版の**対照**は、名前の無い面も平行移動して置き、採点していた
  ・本来の腕は元の位置で `adj == 0` → `continue` なので、**最初から持っていない**
  ・除外版では門が働いて、対照も名前のある面だけになった

**除外版は45面の除去と対照の母集団の縮小を同時に受けている。** 腕の比較は読めるが、
**対照と床の版をまたいだ比較は読めない**(45面の寄与と名前の無い面の寄与が分離していない)。

  ./venv/bin/python tools/evaluation/l18_t110_populations.py
"""
import collections, csv, os, sys

OUT = '/home/sh1/deepl/lineart-fill-region/results/l18_propagate'
RUNS = (('全面', f'{OUT}/regions_both.csv', f'{OUT}/run_full_again.log'),
        ('隙間埋けを外した版', f'{OUT}/regions_both_nogapfill.csv',
         f'{OUT}/run_nogapfill.log'))


def tally(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    n_row = collections.Counter(r['arm'] for r in rows)
    keys = collections.defaultdict(set)
    for r in rows:
        keys[r['arm']].add((r['pid'], r['comp'], r['region']))
    return rows, n_row, keys


def main():
    res, lines = {}, []

    def say(s=''):
        print(s, flush=True)
        lines.append(s)

    say('T110: 版ごとに「どの面が採点されたか」を腕ごとに数える')
    say('  **面の鍵 (pid, comp, region) は一意ではない**(T109。同じコマが複数回返り '
        '`gi` が振り直される)。')
    say('  ここで数えているのは**鍵**であって面の数ではない。版の間の差を見るための量である')
    say()
    say(f'  {"版":<22}{"行 real":>9}{"行 shuffled":>13}{"鍵 real":>9}'
        f'{"鍵 shuffled":>13}{"shuffled にしか無い鍵":>23}')
    for lab, csvp, logp in RUNS:
        if not os.path.exists(csvp):
            say(f'  {lab:<22}  **{csvp} が無い**')
            continue
        rows, n_row, keys = tally(csvp)
        only = keys['shuffled'] - keys['real']
        res[lab] = dict(row_real=n_row['real'], row_sh=n_row['shuffled'],
                        k_real=len(keys['real']), k_sh=len(keys['shuffled']),
                        k_only=len(only), log=os.path.basename(logp))
        say(f'  {lab:<22}{n_row["real"]:>9}{n_row["shuffled"]:>13}'
            f'{len(keys["real"]):>9}{len(keys["shuffled"]):>13}{len(only):>23}')
    if len(res) == 2:
        a, b = res['全面'], res['隙間埋けを外した版']
        d_sh = a['row_sh'] - b['row_sh']
        say()
        say(f'  shuffled の行は {d_sh} 減った。**45面で説明できるのは多くても '
            f'{45 * 5} 行**(面 45 × seed 5)なので、')
        say(f'  残り **{d_sh - 45 * 5} 行は名前の無い面**である')
        say(f'  real の行は {a["row_real"]} -> {b["row_real"]} '
            f'({a["row_real"] - b["row_real"]} 減)、鍵は {a["k_real"]} -> {b["k_real"]} '
            f'({a["k_real"] - b["k_real"]} 減)。')
        say('  **本来の腕の版の差は、定義上ちょうど外した45面だけである**'
            '(両腕から同じ面を落とす。L17 §5)。')
        say(f'  行が {a["row_real"] - b["row_real"]} しか減らないのは、'
            '45面のうち**元の位置で線に接していない / 置けない面は、'
            'もともと行を持っていない**ためで、')
        say('  鍵が 39 なのは **鍵が一意でない**(T109)ためである。'
            '**45・43・39 は別の量で、どれも「面の数」ではない**')
        say(f'  shuffled にしか無い鍵は {a["k_only"]} -> {b["k_only"]}')
        say()
        say('  **読める / 読めない**')
        say('    ・腕の比較は**読める**。本来の腕はどちらの版でも名前のある面だけを見ている')
        say('    ・**対照と床の版をまたいだ比較は読めない。** 45面の寄与と'
            '名前の無い面の寄与が分離していない')
        say('    ・**各版の判定は自分の走行の中で完結しているので有効**'
            '(床は自分の走行の対照の関数。T97 の規律)')
        say('    ・**対照の列は版の中でだけ意味を持つ。版をまたいで引き算しない**')
    say()
    say('  出所: ' + ' / '.join(f'{lab} = {os.path.basename(c)} + {res[lab]["log"]}'
                                for lab, c, _ in RUNS if lab in res))

    with open(f'{OUT}/t110_populations.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['version', 'rows_real', 'rows_shuffled', 'keys_real',
                    'keys_shuffled', 'keys_shuffled_only', 'source_log'])
        for lab in ('全面', '隙間埋けを外した版'):
            if lab in res:
                r = res[lab]
                w.writerow([lab, r['row_real'], r['row_sh'], r['k_real'],
                            r['k_sh'], r['k_only'], r['log']])
    with open(f'{OUT}/t110_populations.log', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'-> {OUT}/t110_populations.csv')
    print(f'-> {OUT}/t110_populations.log')
    return 0


if __name__ == '__main__':
    sys.exit(main())
