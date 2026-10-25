#!/usr/bin/env python
"""L19 の集計: 作者の語と器具(隣接層の argmax)の一致を数える。

    python tools/evaluation/l19_agreement.py

入力  labels/fill_words.jsonl (追記型。order ごとに最後の記録を採る)
      results/l19_ui/queue.json (行列。母集団と why)
出力  標準出力 + results/l19_ui/agreement.csv (面ごと)

棒は無い。L19 は器具の採点ではなく UI の実測なので、数えるものを全部印字する。
プールは行を分ける(規律: 世代もプール)。
"""
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / 'labels' / 'fill_words.jsonl'
QUEUE = ROOT / 'results' / 'l19_ui' / 'queue.json'
OUT = ROOT / 'results' / 'l19_ui' / 'agreement.csv'

# UI の語彙が増えた時刻(ts の空白から。§再出題の節で使う)
SEG = [('S1 服/フキダシ以前', '2026-10-25T13:16:23', '2026-10-25T13:24:31'),
       ('S2 黒消し以前', '2026-10-25T13:27:43', '2026-10-25T13:38:11'),
       ('S3 語彙そろう', '2026-10-25T13:41:28', '2026-10-25T15:35:58'),
       ('S4 出し直し', '2026-10-25T15:40:00', '2026-10-25T23:59:59')]
REASK = Path(__file__).resolve().parents[2] / 'results/l19_ui/reask.json'
DUP = Path(__file__).resolve().parents[2] / 'results/l21_dirtiness/duplicate_pages.json'
WORD_CHOICES = ('word', 'free')


def load():
    q = json.loads(QUEUE.read_text())
    rows = [json.loads(l) for l in LAB.open()]
    for r in rows:              # 自由入力で正しい語を書いた分は語の回答として数える
        if r['choice'] == 'free' and r.get('free'):
            r['choice'], r['word'] = 'word', r['free'].strip()
            r['agree'] = int(r['word'] == r['top_word'] and bool(r['top_word']))
    last, n_rec = {}, Counter()
    for r in rows:
        last[r['order']] = r
        n_rec[r['order']] += 1
    return q, last, n_rec, rows


def seg_of(ts):
    for name, a, b in SEG:
        if a <= ts <= b:
            return name
    return '?'


def dup_pages():
    """同じ頁の二状態に関わる頁 (層の md5 で検出。`l21_duplicate_pages.py`)。

    **この階級を黙ってはいけない**: 器具の一致が 0.5189 / 0.2747 と 2 倍違い、
    551 面のうち 318 面が重複側から来ているので、まとめた数は重複側に引かれる
    (基盤 `ruling_l18_authorised_and_l19_queries_20261025.md` §6)。
    """
    if not DUP.exists():
        return set()
    return set(json.loads(DUP.read_text())['pages'])


def pct(a, b):
    return f'{a / b:.4f}' if b else '  -   '


def main():
    q, last, n_rec, all_rows = load()
    n_all_rec = len(all_rows)
    queue = q['queue']
    print(f"行列 {q['n']} (要判断 {q['n_ask']} + 自動の確認 {q['n_check']}"
          f" / 自動 {q['n_auto']}、全面 {q['n_all']})")
    print(f'記録 {n_all_rec} 行 / 回答した問 {len(last)}'
          f'  未回答 {q["n"] - len(last)}'
          f'  (やり直しを含む問 {sum(1 for v in n_rec.values() if v > 1)})')
    answered = [last[i] for i in sorted(last)]

    # --- 1. 選別は測定である: 回答の種類を全部数える -------------------------
    print('\n[1] 回答の種類 (プールごと。語でない回答も落とさず数える)')
    hdr = ['語', '自由入力', '隙間埋め(線画)', '隙間埋め(別層ベタ)', '混在', '判断保留']
    kinds = ['word', 'free', 'gapfill_line', 'gapfill_fill', 'mixed', 'skip']
    print(f'  {"プール":<8}{"問":>5} ' + ' '.join(f'{h:>12}' for h in hdr))
    for pool in ('ako6', 'ako7', '両方'):
        sub = [r for r in answered if pool == '両方' or r['work'] == pool]
        c = Counter(r['choice'] for r in sub)
        print(f'  {pool:<8}{len(sub):>5} '
              + ' '.join(f'{c[k]:>5} {pct(c[k], len(sub)):>6}' for k in kinds))

    # --- 2. 一致率 -----------------------------------------------------------
    dups = dup_pages()
    print('\n[2] 一致率 (器具 = 縁に接する線画層の argmax を語に写したもの)')
    print('    **重複頁の階級を必ず並べる** —— 同じ頁の二状態が 36/65 頁あり、')
    print('    その階級と残りで一致が 2 倍違うので、まとめた数は重複側に引かれる')
    print('    分母A = 回答した問すべて (語でない回答は不一致として数える)')
    print('    分母B = 作者が語を1つ選んだ問だけ')
    print(f'  {"プール":<8}{"why":<7}{"A の分母":>9}{"一致/A":>9}{"B の分母":>9}{"一致/B":>9}')
    for pool in ('ako6', 'ako7', '両方'):
        for why in ('ask', 'check', 'すべて'):
            sub = [r for r in answered
                   if (pool == '両方' or r['work'] == pool)
                   and (why == 'すべて' or r['why'] == why)]
            if not sub:
                continue
            b = [r for r in sub if r['choice'] == 'word']
            ag = sum(r['agree'] for r in sub)
            print(f'  {pool:<8}{why:<7}{len(sub):>9}{pct(ag, len(sub)):>9}'
                  f'{len(b):>9}{pct(ag, len(b)):>9}')

    _d = json.loads(DUP.read_text()) if DUP.exists() else {'thr': None, 'pairs': []}
    print(f'\n  重複/非重複の階級 (分母A。基盤の裁定 §6 で必須)。'
          f'**階級の作り方 = 層の md5 の Jaccard >= {_d["thr"]}** '
          f'({len(_d["pairs"])} 組 / 関わる頁 {len(dups)})')
    print(f'  {"プール":<8}{"階級":<12}{"問":>5}{"一致/A":>9}{"いつも髪/A":>11}')
    for pool in ('ako6', 'ako7', '両方'):
        for nm, want in (('重複頁', True), ('それ以外', False)):
            sub = [r for r in answered
                   if (pool == '両方' or r['work'] == pool)
                   and ((r['pid'] in dups) == want)]
            if not sub:
                continue
            ag = sum(r['agree'] for r in sub)
            hair = sum(1 for r in sub if r['choice'] == 'word' and r['word'] == '髪')
            print(f'  {pool:<8}{nm:<12}{len(sub):>5}{pct(ag, len(sub)):>9}'
                  f'{pct(hair, len(sub)):>11}')
    print('  **ako6 と ako7 は互いに素なプールではない** —— 551 面のうち 24 面が、')
    print('  2018 と 2019 で絵を共有する 1 組 (`ako6-18.clip` / `ako7/page0073.clip`) から来る')

    # --- 3. 自明なベースライン: いつも「髪」 ---------------------------------
    print('\n[3] 自明なベースライン (器具を使わず、いつも同じ語を答える)')
    for pool in ('ako6', 'ako7', '両方'):
        sub = [r for r in answered if pool == '両方' or r['work'] == pool]
        b = [r for r in sub if r['choice'] == 'word']
        top = Counter(r['word'] for r in b).most_common(3)
        inst_a = sum(r['agree'] for r in sub) / len(sub)
        line = '  '.join(f'{w} {pct(n, len(b))}' for w, n in top)
        print(f'  {pool:<8} 器具 {inst_a:.4f} (分母A)   最多の語(分母B): {line}')
        if top:
            print(f'           いつも「{top[0][0]}」= {pct(top[0][1], len(sub))} (分母A)'
                  f' / {pct(top[0][1], len(b))} (分母B)')

    # --- 4. purity は一致を予測するか ---------------------------------------
    print('\n[4] purity の階級ごとの一致 (器具が自分の確信度として出せる唯一の量)')
    bins = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]
    print(f'  {"purity":<14}{"問":>5}{"一致/A":>9}{"語の問":>7}{"一致/B":>9}')
    for lo, hi in bins:
        sub = [r for r in answered if lo <= r['purity'] < hi]
        b = [r for r in sub if r['choice'] == 'word']
        ag = sum(r['agree'] for r in sub)
        print(f'  [{lo:.2f},{hi:.2f})  {len(sub):>5}{pct(ag, len(sub)):>9}'
              f'{len(b):>7}{pct(ag, len(b)):>9}')

    # --- 5. 語ごと -----------------------------------------------------------
    print('\n[5] 語ごと (縦 = 作者の語、横 = 器具がその面に出した語)')
    by_word = defaultdict(Counter)
    for r in answered:
        if r['choice'] == 'word':
            by_word[r['word']][r['top_word'] or '(なし)'] += 1
    print(f'  {"作者の語":<10}{"問":>5}{"一致":>7}{"率":>8}   器具の答(上位3)')
    for w, c in sorted(by_word.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        top3 = '  '.join(f'{k}:{v}' for k, v in c.most_common(3))
        print(f'  {w:<10}{n:>5}{c[w]:>7}{pct(c[w], n):>8}   {top3}')

    print('\n[5b] 器具の答ごと (その語を出したとき当たる率 = 精度)')
    inst = defaultdict(Counter)
    for r in answered:
        if r['choice'] == 'word':
            inst[r['top_word'] or '(なし)'][r['word']] += 1
    print(f'  {"器具の答":<10}{"問":>5}{"一致":>7}{"率":>8}   作者の答(上位3)')
    for w, c in sorted(inst.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        top3 = '  '.join(f'{k}:{v}' for k, v in c.most_common(3))
        print(f'  {w:<10}{n:>5}{c[w]:>7}{pct(c[w], n):>8}   {top3}')

    # --- 6. 候補に入っていたか ----------------------------------------------
    print('\n[6] 作者の語が器具の候補(縁に接した層の語)に入っていたか')
    for pool in ('ako6', 'ako7', '両方'):
        sub = [r for r in answered
               if (pool == '両方' or r['work'] == pool) and r['choice'] == 'word']
        inn = sum(1 for r in sub if r['word'] in (r['word_share'] or {}))
        print(f'  {pool:<8}{len(sub):>5} 候補内 {inn:>4} {pct(inn, len(sub))}'
              f'   候補外 {len(sub) - inn:>4} {pct(len(sub) - inn, len(sub))}')
    out = Counter(r['word'] for r in answered
                  if r['choice'] == 'word' and r['word'] not in (r['word_share'] or {}))
    print('  候補外の語の中身:', '  '.join(f'{k}:{v}' for k, v in out.most_common()))

    # --- 7. 隙間埋め --------------------------------------------------------
    print('\n[7] 隙間埋め (= そもそも「何かのベタ」ではない面)')
    for pool in ('ako6', 'ako7', '両方'):
        sub = [r for r in answered if pool == '両方' or r['work'] == pool]
        g = [r for r in sub if r['choice'].startswith('gapfill')]
        gl = sum(1 for r in g if r['choice'] == 'gapfill_line')
        print(f'  {pool:<8}{len(sub):>5} 隙間埋め {len(g):>4} {pct(len(g), len(sub))}'
              f'  (線画の抜け {gl} / 別層ベタの隙間 {len(g) - gl})')

    # --- 8. 所要時間と再出題の必要な範囲 ------------------------------------
    ms = sorted(r['ms'] for r in answered if r['ms'] > 0)
    print(f'\n[8] 1問の所要 中央 {statistics.median(ms) / 1000:.1f}s'
          f'  四分 {ms[len(ms) // 4] / 1000:.1f}/{ms[3 * len(ms) // 4] / 1000:.1f}s'
          f'  合計 {sum(ms) / 60000:.0f}分')
    print('\n[9] UI の語彙が増えた区間と、出し直しの効果')
    segc = Counter(seg_of(r['ts']) for r in answered)
    for name, _, _ in SEG:
        print(f'  {name:<16}{segc[name]:>5} 問 (最後の回答がこの区間にある問)')
    if segc['?']:
        print(f'  {"区間外":<16}{segc["?"]:>5} 問')
    if REASK.exists():
        got = sorted(set(json.loads(REASK.read_text())))
        before, after = {}, {}
        for r in all_rows:
            if r['order'] in set(got):
                (after if seg_of(r['ts']) == 'S4 出し直し' else before)[r['order']] = r
        done = [o for o in got if o in after]
        print(f'  出し直し {len(got)} 問 → 済み {len(done)}')
        if done:
            def key(r):
                return (r['choice'], r['word'])
            ch = [o for o in done if key(before[o]) != key(after[o])]
            ab = sum(before[o]['agree'] for o in done)
            aa = sum(after[o]['agree'] for o in done)
            print(f'    答が変わった問 {len(ch)}/{len(done)}: '
                  + '  '.join(f'{o}:{before[o]["word"] or before[o]["choice"]}'
                              f'->{after[o]["word"] or after[o]["choice"]}' for o in ch))
            print(f'    この {len(done)} 問の一致/A  前 {ab}/{len(done)} = {ab / len(done):.4f}'
                  f'   後 {aa}/{len(done)} = {aa / len(done):.4f}')

    # --- 11. 開示 (棒ではない。UI が何を既定で選ぶかは設計値、規律6) ---------
    print('\n[11] 開示: 候補から「肌」を外すと器具はどう変わるか')
    print('     (結果を見てから決めた規則なので PASS ではない。UI の既定値の設計に使う)')
    for pool in ('ako6', 'ako7', '両方'):
        sub = [r for r in answered
               if (pool == '両方' or r['work'] == pool) and r['choice'] == 'word']
        ag = ag2 = 0
        for r in sub:
            sh = {k: v for k, v in (r['word_share'] or {}).items() if k != '肌'}
            ag += r['agree']
            if sh and max(sh, key=sh.get) == r['word']:
                ag2 += 1
        print(f'  {pool:<8}{len(sub):>5} そのまま {pct(ag, len(sub))}'
              f'   肌を外す {pct(ag2, len(sub))}')

    print('\n[12] 作者の語が候補の何番目にあるか (B の分母。UI の並べ方の設計値)')
    rank = Counter()
    for r in answered:
        if r['choice'] != 'word':
            continue
        sh = sorted((r['word_share'] or {}).items(), key=lambda kv: -kv[1])
        ks = [k for k, _ in sh]
        rank[ks.index(r['word']) + 1 if r['word'] in ks else '候補外'] += 1
    nb = sum(rank.values())
    for k in [1, 2, 3, 4, '候補外']:
        print(f'  {str(k):>5} 番目  {rank[k]:>4} {pct(rank[k], nb)}')

    print(f'\n[13] 全面を人手で読む見積り: {len(answered)} 問 {sum(ms) / 60000:.0f}分'
          f' → 全 {q["n_all"]} 面で {sum(ms) / 60000 * q["n_all"] / len(answered):.0f}分'
          f' (自動 {q["n_auto"]} 面も信用できないので、これが実際の作業量)')

    free = [(r['order'], r['free']) for r in answered if r['choice'] == 'free']
    if free:
        print('\n[10] 自由入力の中身:', free)
    print(f"     混在(語を1つに決められない) {sum(1 for r in answered if r['choice'] == 'mixed')} 問")

    with OUT.open('w') as fh:
        fh.write('order,work,pid,comp,region,why,purity,n_lay,n_lay90,'
                 'top_name,top_word,choice,word,agree,ms,ts\n')
        for r in answered:
            fh.write(f'{r["order"]},{r["work"]},{r["pid"]},{r["comp"]},{r["region"]},'
                     f'{r["why"]},{r["purity"]},{r["n_lay"]},{r["n_lay90"]},'
                     f'"{r["top_name"]}",{r["top_word"]},{r["choice"]},'
                     f'{r["word"] or ""},{r["agree"]},{r["ms"]},{r["ts"]}\n')
    print(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
