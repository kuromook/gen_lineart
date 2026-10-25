"""M19: 追補12 §K2 の規則で**重複の塊を分類し、代表(完成版)を選ぶ**。
結果は**抽出側の DB に列として戻す**ための CSV にする(作者の指示、2026-10-25)。

登録(追補12 §K2。走らせる前に commit した `1ad0f9e`):
  R-1 フォルダーの年が新しい方を採る(作者の 1・2・5)
  R-2 年が同じなら `.clip` の更新時刻が新しい方(**納品に時刻の列が無いので保留**)
  R-3 どちらも決まらなければ作者に聞く(**ここだけ。最小限に**)
  名前は選別器にしない。接尾辞は選別の**後**に一致率を印字する falsifier(P25)

塊の種類(追補12 §K2 の表):
  A 年を跨ぐ同一頁(同じ作品・年が違う) / B 同じ年の別版(同じ作品・同じ年) /
  C 同じ作品の別の頁(版の問題ではない。**両方残す**) / D 作品を跨ぐ

**種類の判定には名前を使う**(頁の名幹の前方一致)。**代表の選別には使わない。**
名幹に前方一致が無く、同じ作品・同じ年の組は **C として両方残す**(安全側)。
**落とすより残す。** 迷った組は作者に出す候補として印字する。

**年の比較**: `2019b` `2024c` のような接尾辞つきの年がある。
**数字の年だけで比較し、接尾辞の違いは「年が同じ」として R-2 に回す**
(`2023` と `2023b` のどちらが新しいかを名前から決めない)。

DB に戻す列(この器具が出す):
  `dup_cluster_id`  塊の番号(重複なしは空)
  `cluster_kind`    A / B / C / D
  `version_role`    final / superseded / sibling / **undecided** / single
                    (**undecided は落とさない。安全側は「残す」**)
  `decided_by`      year / mtime_needed / author_needed / none
  `superseded_by`   その頁を置き換える頁の `page_path`(final / sibling は空)

期待(走らせる前に書く):
  P22 種類A・D の組で、**新しい年の層数 >= 古い年の層数** が >= 0.80
  P23 同じ組で、**描画 `drawn_ratio` の和が 新しい年 >= 古い年** が >= 0.80
  P24 marisa の組で、形式5(カラー)の層数が **新しい年 > 古い年**
  P25 代表の名前が `Done`/`rev`/`b`/`done` 等を持つ割合を印字(選別には使わない)
"""
import collections
import csv
import hashlib
import re
import sys
from pathlib import Path

CSV872 = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_872.csv')
OUT = Path('results/m19_versions'); OUT.mkdir(parents=True, exist_ok=True)
SUFFIX = re.compile(r'(done|rev|fin|comp|new|_b$|b$|\d+b$)', re.I)


def num(s, d=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return d


def parts(p):
    q = p.replace('\\', '/').split('/')
    # F:/manuscripts/<year>/<work>/.../<file>.clip
    year = q[2] if len(q) > 2 else ''
    work = q[3] if len(q) > 3 else ''
    stem = Path(q[-1]).stem
    m = re.match(r'(\d+)(.*)', year)
    return {'year_txt': year, 'year_num': int(m.group(1)) if m else -1,
            'year_suf': m.group(2) if m else '', 'work': work, 'stem': stem}


def main():
    rows = list(csv.DictReader(open(CSV872, encoding='utf-8')))
    pages = list(dict.fromkeys(r['page_path'] for r in rows))
    info = {p: parts(p) for p in pages}
    per = collections.defaultdict(lambda: {'layers': 0, 'drawn': 0,
                                           'dr': 0.0, 'f5': 0})
    for r in rows:
        a = per[r['page_path']]
        a['layers'] += 1
        d = num(r['drawn_ratio'], 0) or 0
        if r['folder'] != 'True' and d > 0:
            a['drawn'] += 1
            a['dr'] += d
            if r['formats'] == '5':
                a['f5'] += 1

    def sig(r):
        return hashlib.md5('|'.join((
            r['name'], r['layer_type'], r['drawn_ratio'], r['sat_mean'],
            r['top_colour_share'], r['colours_q16'])).encode()).hexdigest()[:16]
    inv, pset = collections.defaultdict(set), collections.defaultdict(set)
    for r in rows:
        if r['folder'] == 'True' or (num(r['drawn_ratio'], 0) or 0) <= 0:
            continue
        s = sig(r)
        inv[s].add(r['page_path'])
        pset[r['page_path']].add(s)
    shared = collections.Counter()
    for s, ps in inv.items():
        if 1 < len(ps) <= 12:
            pl = sorted(ps)
            for i in range(len(pl)):
                for j in range(i + 1, len(pl)):
                    shared[(pl[i], pl[j])] += 1
    pairs = []
    for (a, b), c in shared.items():
        u = len(pset[a]) + len(pset[b]) - c
        if u and c / u >= 0.5:
            pairs.append((c / u, c, a, b))
    pairs.sort(reverse=True)

    # 塊(union-find)
    par = {p: p for p in pages}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for _, _, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    cl = collections.defaultdict(list)
    for p in pages:
        cl[find(p)].append(p)
    clusters = [sorted(v) for v in cl.values() if len(v) > 1]
    clusters.sort(key=lambda v: -len(v))

    def kind_of(group):
        works = {(info[p]['year_txt'], info[p]['work']) for p in group}
        if len({info[p]['work'] for p in group}) > 1:
            return 'D'
        if len({info[p]['year_num'] for p in group}) > 1:
            return 'A'
        stems = sorted({info[p]['stem'] for p in group})
        if len(stems) == 1:
            return 'B'
        pref = any(a != b and (a.startswith(b) or b.startswith(a))
                   for a in stems for b in stems)
        return 'B' if pref else 'C'

    out, kinds = {}, collections.Counter()
    need_mtime, need_author, pairs_ad = [], [], []
    for ci, group in enumerate(clusters, start=1):
        k = kind_of(group)
        kinds[k] += 1
        if k == 'C':
            for p in group:
                out[p] = dict(dup_cluster_id=ci, cluster_kind=k,
                              version_role='sibling', decided_by='none',
                              superseded_by='')
            need_author.append((ci, group))
            continue
        ys = sorted({info[p]['year_num'] for p in group})
        if len(ys) > 1:
            best = max(group, key=lambda p: info[p]['year_num'])
            dec = 'year'
            for p in group:
                if p is not best and info[p]['year_num'] == info[best]['year_num']:
                    dec = 'mtime_needed'
            olds = [p for p in group if p != best]
            for o in olds:
                pairs_ad.append((best, o))
        else:
            best, dec = None, 'mtime_needed'
        for p in group:
            # best が決まらない塊(年が同じ)は **undecided**。落とさない。
            out[p] = dict(
                dup_cluster_id=ci, cluster_kind=k,
                version_role=('final' if p == best else 'superseded')
                if best else 'undecided',
                decided_by=dec,
                superseded_by='' if (best is None or p == best) else best)
        if best is None:
            need_mtime.append((ci, group))
    for p in pages:
        out.setdefault(p, dict(dup_cluster_id='', cluster_kind='',
                               version_role='single', decided_by='none',
                               superseded_by=''))

    print(f'== 塊 {len(clusters)} 個 / 関わるページ '
          f'{sum(len(g) for g in clusters)} / {len(pages)} ==')
    print(f'  種類: ' + '  '.join(f'{k} {kinds[k]}' for k in 'ABCD'))
    print(f'  R-1(年)で決まった塊: '
          f'{sum(1 for g in clusters if kind_of(g) != "C" and len({info[p]["year_num"] for p in g}) > 1)}')
    print(f'  **R-2(更新時刻)が要る塊: {len(need_mtime)}**  '
          f'関わるページ {sum(len(g) for _, g in need_mtime)}')
    print(f'  **作者に出す塊(種類C。両方残す): {len(need_author)}**  '
          f'関わるページ {sum(len(g) for _, g in need_author)}')
    role = collections.Counter(v['version_role'] for v in out.values())
    print(f'  役割: {dict(role)}')
    keep = role['single'] + role['final'] + role['sibling'] + role['undecided']
    print(f'  → **腕の主値 = single + final + sibling + undecided = {keep} ページ**'
          f'(**年で置き換えが確定した superseded {role["superseded"]} だけ落とす**)')
    print(f'    undecided {role["undecided"]} ページは更新時刻が届いたら減る'
          f'(**その時点で母集団が変わることを先に書いておく**)')

    print('\n== P22 / P23: 年を跨ぐ組で、新しい年のほうが大きいか ==')
    if pairs_ad:
        ge_l = sum(1 for n, o in pairs_ad if per[n]['layers'] >= per[o]['layers'])
        ge_d = sum(1 for n, o in pairs_ad if per[n]['dr'] >= per[o]['dr'])
        gt_l = sum(1 for n, o in pairs_ad if per[n]['layers'] > per[o]['layers'])
        eq = sum(1 for n, o in pairs_ad
                 if per[n]['layers'] == per[o]['layers']
                 and abs(per[n]['dr'] - per[o]['dr']) < 1e-9)
        print(f'  **うち新旧が完全に同じ大きさの組 {eq} / {len(pairs_ad)}、'
              f'新のほうが厳密に大きい組 {gt_l}** '
              f'(棒は >= なので同じでも通る = 歯が立っていない)')
        print(f'  組 {len(pairs_ad)}  層数 新>=旧 **{ge_l} ({ge_l/len(pairs_ad):.3f})** '
              f'棒 0.80 -> {"PASS" if ge_l/len(pairs_ad) >= 0.8 else "**FAIL**"}')
        print(f'              描画の和 新>=旧 **{ge_d} ({ge_d/len(pairs_ad):.3f})** '
              f'棒 0.80 -> {"PASS" if ge_d/len(pairs_ad) >= 0.8 else "**FAIL**"}')
        for n, o in pairs_ad[:10]:
            print(f'    新 {info[n]["year_txt"]:>6}/{info[n]["work"]:<12} '
                  f'{Path(n.replace(chr(92),"/")).name:<22} 層{per[n]["layers"]:>4} '
                  f'dr{per[n]["dr"]:7.2f}   旧 {info[o]["year_txt"]:>6}/'
                  f'{info[o]["work"]:<12} {Path(o.replace(chr(92),"/")).name:<22} '
                  f'層{per[o]["layers"]:>4} dr{per[o]["dr"]:7.2f}')
    print('\n== P24: marisa(作者の 3。追加の彩色) ==')
    mar = [(n, o) for n, o in pairs_ad if 'marisa' in info[n]['work'].lower()
           or 'marisa' in info[o]['work'].lower()]
    for n, o in mar:
        print(f'  新 {info[n]["year_txt"]} {Path(n.replace(chr(92),"/")).name} '
              f'形式5 {per[n]["f5"]}  <-  旧 {info[o]["year_txt"]} '
              f'{Path(o.replace(chr(92),"/")).name} 形式5 {per[o]["f5"]}  '
              f'-> {"PASS" if per[n]["f5"] > per[o]["f5"] else "**FAIL**"}')
    if not mar:
        print('  年を跨ぐ marisa の組は R-1 の対象に無い(同じ年か種類C)')

    print('\n== P24-b marisa_color(作者の「追加依頼で彩色」はこちらではないか) ==')
    for p in pages:
        if 'marisa' in info[p]['work'].lower():
            print(f'  {info[p]["year_txt"]:>6}/{info[p]["work"]:<14} '
                  f'{Path(p.replace(chr(92),"/")).name:<26} '
                  f'層{per[p]["layers"]:>4} 形式5 {per[p]["f5"]:>4} '
                  f'描画の和 {per[p]["dr"]:7.2f}')

    print('\n== P25 名前の接尾辞(選別には使わない。falsifier) ==')
    fin = [p for p, v in out.items() if v['version_role'] == 'final']
    sup = [p for p, v in out.items() if v['version_role'] == 'superseded']
    if fin:
        print(f'  代表(final {len(fin)})のうち接尾辞あり '
              f'{sum(1 for p in fin if SUFFIX.search(info[p]["stem"]))} '
              f'/ 置き換えられた側({len(sup)})のうち '
              f'{sum(1 for p in sup if SUFFIX.search(info[p]["stem"]))}')

    print('\n== 作者に出す候補(種類C。両方残すので判定は急がない) ==')
    for ci, g in need_author[:12]:
        print(f'  塊{ci:>3} {info[g[0]]["year_txt"]}/{info[g[0]]["work"]}: '
              f'{[Path(p.replace(chr(92),"/")).name for p in g]}')
    print(f'\n== R-2(更新時刻)が要る塊 {len(need_mtime)} ==')
    for ci, g in need_mtime[:12]:
        print(f'  塊{ci:>3} {info[g[0]]["year_txt"]}/{info[g[0]]["work"]}: '
              f'{[Path(p.replace(chr(92),"/")).name for p in g]}')

    cols = ['page_path', 'dup_cluster_id', 'cluster_kind', 'version_role',
            'decided_by', 'superseded_by']
    with open(OUT / 'version_roles.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for p in pages:
            w.writerow({'page_path': p, **out[p]})
    print(f'\nwrote {OUT/"version_roles.csv"}  '
          f'({len(pages)} 行。抽出側の DB に戻す列)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
