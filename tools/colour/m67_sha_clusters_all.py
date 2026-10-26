#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追補70 §E の項目1 —— `.clip` 全数の sha256 から塊を組み直す。**

**記述であって判定ではない**(棒は置かない。追補70 §E に先に書いた)。
あわせて項目2・3 を全数の上で読み直す(こちらの116塊が、全数の塊のどこに入るか)。

**母集団は `format = 'clip'` だけ。`.lip` は混ぜない**(追補60 §E。別の母集団)。

  <venv>/python tools/colour/m67_sha_clusters_all.py
"""
import collections
import csv
import json
import pathlib
import sys

ALL = pathlib.Path('../lineart/inbox/request9_file_sha256_all_20261026.csv')
MINE = pathlib.Path('../lineart/inbox/sha256_dup_clusters_20261026.csv')
OUT = pathlib.Path('results/m67_sha_all')
PATHS = pathlib.Path('results/alias/ah_question_paths.json')   # **追跡外**


def load_paths():
    if not PATHS.exists():
        raise SystemExit(f'**道の一覧が無い: {PATHS}。走らせない**')
    return json.loads(PATHS.read_text(encoding='utf-8'))


def main():
    cfg = load_paths()
    pre = cfg['root_prefix']
    norm = (lambda p: p[len(pre):] if p.startswith(pre) else p)

    rows = [r for r in csv.DictReader(ALL.open(encoding='utf-8'))
            if r['format'] == 'clip']
    err = [r for r in rows if r['f_read_error'].strip()]
    print(f'母集団 `.clip` {len(rows)} 行  読めなかった {len(err)}')
    assert not err, '読めない行が在る。数える前に止める'

    by_sha = collections.defaultdict(list)
    for r in rows:
        by_sha[r['f_sha256']].append(r)
    groups = [g for g in by_sha.values() if len(g) > 1]
    files_in = sum(len(g) for g in groups)
    surplus = files_in - len(groups)
    print(f'sha256 の異なり {len(by_sha)}')

    # ---- 既知セル(追補71 §A。**在ることで通るセル**) ----
    want = [norm(x) for x in cfg['known_cell_is_a_cluster']]
    got = [r for r in rows if norm(r['page_path']) in want]
    shas = {r['f_sha256'] for r in got}
    ok = len(got) == len(want) and len(shas) == 1
    print('\n== 既知セル(塊5 の3頁。追補71 §A)==')
    print(f'  全数の表に在る {len(got)} / 期待 {len(want)}  sha256 の異なり {len(shas)}')
    if not ok:
        print('  **FAIL。結果は読まない**')
        return 1
    g = by_sha[shas.pop()]
    print(f'  **PASS**(1つの塊になった。その塊の大きさ {len(g)} 頁)')

    # ---- 項目1: 記述(棒は置かない) ----
    print('\n== 項目1: `.clip` 全数の塊(**記述。判定ではない**)==')
    print(f'  **バイト同一の塊 {len(groups)}**')
    print(f'  **その塊に属するファイル {files_in}**'
          f'  (= 全体の {files_in / len(rows) * 100:.1f}%)')
    print(f'  **余り(ファイル − 塊) {surplus}**'
          f'   ← 台帳の「複製 1,473」はこの数')
    sz = collections.Counter(len(g) for g in groups)
    print(f'  2頁の塊 {sz[2]}  /  3頁以上の塊 {sum(v for k, v in sz.items() if k >= 3)}')
    print('  大きさの分布(行ごとに印字する):')
    for k in sorted(sz):
        print(f'    {k:>3}頁の塊 {sz[k]:>5}   ファイル {k * sz[k]:>5}')

    # ---- 項目2: こちらの116塊が全数の塊のどこに入るか ----
    mine = list(csv.DictReader(MINE.open(encoding='utf-8')))
    by_cl = collections.defaultdict(list)
    for r in mine:
        by_cl[r['dup_cluster_id']].append(r)
    sha_of = {norm(r['page_path']): r['f_sha256'] for r in rows}
    missing = [r for r in mine if norm(r['page_path']) not in sha_of]
    print(f'\n== 項目2: こちらの116塊(248頁)を全数の塊に重ねる ==')
    print(f'  全数の表で引けなかった頁 {len(missing)}')
    cat, detail = collections.Counter(), []
    for cid, mem in by_cl.items():
        s = [sha_of.get(norm(r['page_path'])) for r in mem]
        gs = {x: len(by_sha[x]) for x in set(s) if x}
        k = len(set(s))
        if k == 1:
            big = max(gs.values())
            c = ('I=1 全員同一で、全数でも同じ大きさ' if big == len(mem)
                 else 'I+ 全員同一だが、全数の塊はもっと大きい')
        elif any(v > 1 for v in gs.values()):
            c = 'P 一部が全数のどれかの塊に入る'
        else:
            c = 'S 全員がバイト別(全数でも塊を作らない)'
        cat[c] += 1
        detail.append({'dup_cluster_id': cid, 'n_pages': len(mem),
                       'kind': mem[0]['cluster_kind'], 'cls': c,
                       'n_sha': k, 'max_group_all': max(gs.values()) if gs else 0,
                       'roles': '|'.join(sorted({r['version_role'] for r in mem}))})
    tot = 0
    for c, n in sorted(cat.items()):
        print(f'  {c:36s} 塊 {n:>4}'
              f'  頁 {sum(d["n_pages"] for d in detail if d["cls"] == c):>4}')
        tot += n
    print(f'  {"合計":36s} 塊 {tot:>4}   **残り {len(by_cl) - tot}**')

    # ---- 項目3: 署名でしか捕まらない塊(全数の上で) ----
    only = [d for d in detail if d['cls'].startswith('S')]
    cover = sum(1 for d in detail if d['cls'].startswith('I'))
    print('\n== 項目3: 全数の上で読み直した主値 ==')
    print(f'  **署名でしか捕まらない塊 {len(only)} / 頁'
          f' {sum(d["n_pages"] for d in only)}**')
    print(f'  こちらの116塊が覆う、全数の塊の数 {cover} / {len(groups)}'
          f'  = **{cover / len(groups) * 100:.1f}%**')
    for key in ('kind', 'roles'):
        cc = collections.Counter(d[key] for d in only)
        print(f'  {key}: ' + ' / '.join(f'{k}={v}' for k, v in sorted(cc.items())))

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'mine_vs_all.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader(); w.writerows(sorted(detail, key=lambda d: d['cls']))
    with (OUT / 'all_clusters.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['sha256', 'n_pages', 'page_paths'])
        for sha, g in sorted(by_sha.items(), key=lambda kv: -len(kv[1])):
            if len(g) > 1:
                w.writerow([sha, len(g),
                            ' ;; '.join(norm(r['page_path']) for r in g)])
    print(f'\nwrote {OUT}/all_clusters.csv, {OUT}/mine_vs_all.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
