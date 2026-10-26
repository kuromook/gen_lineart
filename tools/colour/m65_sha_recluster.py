#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追補70 の事前登録を走らせる(項目2・3・4)。**

主値 = **署名(層のメタデータ指紋)の塊のうち、sha256 では1つの塊にならないものの数**。
**帰無値 0**(全部バイト同一なら、署名に固有の取り分は無い)。

項目1(1,473 の複製が何塊になるか)は**全7,667行の sha256 が要る**ので、ここには無い。

  <venv>/python tools/colour/m65_sha_recluster.py
"""
import collections
import csv
import json
import pathlib
import sys

SRC = pathlib.Path('../lineart/inbox/sha256_dup_clusters_20261026.csv')
OUT = pathlib.Path('results/m65_sha_recluster')
# **道の一覧は追跡外**(`check_public_names.py` と同じ形)。
# **探す名前を列挙する器具は、その名前を公開する** —— 追補68 の罠2つめ。
# 無ければ**走らない**(名前を既定値で埋めない)
PATHS = pathlib.Path('results/alias/ah_question_paths.json')


def load_paths():
    if not PATHS.exists():
        raise SystemExit(f'**道の一覧が無い: {PATHS}。走らせない**')
    return json.loads(PATHS.read_text(encoding='utf-8'))


def make_norm(prefix):
    """**根の道も追跡外から取る**(ここに書くと公開される)。"""
    return lambda p: p[len(prefix):] if p.startswith(prefix) else p


def main():
    if not SRC.exists():
        raise SystemExit(f'**材料が無い: {SRC}**')
    cfg = load_paths()
    norm = make_norm(cfg['root_prefix'])
    KNOWN = cfg['known_cell_not_a_cluster']
    AH = cfg['ah']
    rows = list(csv.DictReader(SRC.open(encoding='utf-8')))
    print(f'材料 {SRC}  行 {len(rows)}')

    by_cl = collections.defaultdict(list)
    for r in rows:
        by_cl[r['dup_cluster_id']].append(r)
    print(f'署名の塊 {len(by_cl)} / 頁 {len(rows)}')

    # ---- 既知セル(§F)。**再現しなければ他を読まない** ----
    a = [r for r in rows if norm(r['page_path']) == KNOWN[0]]
    b = [r for r in rows if norm(r['page_path']) == KNOWN[1]]
    same_cl = bool(a and b and a[0]['dup_cluster_id'] == b[0]['dup_cluster_id'])
    same_sha = bool(a and b and a[0]['sha256'] == b[0]['sha256'])
    print('\n== 既知セル(追補70 §F)==')
    print(f'  {KNOWN[0]}  表に在る: {bool(a)}')
    print(f'  {KNOWN[1]}  表に在る: {bool(b)}')
    print(f'  同じ署名の塊: {same_cl}  同じ sha256: {same_sha}')
    if same_cl or same_sha:
        print('  **FAIL。実装を疑う。結果は読まない**')
        return 1
    print('  **PASS**(どちらの腕でも同じ塊にならない。'
          '表に在らない場合も「塊でない」と一致する)')

    # ---- 項目2: 116塊が sha256 の組み直しにどう入るか ----
    # 分類は塊の中の sha256 の分かれ方で決まる。**残り 0 にする**(§E)
    cat = collections.Counter()
    detail = []
    for cid, mem in by_cl.items():
        shas = collections.Counter(r['sha256'] for r in mem)
        n, k = len(mem), len(shas)
        biggest = max(shas.values())
        if k == 1:
            c = 'I 全員がバイト同一'
        elif biggest == 1:
            c = 'S 全員がバイト別'
        else:
            c = 'P 一部がバイト同一'
        cat[c] += 1
        detail.append({
            'dup_cluster_id': cid, 'n_pages': n, 'n_sha': k,
            'biggest_sha_group': biggest, 'kind': mem[0]['cluster_kind'],
            'cls': c,
            'cross_work': len({norm(r['page_path']).split('\\')[1]
                               for r in mem if '\\' in norm(r['page_path'])}) > 1,
            'roles': '|'.join(sorted({r['version_role'] for r in mem})),
            'paths': ' ;; '.join(norm(r['page_path']) for r in mem),
        })
    print('\n== 項目2: 署名の116塊を sha256 で割る(理由別。残り 0)==')
    tot = 0
    for c, n in sorted(cat.items()):
        pg = sum(d['n_pages'] for d in detail if d['cls'] == c)
        print(f'  {c:18s} 塊 {n:>4}  頁 {pg:>4}')
        tot += n
    print(f'  {"合計":18s} 塊 {tot:>4}  頁 {sum(d["n_pages"] for d in detail):>4}'
          f'   **残り {len(by_cl) - tot}**')

    # 実装の正気の点検のみ(§G。独立な確認ではない)
    bad = sum(1 for r in rows
              if int(r['n_same_sha_in_cluster'] or 0) !=
              sum(1 for o in by_cl[r['dup_cluster_id']]
                  if o is not r and o['sha256'] == r['sha256']))
    print(f'  (正気の点検: `n_same_sha_in_cluster` と合わない行 {bad}'
          f' —— **独立な確認ではない**)')

    # ---- 項目3: 署名でしか捕まらない塊 = 主値 ----
    only = [d for d in detail if d['cls'] == 'S 全員がバイト別']
    print('\n== 項目3(主値): 署名でしか捕まらない塊 ==')
    print(f'  **塊 {len(only)} / 頁 {sum(d["n_pages"] for d in only)}**'
          f'  (帰無値 0)')
    for key, lab in (('kind', 'cluster_kind'), ('roles', 'version_role')):
        cc = collections.Counter(d[key] for d in only)
        print(f'  {lab}: ' + ' / '.join(f'{k}={v}' for k, v in sorted(cc.items())))
    cw = collections.Counter(d['cross_work'] for d in only)
    print(f'  作品を跨ぐ: {cw[True]} / 跨がない: {cw[False]}')

    # 層別は行ごとに印字する(§D。プールを跨いで平均しない)
    print('\n== 層別(行ごと。平均しない)==')
    print(f'  {"層":34s} {"塊":>5} {"I":>5} {"P":>5} {"S":>5}')
    strata = [('cluster_kind=' + k,
               [d for d in detail if d['kind'] == k])
              for k in sorted({d['kind'] for d in detail})]
    strata += [('作品を跨ぐ', [d for d in detail if d['cross_work']]),
               ('作品を跨がない', [d for d in detail if not d['cross_work']])]
    for lab, ds in strata:
        c = collections.Counter(d['cls'][0] for d in ds)
        print(f'  {lab:34s} {len(ds):>5} {c["I"]:>5} {c["P"]:>5} {c["S"]:>5}')

    # ---- 項目4: A〜H のうち向きの問いが消えるもの ----
    print('\n== 項目4: A〜H の向きの問い ==')
    print('  **消えるのは「構成ファイルが互いにバイト同一」の時だけ**。'
          '工程の問いは消えない(§G)')
    ah_rows = []
    for q, pats in sorted(AH.items()):
        hit = [r for r in rows
               if any(p.rstrip('\\') in norm(r['page_path']) for p in pats)]
        shas = collections.Counter(r['sha256'] for r in hit)
        if not hit:
            verdict = '表に無い(248頁の外)'
        elif len(shas) == 1 and len(hit) > 1:
            verdict = '**向きの問いは消える**'
        else:
            verdict = '消えない'
        print(f'  {q}  頁 {len(hit):>3}  sha256 の異なり {len(shas):>3}  {verdict}')
        ah_rows.append({'q': q, 'n_pages': len(hit), 'n_sha': len(shas),
                        'verdict': verdict})

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'cluster_vs_sha256.csv').open('w', newline='',
                                              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader()
        w.writerows(sorted(detail, key=lambda d: (d['cls'], -d['n_pages'])))
    with (OUT / 'ah_questions.csv').open('w', newline='',
                                         encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(ah_rows[0]))
        w.writeheader()
        w.writerows(ah_rows)
    print(f'\nwrote {OUT}/cluster_vs_sha256.csv, {OUT}/ah_questions.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
