#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追補71 §F の項目4′ —— A〜H の向きの問いを、塊を単位にして数える。**

追補71 §D は「問い」を単位にした(**登録した棒。1つも消えなかった**)。
**A はフォルダー対1件だが塊は9つ**なので、単位が間違っていた(§E)。
ここは**塊**を単位にする。**主値 = 構成する `page_path` が全員バイト同一の塊の数**。
**帰無値 0**。

  <venv>/python tools/colour/m66_ah_by_cluster.py
"""
import collections
import csv
import json
import pathlib
import sys

SRC = pathlib.Path('../lineart/inbox/sha256_dup_clusters_20261026.csv')
OUT = pathlib.Path('results/m65_sha_recluster/ah_by_cluster.csv')
# **道の一覧と塊数は追跡外**(`check_public_names.py`・`m65` と同じ形)。
# 塊数は `notice_version_identity_ask_the_author_20261026.md` §4 の値
PATHS = pathlib.Path('results/alias/ah_question_paths.json')


def load_paths():
    if not PATHS.exists():
        raise SystemExit(f'**道の一覧が無い: {PATHS}。走らせない**')
    return json.loads(PATHS.read_text(encoding='utf-8'))


def make_norm(prefix):
    return lambda p: p[len(prefix):] if p.startswith(prefix) else p


def main():
    cfg = load_paths()
    norm = make_norm(cfg['root_prefix'])
    AH = cfg['ah']
    NOTICE_N = cfg['ah_notice_n']
    rows = list(csv.DictReader(SRC.open(encoding='utf-8')))
    by_cl = collections.defaultdict(list)
    for r in rows:
        by_cl[r['dup_cluster_id']].append(r)

    out, per_q = [], {}
    for q, pats in sorted(AH.items()):
        cls = sorted({r['dup_cluster_id'] for r in rows
                      if any(p in norm(r['page_path']) for p in pats)})
        c = collections.Counter()
        for cid in cls:
            mem = by_cl[cid]
            shas = {r['sha256'] for r in mem}
            if len(mem) < 2:
                v = '決められない(2頁未満)'
            elif len(shas) == 1:
                v = '**消える**'
            else:
                v = '消えない'
            c[v] += 1
            out.append({'q': q, 'dup_cluster_id': cid, 'n_pages': len(mem),
                        'n_sha': len(shas), 'kind': mem[0]['cluster_kind'],
                        'verdict': v,
                        'paths': ' ;; '.join(norm(r['page_path']) for r in mem)})
        per_q[q] = (len(cls), c)

    print('== 項目4′: A〜H の向きの問い、**塊を単位に**(追補71 §F)==')
    print(f'  {"問い":5s} {"塊":>4} {"通達":>5} {"消える":>7} {"消えない":>9} {"決めら":>7}')
    tot = collections.Counter()
    for q, (n, c) in per_q.items():
        nn = NOTICE_N[q]
        print(f'  {q:5s} {n:>4} {str(nn) if nn else "2+":>5} '
              f'{c["**消える**"]:>7} {c["消えない"]:>9} '
              f'{c["決められない(2頁未満)"]:>7}')
        tot.update(c)
    s = sum(tot.values())
    print(f'  **主値(消える塊) {tot["**消える**"]}** / 消えない {tot["消えない"]}'
          f' / 決められない {tot["決められない(2頁未満)"]}  合計 {s}'
          f'  **残り {s - sum(tot.values())}**')
    print('  **文字をまたいで合計しない**(上の合計は残り 0 の点検のためだけ)')
    print('  **消えるのは向きの問いだけ。工程の問いは残る**(追補70 §G)')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f'\nwrote {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
