#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**追跡ファイルに原稿の固有名が入っていないかを数える**(push の前に回す)。

2026-10-26: `origin`(`github.com/kuromook/gen_lineart`)が**匿名で読める**ことが分かった
(`git ls-remote https://...` が認証なしで通る)。**作者は「今回は危険な情報はない」と判断**したが、
**今後のために増やさない**ための点検。

**この点検は「名前が在るか」を数えるだけで、消さない。**
**消す判断は作者のもの**(force-push では履歴から消えない、という実例をこのプロジェクトは持っている)。

**規律4 の形**: **5つの規律のうち唯一、誰の注意力にも依存しない点検**にしたいので、
**終了コードで落とす**。`--baseline` で「いまの数」を記録し、**それより増えたら 1 を返す**。

  ./venv/bin/python tools/check_public_names.py            # 数えて、基準より増えていたら 1
  ./venv/bin/python tools/check_public_names.py --baseline # いまの数を基準として書く
"""
import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys

BASE = pathlib.Path('doc/public_names_baseline.json')
PATH = re.compile(r'[A-Z]:\\\\?manuscripts\\\\?[^\s`|)]*')
# 一覧は追跡外(上の NAMES)。**増えたらそちらに足す**
# **作品名の一覧は追跡外に置く。**
# **探す名前を列挙する点検は、その名前を公開する** ——
# 2026-10-26、この点検の初回の発火が、この点検自身だった。
NAMES = pathlib.Path('results/alias/work_name_patterns.txt')


def works_re():
    if not NAMES.exists():
        raise SystemExit(f'**作品名の一覧が無い: {NAMES}。点検できない**')
    pats = [x.strip() for x in NAMES.read_text(encoding='utf-8').splitlines()
            if x.strip()]
    return re.compile(r'\b(' + '|'.join(pats) + r')\b')


def scan():
    wre = works_re()
    files = subprocess.run(['git', 'ls-files'], capture_output=True,
                           text=True).stdout.split()
    hits, works, paths = collections.Counter(), collections.Counter(), set()
    for f in files:
        p = pathlib.Path(f)
        if p.suffix.lower() not in ('.md', '.py', '.csv', '.txt', '.json',
                                    '.jsonl', '.js', '.sh'):
            continue
        try:
            s = p.read_text(encoding='utf-8', errors='replace')
        except Exception:                                      # noqa: BLE001
            continue
        # **伏字(`<年>` の形)は名前ではない。数えない**
        pa = [x for x in PATH.findall(s) if '<' not in x]
        wo = wre.findall(s)
        if pa or wo:
            hits[f] = len(pa) + len(wo)
        works.update(wo); paths.update(pa)
    return hits, works, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', action='store_true')
    a = ap.parse_args()
    hits, works, paths = scan()
    total = sum(hits.values())
    print(f'追跡ファイルのうち固有名を含むもの: **{len(hits)}** / 出現 **{total}**')
    print(f'作品名の異なり **{len(works)}** / フルパスの異なり **{len(paths)}**')
    for f, n in hits.most_common(8):
        print(f'  {f:56s} {n:>5}')
    if len(hits) > 8:
        print(f'  (ほか {len(hits) - 8} ファイル)')

    if a.baseline:
        BASE.write_text(json.dumps(
            {'files': len(hits), 'total': total, 'works': len(works),
             'paths': len(paths),
             'note': '2026-10-26 時点。**減らす方向は歓迎、増やす方向は要確認**'},
            ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'\n基準を書いた: {BASE}')
        return 0
    if not BASE.exists():
        print('\n**基準が無い。`--baseline` で先に書くこと**')
        return 1
    b = json.loads(BASE.read_text(encoding='utf-8'))
    bad = []
    for k, v in (('files', len(hits)), ('total', total),
                 ('works', len(works)), ('paths', len(paths))):
        mark = '増えた' if v > b[k] else ('減った' if v < b[k] else '同じ')
        print(f'  {k:8s} 基準 {b[k]:>6} / いま {v:>6}  **{mark}**')
        if v > b[k]:
            bad.append(k)
    if bad:
        print(f'\n**増えている: {", ".join(bad)}。push の前に確認すること**')
        print('**消す判断は作者のもの。** この点検は数えるだけ')
        return 1
    print('\nPASS(基準より増えていない)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
