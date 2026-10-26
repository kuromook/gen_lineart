#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**作者の答のファイルから、原稿の固有名を別名に置き換える**(作者の指示、2026-10-26)。

`origin` は匿名で読める。**作者は「今回は危険な情報は無い」と判断したが、今後は増やさない。**
**選ばれた2つ**: **(1) push の前に点検を回す**(`check_public_names.py`)/
**(2) 答えのファイルを別名にする**(これ)。

| 置き換える欄 | 前 | 後 |
|---|---|---|
| `pid` | `F:\\manuscripts\\2018\\skima11\\skima12H.clip` | `P0412` |
| `work` | `000_2018_skima11_skima12H  層 140「…」 1/3` | `P0412  層 140  1/3`(**層名も外す**) |

**別名は台帳の `page_path` の辞書順から決める**(決定論。**台帳が在れば誰でも作り直せる**)。
**対応表は `results/alias/page_alias.csv`** —— **`results/**` は追跡外**なので公開側には出ない。
**手元では今までどおり辿れる。**

**冪等**: 既に別名になっている行は触らない。**元のファイルは `.orig` に残す**(消さない)。

  ./venv/bin/python tools/alias_labels.py --dry    # 置き換えずに数える
  ./venv/bin/python tools/alias_labels.py
"""
import argparse
import csv
import json
import pathlib
import re
import sys

LEDGER = pathlib.Path('results/m64_ledger892/m_colour_page_layers_892.csv')
MAP = pathlib.Path('results/alias/page_alias.csv')
ALIAS = re.compile(r'^P\d{4}$')
LAYERNAME = re.compile(r'「[^」]*」')


def build_map():
    if not LEDGER.exists():
        raise SystemExit(f'**台帳が無い: {LEDGER}。別名を決められない**')
    paths = sorted({r['page_path'] for r in
                    csv.DictReader(open(LEDGER, encoding='utf-8'))})
    m = {p: f'P{i:04d}' for i, p in enumerate(paths, 1)}
    MAP.parent.mkdir(parents=True, exist_ok=True)
    with open(MAP, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['alias', 'page_path'])
        for p, a in sorted(m.items(), key=lambda t: t[1]):
            w.writerow([a, p])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    m = build_map()
    print(f'別名の対応表: {MAP}  ({len(m)} 頁。**追跡外**)')
    tot = {'行': 0, 'pid を置換': 0, 'work を置換': 0,
           '台帳に無い pid': 0, '既に別名': 0}
    unknown = set()
    for f in sorted(pathlib.Path('labels').glob('*.jsonl')):
        out, n = [], 0
        for ln in open(f, encoding='utf-8'):
            if not ln.strip():
                continue
            r = json.loads(ln); tot['行'] += 1; n += 1
            pid = r.get('pid')
            if isinstance(pid, str) and ALIAS.match(pid):
                tot['既に別名'] += 1
            elif isinstance(pid, str):
                al = m.get(pid)
                if al is None:
                    tot['台帳に無い pid'] += 1; unknown.add(pid)
                else:
                    r['pid'] = al; tot['pid を置換'] += 1
                    w = r.get('work')
                    if isinstance(w, str):
                        # 頁の名札を別名に、層名「…」を外す
                        w2 = LAYERNAME.sub('', w)
                        w2 = re.sub(r'^\S+', al, w2.strip())
                        r['work'] = re.sub(r'\s+', ' ', w2).strip()
                        tot['work を置換'] += 1
            out.append(r)
        print(f'  {f}  {n} 行')
        if not a.dry:
            f.with_suffix('.jsonl.orig').write_text(
                f.read_text(encoding='utf-8'), encoding='utf-8')
            with open(f, 'w', encoding='utf-8') as fh:
                for r in out:
                    fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print()
    for k, v in tot.items():
        print(f'  {k:18s} {v:>5}')
    if unknown:
        print(f'\n  **台帳に無い pid の異なり {len(unknown)}**(置き換えていない):')
        for p in sorted(unknown)[:6]:
            print(f'    {p}')
    if a.dry:
        print('\n(--dry: 書いていない)')
    else:
        print('\n**元のファイルは `.jsonl.orig` に残した(追跡外にするのは別の手)**')
    return 0


if __name__ == '__main__':
    sys.exit(main())
