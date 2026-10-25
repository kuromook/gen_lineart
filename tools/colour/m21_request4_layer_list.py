"""M21: 依頼4 の層一覧を作る。**作者が 2026-10-25 に逐語で発注を承認した**
(work_log 20:05 の節。問い「10,622層・約1.7GB・786ページ…を発注してよいか」→「ok」)。

一覧に入れるもの(追補10 §I4 / 追補13 §L8 で登録した形):
  (1) **描画層のクリップのベース**(非フォルダー・描画画素>0・`rendered=True`)
  (2) **フォルダーがベースのときは、そのフォルダーの子孫の描画層すべて**
      (フォルダー自身にラスタは無く、面はその子孫のアルファの和である)

母集団(追補15 §N3 で登録):
  **786ページ** = 872 − `superseded` 74 − `duplicate_content` 12。
  **`undecided` 24 も入れる** —— 更新時刻が来るまでどちらが残るか決まらないので、
  **両方取る**(落とす側の費用は最大12ページぶん = 全体の約1.5%)。
  **`sibling` は版ではないので両方入れる。**

既知セル: **先頭8ページ(パイロットと同じページ)を一覧の先頭に置く。**
納品が来たら、**パイロットの 167層のアルファとバイト一致するか**を突き合わせる。
(依頼3 でも同じ形を使い、1,005行×64列が一致した。)

**この器具は腕を採点しない。**
"""
import collections
import csv
import sys
from pathlib import Path

CSV872 = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_872.csv')
ROLES = Path('results/m19_versions/version_roles.csv')
OUT = Path('results/m21_request4'); OUT.mkdir(parents=True, exist_ok=True)
PAPER = '1584'
REG8 = 8
DROP = {'superseded', 'duplicate_content'}


def num(s, d=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return d


def main():
    rows = list(csv.DictReader(open(CSV872, encoding='utf-8')))
    roles = {r['page_path']: r['version_role']
             for r in csv.DictReader(open(ROLES, encoding='utf-8'))}
    pages = list(dict.fromkeys(r['page_path'] for r in rows))
    reg = pages[:REG8]
    keep = [p for p in pages if roles[p] not in DROP]
    order = reg + [p for p in keep if p not in set(reg)]
    print(f'母集団 {len(keep)} ページ(872 − ' +
          ' − '.join(f'{k} {sum(1 for p in pages if roles[p] == k)}'
                     for k in sorted(DROP)) + ')')
    print(f'  役割の内訳: ' + '  '.join(
        f'{k} {v}' for k, v in sorted(collections.Counter(
            roles[p] for p in keep).items())))

    bypage = collections.defaultdict(list)
    for r in rows:
        bypage[r['page_path']].append(r)
    kids = collections.defaultdict(list)
    for r in rows:
        if r['parent_id'] != '':
            kids[(r['page_path'], r['parent_id'])].append(r)

    def desc(p, lid):
        out = []
        for c in kids.get((p, lid), []):
            out.append(c)
            out += desc(p, c['layer_id'])
        return out

    out_rows, stat = [], collections.Counter()
    for p in order:
        v = bypage[p]
        sib = collections.defaultdict(list)
        for r in v:
            if r['layer_type'] == PAPER or r['parent_id'] == '':
                continue
            sib[r['parent_id']].append(r)
        bases = []
        for g in sib.values():
            g.sort(key=lambda r: int(r['order_in_parent']))
            base = None
            for r in g:
                if r['clip'] == '0':
                    base = r
                elif base is not None and base not in bases:
                    bases.append(base)
        for b in bases:
            if b['folder'] == 'True':
                stat['folder_base'] += 1
                for c in desc(p, b['layer_id']):
                    if c['folder'] == 'True' or c['rendered'] != 'True':
                        continue
                    if (num(c['drawn_ratio'], 0) or 0) <= 0:
                        continue
                    out_rows.append((p, c['layer_id'], 'folder_base_child',
                                     b['layer_id'], c['name']))
                    stat['child'] += 1
            else:
                if b['rendered'] != 'True' or (num(b['drawn_ratio'], 0) or 0) <= 0:
                    stat['base_no_raster'] += 1
                    continue
                out_rows.append((p, b['layer_id'], 'base', '', b['name']))
                stat['base'] += 1
    seen = set()
    uniq = []
    for r in out_rows:
        if (r[0], r[1]) in seen:
            continue
        seen.add((r[0], r[1]))
        uniq.append(r)
    print(f'\n一覧: **{len(uniq)} 層** / {len({r[0] for r in uniq})} ページ'
          f'  (ベース {stat["base"]} + フォルダーのベースの子孫 {stat["child"]}'
          f'、重複除去で {len(out_rows) - len(uniq)} 行減)')
    print(f'  ラスタを持たないベース(一覧に入れられない) {stat["base_no_raster"]}')
    print(f'  ベースが1枚も無いページ(アルファ不要・プレビューのみ): '
          f'{len(keep) - len({r[0] for r in uniq})}')

    with open(OUT / 'request4_layer_list.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['順', 'page_path', 'layer_id', '役割', '親フォルダーの layer_id',
                    'name', '既知セル'])
        for i, (p, lid, role, par, nm) in enumerate(uniq, start=1):
            w.writerow([i, p, lid, role, par, nm, 'yes' if p in set(reg) else ''])
    with open(OUT / 'request4_page_list.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['順', 'page_path', 'version_role', 'アルファ要求層数', '既知セル'])
        cnt = collections.Counter(r[0] for r in uniq)
        for i, p in enumerate(order, start=1):
            if p not in set(keep):
                continue
            w.writerow([i, p, roles[p], cnt[p], 'yes' if p in set(reg) else ''])
    print(f'\nwrote {OUT/"request4_layer_list.csv"} / '
          f'{OUT/"request4_page_list.csv"}')
    print(f'先頭8ページ(既知セル)の層数: '
          f'{sum(1 for r in uniq if r[0] in set(reg))}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
