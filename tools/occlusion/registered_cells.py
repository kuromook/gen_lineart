#!/usr/bin/env python3
"""登録した既知セル・帯の数を、構造の表から1つの導出で印字する。

基盤が 2026-10-27 に登録した数は、どれも使い捨ての heredoc の出力だった。
述語を1つ直したとき(祖先のフォルダーの表示を見る)、4つの文書を別々に直す
ことになり、誤りは3つの別のセッションが見つけた。再計算される器具が無いと、
定義を直しても数は直らない。この器具がその1つの導出である。

  ./venv/bin/python tools/occlusion/registered_cells.py <構造の表.csv>

構造の表は inbox/ の納品物(.gitignore 配下)なので、引数で渡す。
"""
import csv, sys, collections

FOLDER_TRUE = ('1', 'True', 'true', 't')


def load(path):
    pages = collections.defaultdict(list)
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh, quotechar='"', doublequote=True):
            pages[row['page_path']].append(row)
    return pages


def fmt_of(page_path):
    return 'lip' if page_path.lower().endswith('.lip') else 'clip'


def is_folder(row):
    return row['is_folder'] in FOLDER_TRUE


def self_on(row):
    """作者の定義: 非表示を外し、不透明度 0 を外す。表示は visibility の1の位。

    2の位は「マスクが効いている」で表示ではない(T142)。
    """
    return int(row['visibility'] or 0) % 2 == 1 and int(row['opacity'] or 0) > 0


def ancestor_hidden(row, by_id):
    """祖先のフォルダーが非表示なら、その層は紙面に出ない(T143)。

    層自身の visibility だけを見る1列の走査が、点検・正本・通達・他 track の
    標準の4つに落ちていた。述語はここにしか無い。
    """
    parent = row['parent_id']
    seen = 0
    while parent and parent in by_id and seen < 200:
        anc = by_id[parent]
        seen += 1
        if is_folder(anc) and not self_on(anc):
            return True
        parent = anc['parent_id']
    return False


def occluder_sets(composite):
    """遮蔽物の3組。O_wide が上限になるのは 2 = 乗算 のときだけなので O_all も出す。"""
    return {
        'strict': composite == '0',          # 合成が通常のみ
        'wide': composite != '2',            # 乗算(値2)以外
        'all': True,                         # 合成を見ない。対応表に依存しない端
    }


def main(path):
    pages = load(path)
    c = collections.Counter()
    base = collections.Counter()
    folders = collections.defaultdict(collections.Counter)

    for page_path, rows in pages.items():
        fmt = fmt_of(page_path)
        by_id = {r['layer_id']: r for r in rows}
        hidden_anc = {r['layer_id']: ancestor_hidden(r, by_id) for r in rows}
        c[(fmt, 'rows')] += len(rows)

        # --- 母集団と遮蔽物の集合 ---
        ordered = []
        for r in rows:
            if is_folder(r) or r['z_global'] == '':
                continue
            c[(fmt, 'nonfolder_z')] += 1
            on, anc_ok = self_on(r), not hidden_anc[r['layer_id']]
            if not on:
                c[(fmt, 'drop_self_hidden')] += 1
            elif not anc_ok:
                c[(fmt, 'drop_ancestor_hidden')] += 1
            if on:
                c[(fmt, 'measured_self_only')] += 1
            if anc_ok:
                c[(fmt, 'measured_ancestors_only')] += 1
            if on and anc_ok:
                c[(fmt, 'measured_visible')] += 1
                for name, ok in occluder_sets(r['composite'] or '0').items():
                    if ok:
                        c[(fmt, 'O_' + name)] += 1
            ordered.append((int(r['z_global']), on and anc_ok, r['composite'] or '0'))

        # --- 既知セル: 上に遮蔽物が無い層は被覆 0 ---
        ordered.sort()
        n = len(ordered)
        suffix = {k: [0] * (n + 1) for k in ('strict', 'wide', 'all')}
        for i in range(n - 1, -1, -1):
            _, on, composite = ordered[i]
            sets = occluder_sets(composite)
            for k in suffix:
                suffix[k][i] = suffix[k][i + 1] + (1 if on and sets[k] else 0)
        for i, (z, _, _) in enumerate(ordered):
            j = i + 1
            while j < n and ordered[j][0] == z:   # 同じ z の層は「上」に数えない
                j += 1
            for k in suffix:
                if suffix[k][j] == 0:
                    c[(fmt, 'IC1_' + k)] += 1

        # --- クリップ層の下地 ---
        siblings = collections.defaultdict(list)
        for r in rows:
            if r['order_in_parent'] != '' and r['parent_id'] != '':
                siblings[r['parent_id']].append(r)
        for kids in siblings.values():
            kids.sort(key=lambda r: int(r['order_in_parent']))
            for i, r in enumerate(kids):
                if (r['clip'] or '0') != '1' or is_folder(r):
                    continue
                base['n'] += 1
                j, steps = i - 1, 0
                while j >= 0 and (kids[j]['clip'] or '0') == '1':
                    j -= 1
                    steps += 1
                if j < 0:
                    base['none'] += 1
                    continue
                if steps:
                    base['consecutive'] += 1
                b = kids[j]
                if is_folder(b):
                    base['folder'] += 1
                elif not self_on(b):
                    base['hidden_self'] += 1
                elif hidden_anc[b['layer_id']]:
                    base['hidden_ancestor'] += 1
                else:
                    base['visible'] += 1

        # --- 名前つきフォルダー(黒消し・white): 述語を2つ印字する(T145 の 57/67) ---
        children = collections.defaultdict(list)
        for r in rows:
            children[r['parent_id']].append(r)
        for r in rows:
            if not is_folder(r) or r['layer_name'] not in ('黒消し', 'white'):
                continue
            kids = children.get(r['layer_id'], [])
            f = folders[r['layer_name']]
            f['n'] += 1
            f['with_rows'] += bool(kids)
            f['with_layers'] += any(not is_folder(k) for k in kids)
            if self_on(r):
                f['with_rows_visible'] += bool(kids)
                f['with_layers_visible'] += any(not is_folder(k) for k in kids)

    emit(c, base, folders)


def emit(c, base, folders):
    both = lambda key: c[('clip', key)] + c[('lip', key)]
    print('# 登録した既知セルと帯 (tools/occlusion/registered_cells.py)')
    print('# 表示 = visibility の1の位 ∧ opacity>0、祖先のフォルダーも同じ規則で見る')
    print()
    print('%-28s %10s %10s %10s' % ('', 'clip', 'lip', '両形式'))
    rows = [
        ('層の行', 'rows'),
        ('重なり順の非フォルダー層', 'nonfolder_z'),
        ('measured_all = 同上', 'nonfolder_z'),
        ('measured_self_only', 'measured_self_only'),
        ('measured_ancestors_only', 'measured_ancestors_only'),
        ('measured_visible (主)', 'measured_visible'),
        ('  外した: 自身が非表示', 'drop_self_hidden'),
        ('  外した: 祖先が非表示', 'drop_ancestor_hidden'),
        ('IC4 O_strict', 'O_strict'),
        ('IC4 O_wide', 'O_wide'),
        ('IC4 O_all', 'O_all'),
        ('IC1 既知セル strict', 'IC1_strict'),
        ('IC1 既知セル wide', 'IC1_wide'),
        ('IC1 既知セル all', 'IC1_all'),
    ]
    for label, key in rows:
        print('%-28s %10d %10d %10d' % (label, c[('clip', key)], c[('lip', key)], both(key)))

    print()
    print('# 2つの述語(自身だけ / 祖先も見る)が一致する範囲')
    print('#   食い違うのは「自身は表示・祖先が非表示」の層だけ (P1 ⊇ P2)')
    for fmt in ('clip', 'lip'):
        n, d = c[(fmt, 'nonfolder_z')], c[(fmt, 'drop_ancestor_hidden')]
        print('  %-6s 1 - %d/%d = %.4f' % (fmt, d, n, 1 - d / n if n else 0))
    n, d = both('nonfolder_z'), both('drop_ancestor_hidden')
    print('  %-6s 1 - %d/%d = %.4f' % ('両形式', d, n, 1 - d / n if n else 0))

    print()
    print('# 非フォルダーのクリップ層の下地 (両形式)')
    n = base['n'] or 1
    for label, key in (('下地が表示', 'visible'), ('下地がフォルダー', 'folder'),
                       ('下地が非表示(自身)', 'hidden_self'),
                       ('下地が非表示(祖先)', 'hidden_ancestor'),
                       ('下地が無い', 'none'), ('連続クリップ', 'consecutive')):
        print('  %-22s %7d  %6.2f%%' % (label, base[key], 100 * base[key] / n))
    print('  %-22s %7d' % ('計', base['n']))
    hidden = base['hidden_self'] + base['hidden_ancestor']
    print('  %-22s %7d  %6.2f%%' % ('下地が非表示(合計)', hidden, 100 * hidden / n))

    print()
    print('# 名前つきフォルダー: 子の数え方を2つ印字する')
    for name, f in sorted(folders.items()):
        print('  %s  フォルダー %d' % (name, f['n']))
        print('    直下に行が在る %4d / うち表示 %4d' % (f['with_rows'], f['with_rows_visible']))
        print('    直下に層が在る %4d / うち表示 %4d' % (f['with_layers'], f['with_layers_visible']))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
