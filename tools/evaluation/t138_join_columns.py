"""T138: **2つの表に同名の列が在るとき、使う前に結んで不一致数を印字する。**

2026-10-27 に基盤が登録した罠(台帳 `8bceaa7`)。供給側の新しい表の `drawn_ratio` が
既存の `layer_colour` と 55,810層中 55,671層で一致し、**食い違う 139層は全部 用紙**。
原因は読み手の差ではなく**定義の差**で、`layer_colour` は用紙を描画色で全面に
塗り直してから測る。**同じ列名が2つの表で違う量を測っていた。**

器具の中なら起動時点検で止められる(T134 (g))。**しかし2つの器具の間には起動が無い。**
だから**突き合わせ自体が唯一の点検**になる —— そしてそれを器具にしないと、
結んだことが記録の中の文にしか残らず、次に表が来たときにまた手で数える。

使い方:
    t138_join_columns.py <表A> <表B> <列名> [--key page_path,layer_id]

印字するもの(**どれも落とさない。規律9**):
  結んだ行数 / 結べない行数(両側) / 両方空 / **不一致数** / 不一致の内訳

**重なりが小さいときは一致しても安心しない。** `page_layers_priority65.csv` と
905頁表は共通 8頁・604行しか無く、残り 6,758行は結べない。
**結べない行について、この点検は何も言っていない。**

**この器具自身に T138 と同じ形の盲点が在る**(2026-10-27、基盤と突き合わせて確認):

    **突き合わせは、2つの表が鍵を共有している所でしか列を比べられない。**
    **単位が違えば、並べる行が存在しない。**

実例: `labels/layer_kind.jsonl` は**面(`region`/`comp`)についての作者の語**
(「線(完成の線画)」)で、供給側の `v_layer_kind` は**層(`layer_id`)についての種別**
(`mask_fill` 等)。**同じ `layer_kind` が2つの単位の2つの量に付いている。**
値が日本語と英語で重なっていないので今日まで衝突していないだけである。
**鍵が `region` と `layer_id` で別なので、この器具では映らない。**

だから作法は2つ: **単位を列名に入れる**(安い直し)。そして
**突き合わせが届かない列名は、単位を手で確かめる。**
"""
import csv, sys, collections

def num(s):
    s = (s or '').strip()
    try:
        return float(s)
    except ValueError:
        return None

def rows(p):
    return list(csv.DictReader(open(p, encoding='utf-8-sig')))

def join(a_path, b_path, col, key, label=None):
    key = tuple(key)
    A, B = rows(a_path), rows(b_path)
    for name, t in (('A', A), ('B', B)):
        missing = [k for k in (*key, col) if k not in t[0]]
        if missing:
            sys.exit(f'表{name} に列が無い: {missing}  (在る列: {list(t[0])})')
    bd = {tuple(r[k] for k in key): r for r in B}
    seen = set()
    joined = unjoined_a = both_blank = 0
    mism = []
    for r in A:
        k = tuple(r[k_] for k_ in key)
        q = bd.get(k)
        if q is None:
            unjoined_a += 1
            continue
        seen.add(k); joined += 1
        x, y = num(r[col]), num(q[col])
        if x is None and y is None:
            both_blank += 1; continue
        if x is None or y is None or abs(x - y) > 1e-9:
            mism.append((r, q, x, y))
    unjoined_b = len(bd) - len(seen)
    print(f'=== T138 突き合わせ: 列 {col!r}  鍵 {key} ===')
    print(f'  A {a_path}  {len(A)}行')
    print(f'  B {b_path}  {len(B)}行')
    print(f'  結んだ {joined} / A 側で結べない {unjoined_a} / B 側で結べない {unjoined_b}'
          f' / 両方空 {both_blank}')
    print(f'  **不一致 {len(mism)}行**')
    if mism:
        # 内訳は名前が在れば名前で。無ければ鍵の先頭で。**理由別に数えて印字する**
        f = 'name' if 'name' in mism[0][0] else key[0]
        by = collections.Counter(r[f] for r, _, _, _ in mism)
        for k_, v in by.most_common(15):
            print(f'     {k_!r:30} {v}')
        for r, q, x, y in mism[:8]:
            print(f'     {r.get(f)!r:20} {key}={tuple(r[k_] for k_ in key)}  A={x} -> B={y}')
    if joined and unjoined_a:
        print(f'  **結べない {unjoined_a}行について、この点検は何も言っていない。**'
              f' 覆いは A の {joined/len(A):.1%}')
    return len(mism)

if __name__ == '__main__':
    # **`--key` の値も位置引数に数えていた**(2026-10-27 に直した) ——
    # docstring が `--key` を使い方に書いているのに、渡すと使い方の印字に落ちていた。
    # **書いてある旗が通らない形。名前は点検ではない、の小さい版である。**
    k = 'page_path,layer_id'
    a, skip = [], False
    for x in sys.argv[1:]:
        if skip:
            k, skip = x, False
        elif x == '--key':
            skip = True
        elif not x.startswith('--'):
            a.append(x)
    if len(a) != 3:
        sys.exit(__doc__)
    sys.exit(1 if join(a[0], a[1], a[2], k.split(',')) else 0)
