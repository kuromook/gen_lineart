"""原案の腕 第1段: **「塗りの入っていないマスク層」は鍵を変えても同じ集合になるか**。

**事前登録は 追補177**(走行前に commit `6ed6720`)。**この器具は点数を出さない。**
鍵どうしの一致(Jaccard)と、鍵ごとの枚数を**頁ごとに**印字する。

母集団 **`pop_5pages_no_page0005`**(`page0005` は作者の裁定で除外。追補175)。
**ただし門 IC3 と負の対照 NC1 は `page0005` で当てる** ——
**門と対照は点数ではないので、母集団の外の頁で当ててよい。当てたことを印字する。**

鍵は追補177 のとおり K0〜K7。**どれかを選ばない。全部並べる。**
"""
import collections
import csv
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m12_score_a0_nulls as m12                                     # noqa: E402
import m16_score_corrected_standard as m16                           # noqa: E402
import m77_standard_visible as m77                                   # noqa: E402
import m102_b_standard_v2 as m102                                    # noqa: E402
import m104_b_standard_v3 as m104                                    # noqa: E402

OUT = Path('results/m113_mask_keys')
V3 = Path('results/m104_b_standard_v3/b_standard_v3.csv')
EXCLUDED = 'page0005'                 # 作者の裁定で母集団から外した頁(門と対照には使う)
IC2_K7_5PAGES = 61                    # 92(第3版 IC11)− 31(同 page0005)
IC3 = {'K2': 0, 'K3': 27}             # `page0005` での枚数(追補172 §D)
NC1_LAYER = '238'                     # 作者がパターンブラシと名指しした層
MASKFILL = 'mask_fill'
FLAT_FORMATS = {'2', '1', '1bit_alpha'}


def keyset(rows, kinds, src, *, vis_rule='and1', leaves_only=True):
    """鍵ごとに選ばれた層の `layer_id` の集合。**落とした数も返す**(規律9)。"""
    drop = collections.Counter()
    body = []
    for r in rows:
        v = int(r['visibility'] or 0)
        ok = (v & 1) if vis_rule == 'and1' else (v == 1)
        if not ok:
            drop['非表示'] += 1
            continue
        if leaves_only and r['folder'] == 'True':
            drop['フォルダー'] += 1
            continue
        if r['layer_type'] == m104.PAPER:
            drop['用紙'] += 1
            continue
        body.append(r)

    def lid(r):
        return str(r['layer_id'])

    def kind(r):
        return kinds.get((src, lid(r)))

    def nm(r):
        return (r['name'] or '')

    def flat(r):
        try:
            return float(r['sat_mean'] or -1) == 0.0 and int(r['colours_q16'] or -1) == 1
        except ValueError:
            return False

    K = {
        'K0': set(),
        'K1': {lid(r) for r in body},
        'K2': {lid(r) for r in body if kind(r) == MASKFILL},
        'K3': {lid(r) for r in body if 'MASK' in nm(r).upper()},
        'K4': {lid(r) for r in body if flat(r)},
        'K5': {lid(r) for r in body if (r['formats'] or '') in FLAT_FORMATS},
        'K6': {lid(r) for r in body if 'MASK' in nm(r).upper() and flat(r)},
        'K7': {lid(r) for r in body
               if r['clip'] != '1' and nm(r) in ('color', 'MASK')},
    }
    K['NC1'] = K['K7'] | {lid(r) for r in body if 'color' in nm(r).lower()}
    return K, drop, body


def jac(a, b):
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = m102.load_kinds()
    byp = m16.page_rows()
    v3 = {r['page'] for r in csv.DictReader(open(V3, encoding='utf-8-sig'))}
    pages = [p for p in json.load(open(m12.MANIFEST, encoding='utf-8'))['pages']
             if p['page'] in v3]
    pop = [p for p in pages if EXCLUDED not in p['page']]
    exc = [p for p in pages if EXCLUDED in p['page']]
    print(f'IC1 母集団 `pop_5pages_no_page0005` 期待 5 / 実測 {len(pop)}'
          + ('  PASS' if len(pop) == 5 else '  **FAIL**'))
    if len(pop) != 5:
        return 1

    res, drops = {}, collections.Counter()
    for p in pop + exc:
        vis, dr0 = m77.visible_rows(byp[p['source']])
        K, drop, body = keyset(byp[p['source']], kinds, p['source'])
        res[p['page']] = dict(K=K, n_body=len(body))
        for k, v in drop.items():
            drops[k] += v

    k7 = sum(len(res[p['page']]['K']['K7']) for p in pop)
    print(f'IC2 K7 が選ぶ層(5頁) 期待 {IC2_K7_5PAGES} / 実測 {k7}'
          + ('  PASS' if k7 == IC2_K7_5PAGES else '  **FAIL**'))

    e = res[exc[0]['page']]['K']
    ic3 = all(len(e[k]) == v for k, v in IC3.items())
    print(f'IC3 `{EXCLUDED}`(**母集団の外。門として当てる**) '
          + ' / '.join(f'{k} 期待 {v} 実測 {len(e[k])}' for k, v in IC3.items())
          + ('  PASS' if ic3 else '  **FAIL**'))
    nc1 = NC1_LAYER in e['NC1'] and NC1_LAYER not in e['K7']
    print(f'NC1 「名前に `color` を含む」を混ぜると層{NC1_LAYER} が入る '
          f'(K7 に {"在る" if NC1_LAYER in e["K7"] else "無い"} / '
          f'NC1 に {"在る" if NC1_LAYER in e["NC1"] else "無い"})'
          + ('  PASS' if nc1 else '  **FAIL**'))
    if not (k7 == IC2_K7_5PAGES and ic3 and nc1):
        print('**門または対照が落ちた。以降の行は読まない**')
        return 1

    # NC2 / NC3(**期待は追補177 に書いた**)
    print('\n== 負の対照 2・3(**母集団の5頁で回す**) ==')
    for nm_, kw in (('NC2 `visibility == 1` だけを可視とする(誤り)',
                     dict(vis_rule='eq1')),
                    ('NC3 葉の条件を外す(フォルダーも入れる)',
                     dict(leaves_only=False))):
        tot = collections.Counter()
        for p in pop:
            K2, _, _ = keyset(byp[p['source']], kinds, p['source'], **kw)
            for k in ('K2', 'K3', 'K4', 'K5', 'K6', 'K7'):
                tot[k] += len(K2[k])
        base = {k: sum(len(res[p['page']]['K'][k]) for p in pop)
                for k in ('K2', 'K3', 'K4', 'K5', 'K6', 'K7')}
        print(f'  {nm_}')
        print('    ' + ' / '.join(f'{k} {base[k]}→{tot[k]}' for k in base))

    print('\n== 落とした数(理由別。規律9。**6頁ぶんの延べ**) ==')
    print('  ' + ' / '.join(f'{k} {v}' for k, v in drops.most_common()))

    KEYS = ('K0', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7')
    print('\n=== A. 鍵ごとに選ばれた層の数(**頁ごと。プールしない**。条例2) ===')
    print(f'{"頁":<44}{"可視の葉":>9}' + ''.join(f'{k:>6}' for k in KEYS))
    for p in pop:
        r = res[p['page']]
        print(f'{p["page"]:<44}{r["n_body"]:>9}'
              + ''.join(f'{len(r["K"][k]):>6}' for k in KEYS))
    print(f'{"(母集団の外)" + exc[0]["page"][-12:]:<44}'
          f'{res[exc[0]["page"]]["n_body"]:>9}'
          + ''.join(f'{len(res[exc[0]["page"]]["K"][k]):>6}' for k in KEYS))
    print(f'{"5頁の合計":<44}{sum(res[p["page"]]["n_body"] for p in pop):>9}'
          + ''.join(f'{sum(len(res[p["page"]]["K"][k]) for p in pop):>6}'
                    for k in KEYS))

    print('\n=== B. 鍵どうしの Jaccard(**頁ごと。K0 と K1 は帰無なので外す**) ===')
    REAL = ('K2', 'K3', 'K4', 'K5', 'K6', 'K7')
    pairs = list(itertools.combinations(REAL, 2))
    print(f'{"頁":<30}' + ''.join(f'{a}/{b:>3}' for a, b in pairs))
    worst = {}
    for p in pop:
        K = res[p['page']]['K']
        cells = []
        for a, b in pairs:
            j = jac(K[a], K[b])
            cells.append(f'{j:>7.2f}')
            worst[(a, b)] = min(worst.get((a, b), 1.0), j)
        print(f'{p["page"][-29:]:<30}' + ''.join(cells))
    print(f'{"**最悪の頁**":<30}'
          + ''.join(f'{worst[(a, b)]:>7.2f}' for a, b in pairs))
    print('  **連言は最悪で判定する**(条例1)。1.00 でない組が1つでも在れば C2')

    print('\n=== C. 判定(追補177 の C1〜C4) ===')
    allone = all(v == 1.0 for v in worst.values())
    print(f'  C1 すべての組で Jaccard = 1.00 → '
          f'**{"当たり。鍵は一つに決まる" if allone else "外れ"}**')
    if not allone:
        bad = sorted(worst.items(), key=lambda kv: kv[1])
        print(f'  C2 一致しない組が在る → **当たり。'
              f'原案の腕は鍵を名前に入れる**(最悪 {bad[0][0][0]}/{bad[0][0][1]} '
              f'= {bad[0][1]:.2f})')
        print('    一致しない組(最悪の頁の値。小さい順):')
        for (a, b), v in bad:
            if v < 1.0:
                print(f'      {a}/{b} {v:.2f}')
    print(f'  C3 K4 が層{NC1_LAYER} を除くか('
          f'`{EXCLUDED}` の門で確認。**母集団の外**): '
          f'**{"除く" if NC1_LAYER not in e["K4"] else "除かない"}**')
    print('    **言えるのはここまで** —— 「④ が入っていない」ではない(同定は凍結。追補176)')
    empty = [p['page'] for p in pop
             if any(not res[p['page']]['K'][k] for k in REAL)]
    print(f'  C4 どれかの鍵で0枚になる頁: {len(empty)} / 5')
    for pg in empty:
        z = [k for k in REAL if not res[pg]['K'][k]]
        print(f'      {pg[-29:]:<30} 0枚の鍵: {" ".join(z)}')

    rows_out = []
    for p in pop + exc:
        K = res[p['page']]['K']
        for k in KEYS:
            for lid in sorted(K[k], key=int):
                rows_out.append(dict(page=p['page'],
                                     in_population=int(EXCLUDED not in p['page']),
                                     key=k, layer_id=lid))
    with open(OUT / 'key_members.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['page', 'in_population', 'key',
                                          'layer_id'])
        w.writeheader()
        w.writerows(rows_out)
    print(f'\n表: {OUT / "key_members.csv"}')
    print('**この走行は点数を出していない**(原案の腕の点数・帰無値・棒は第2段。追補177)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
