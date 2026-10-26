# -*- coding: utf-8 -*-
"""L33 `preview_dark >= 0.5` の門に **margin があるか**。

L32 で、同じ層の2つの版で `preview_dark` が最大 **0.1092** 違うことが出た
(同じ `layer_id`・同じ `ink_px`・同じ `canvas`・同じ `drawn_ratio`)。
保存されたプレビューは**納品された層の集合を描き直したものではない**ので、
`preview_dark` は**納品物に対する被覆の指標としては使えない**(T116)。

**だがそれは「門が間違っている」とは違う**(基盤の指摘)。門が較正されたときの
間隔は 0.25 以上(吹き出し 0.25 / 線画 0.90 / 裁ち落とし 1.00)で、観測されたずれは
0.11。**だから示されていないのは「間違い」ではなく「margin」である。**

**この器具が答える問いは1つだけ**: ベタ選別の3器具の母集団に、
**`preview_dark` が 0.5 から ±0.11 の窓に入る層が何層いるか。**
0 なら門は無傷。いれば**件数と頁**を出す。

**両側を数える**: 窓の上側(通った層)は下にずれれば落ち、
窓の下側(落ちた層)は上にずれれば入る。**片側だけ数えると門は安全に見える。**

門は3つ:
  - `l12_panel_regions` (正解のベタ層) —— `drawn_ratio>0` / `preview_dark>=0.5` /
    `keep4` に在る / 被覆<95% / 第1階層が `/text` `/sketch set` でない
  - `l8_inventory_reconcile` (納品規則の再現) —— `bam>=0.95` / `gm<=12.0` /
    `0.002<=dr<=0.9` / `preview_dark>=0.5` / 名前の一族で切らない
  - `l15_occlusion` (遮蔽物の候補 = **落ちた側**を使う器具)

`l8` は import すると本体が走るので、**定数だけを AST で取り出す**
(写すと離れる。T110 の族)。
"""
import ast, collections, csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from l12_panel_regions import COV_MAX, DARK_MIN, DROP_TOP, INV, KEEP4, top_folder

MARGIN = 0.1092            # L32 で観測した対の中の最大のずれ
L8 = f'{HERE}/l8_inventory_reconcile.py'


def family_cut_from_l8():
    """`FAMILY_CUT` を l8 の原文から読む。**写さない。**"""
    tree = ast.parse(open(L8, encoding='utf-8').read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and \
                any(getattr(t, 'id', None) == 'FAMILY_CUT' for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise RuntimeError('l8 に FAMILY_CUT が無い —— 名前が変わった')


FAMILY_CUT = family_cut_from_l8()


def name_cut(name, ancestors=''):
    low = ((name or '') + ' ' + (ancestors or '')).lower()
    return any(k.lower() in low for k in FAMILY_CUT)


def num(v):
    v = (v or '').strip()
    if v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main():
    print(f'門 `preview_dark >= {DARK_MIN}` の margin   窓 = 0.5 ± {MARGIN}'
          f' = [{DARK_MIN - MARGIN:.4f}, {DARK_MIN + MARGIN:.4f}]')
    print(f'  (窓の幅は L32 で観測した**対の中の最大のずれ**。'
          f'較正時の間隔は 0.25 以上)')
    print(f'  l8 から読んだ FAMILY_CUT: {len(FAMILY_CUT)} 語')

    cov = {(r['page_id'], r['layer_id']): float(r['cov'])
           for r in csv.DictReader(open(KEEP4))}
    rows = list(csv.DictReader(open(INV, encoding='utf-8-sig')))

    def base(r):
        """3門の共通の前置き。**ここで落ちた層はどの門の母集団でもない。**"""
        if r['folder'] != 'False' or r['rendered'] != 'True':
            return False
        dr = num(r['drawn_ratio'])
        return dr is not None and dr > 0

    def pop_l12(r):
        """`preview_dark` **以外**の l12 の門。"""
        k = (r['page_id'], r['layer_id'])
        c = cov.get(k)
        if c is None or c >= COV_MAX:
            return False
        return top_folder(r['ancestors']) not in DROP_TOP

    def pop_l8(r):
        """`preview_dark` **以外**の l8 の門。"""
        bam, gm, dr = (num(r['black_among_drawn']), num(r['drawn_gray_mean']),
                       num(r['drawn_ratio']))
        if None in (bam, gm, dr):
            return False
        return (bam >= 0.95 and gm <= 12.0 and 0.002 <= dr <= 0.9
                and not name_cut(r['name'], r['ancestors']))

    def pop_l15(r):
        """`preview_dark` **以外**の l15 の門(遮蔽物の候補)。"""
        k = (r['page_id'], r['layer_id'])
        c = cov.get(k)
        return c is not None and c < COV_MAX

    gates = (('l12_panel_regions (正解のベタ層)', pop_l12),
             ('l8_inventory_reconcile (納品規則の再現)', pop_l8),
             ('l15_occlusion / l15_occlusion2 (遮蔽物の候補)', pop_l15))

    total_rows = []
    for label, pop in gates:
        pool = [r for r in rows if base(r) and pop(r)]
        pd = [(r, num(r['preview_dark'])) for r in pool]
        empty = [r for r, v in pd if v is None]
        above = [(r, v) for r, v in pd if v is not None and v >= DARK_MIN]
        below = [(r, v) for r, v in pd if v is not None and v < DARK_MIN]
        win_up = [(r, v) for r, v in above if v <= DARK_MIN + MARGIN]
        win_dn = [(r, v) for r, v in below if v >= DARK_MIN - MARGIN]
        print()
        print('-' * 92)
        print(f'{label}')
        print(f'  母集団 ({DARK_MIN} の条件を外した段階) {len(pool)} 層   '
              f'通る {len(above)} / 落ちる {len(below)} / '
              f'`preview_dark` が空 {len(empty)}')
        print(f'  **窓の上側(通っているが下にずれれば落ちる)  {len(win_up)} 層**')
        print(f'  **窓の下側(落ちているが上にずれれば入る)    {len(win_dn)} 層**')
        print(f'  **窓の中 合計 {len(win_up) + len(win_dn)} 層 '
              f'= 母集団の {(len(win_up) + len(win_dn)) / max(len(pool), 1):.4f}**')
        for side, items in (('上側(通→落ちうる)', win_up),
                            ('下側(落→入りうる)', win_dn)):
            if not items:
                print(f'  {side}: **0 層。この側は無傷**')
                continue
            pages = collections.Counter(r['page_id'] for r, _ in items)
            print(f'  {side}: 頁 {len(pages)}')
            for r, v in sorted(items, key=lambda t: abs(t[1] - DARK_MIN)):
                print(f'      pd={v:.4f}  差 {abs(v - DARK_MIN):+.4f}  '
                      f'{r["page_id"]} L{r["layer_id"]:>4}  "{r["name"]}"')
                total_rows.append((label, side, v, r))
        if win_up or win_dn:
            print(f'  (**この層が門の判定を変えうる。** 変わるとは言っていない —— '
                  f'ずれの向きは版によって両方ある)')

    print()
    print('=' * 92)
    if not total_rows:
        print('**3門とも窓の中に層が居ない。門は 0.11 のずれに対して無傷。**')
    else:
        pg = sorted({r['page_id'] for _, _, _, r in total_rows})
        print(f'**窓の中に居た層: 延べ {len(total_rows)}   頁 {len(pg)}**')
        print('  頁: ' + ' '.join(pg))
        print('  **門が「間違っている」ことは示されていない。'
              'margin が無い層がこれだけ在るというだけである。**')
        print('  **そして `preview_dark` の値そのものは納品物に対して使えない**(T116)ので、'
              'この層の扱いは値では決められない')


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
def l8_bar_margin():
    """**l8 の既知セルの棒に、門の変数自身のぶれに対する余裕があるか。**

    l8 の P1 は `recall >= 0.98 かつ precision >= 0.98`。
    納品 beta 496層をちょうど全部取り戻し(recall 1.0000)、余分に3層拾っている
    (precision 0.9940)。**棒までの余裕は recall 側で 9層、precision 側で 7層**。

    観測されたぶれ(0.11)で窓の中に居る層がそれより多ければ、
    **棒は門の変数自身の不確かさより細い。** それを数える。
    """
    import json
    MAN = ['/home/sh1/deepl/lineart/data/incoming_manuscript/ako6/fill/beta_manifest.json',
           '/home/sh1/deepl/lineart/data/incoming_manuscript/ako7/fill/beta_manifest.json']
    rows = list(csv.DictReader(open(INV, encoding='utf-8-sig')))
    pids = {r['page_id'] for r in rows}
    beta = set()
    for mp in MAN:
        man = json.load(open(mp, encoding='utf-8'))
        for p in man['pages']:
            if p['page'] in pids:
                for b in p.get('beta') or []:
                    beta.add((p['page'], str(b['layer_id'])))

    def passes8(r, dark=True):
        bam, gm, dr, pd = (num(r['black_among_drawn']), num(r['drawn_gray_mean']),
                           num(r['drawn_ratio']), num(r['preview_dark']))
        if None in (bam, gm, dr, pd):
            return False
        if dark and pd < DARK_MIN:
            return False
        return (bam >= 0.95 and gm <= 12.0 and 0.002 <= dr <= 0.9
                and not name_cut(r['name'], r['ancestors']))

    live = [r for r in rows if r['folder'] == 'False' and r['rendered'] == 'True']
    repro = {(r['page_id'], r['layer_id']) for r in live if passes8(r)}
    tp = len(repro & beta)
    print()
    print('=' * 92)
    print('**l8 の既知セルの棒 対 門の変数自身のぶれ**')
    print(f'  納品 beta {len(beta)}   こちらの再現 {len(repro)}   一致 {tp}   '
          f'recall {tp / len(beta):.4f}   precision {tp / len(repro):.4f}')
    r_room = len(beta) - int(-(-0.98 * len(beta) // 1))
    print(f'  棒 0.98 までの余裕: recall 側 **{r_room} 層失っても通る**')

    up = [r for r in live if passes8(r, dark=False)
          and DARK_MIN <= (num(r['preview_dark']) or 0) <= DARK_MIN + MARGIN]
    dn = [r for r in live if passes8(r, dark=False)
          and DARK_MIN - MARGIN <= (num(r['preview_dark']) or 0) < DARK_MIN]
    up_beta = [r for r in up if (r['page_id'], r['layer_id']) in beta]
    print(f'  窓の上側 {len(up)} 層   **そのうち納品 beta に居る {len(up_beta)} 層**')
    print(f'  窓の下側 {len(dn)} 層   (上にずれて入れば precision が下がる)')
    new_prec = tp / (len(repro) + len(dn)) if repro else 0
    new_rec = (tp - len(up_beta)) / len(beta) if beta else 0
    print(f'  **最悪の読み** (窓の上側の納品層が全部下にずれる): '
          f'recall {new_rec:.4f} -> {"FAIL" if new_rec < 0.98 else "PASS"}')
    print(f'  **最悪の読み** (窓の下側が全部上にずれて入る):     '
          f'precision {new_prec:.4f} -> {"FAIL" if new_prec < 0.98 else "PASS"}')
    print('  **これは「門が間違っている」ではない。** 0.11 のぶれが同じ向きに'
          'まとまって起きたらという最悪の読みで、**実際の向きは版によって両方ある**')
    print('  (L32: 6対のうち5対は `page00NN` 側が暗い。'
          '**一方向のぶれは観測されていない**)')


if __name__ == '__main__':
    l8_bar_margin()
