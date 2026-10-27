# -*- coding: utf-8 -*-
"""**作者が確定した覆いの定義を、こちらの65頁の母集団に当てて差の大きさを測る。**

正本は基盤の `doc/occlusion_definition.md`(作者の答、2026-10-27):

> 非表示レイヤーは含めない / 不透明度が0でなければ遮蔽物 /
> クリップ層は下地と合わせて遮蔽範囲を決める / 乗算は遮蔽物ではない /
> レイヤーマスクはマスクが有効化されているなら適用後の範囲が遮蔽範囲

**こちらの O1 / O2 / O2' / O3 の候補の述語は `visibility` の1の位と `/sketch set` と
`z_global` の有無だけで、不透明度も合成モードもクリップも見ていない。**
だから登録の前に**何を構造的に落とす/余計に数えるか**を数で書く(2026-10-27 00:24 の節)。

**器具ではなく表の問題なので、ここは画素を1枚も読まない。** 使うのは:

- 在庫 `inbox/page_layers_priority65.csv`(`visibility` / `opacity` を持つ。**合成とクリップは無い**)
- 供給側の構造の表 `../lineart/inbox/census_r8_layer_structure_20261027.csv`
  (`composite` / `clip` / `z_global` / `is_folder` を持つ。437,561行)

**鍵は `(page_path, layer_id)`。** T138 の作法どおり、**使う前に共有している列で突き合わせた**:
`opacity` / `visibility` / `layer_type` の3列で **7,362行すべてが結び、不一致 0**
(905頁表との 8.2% と違い、**こちらは覆い 100%**)。だから `composite` と `clip` を
こちらの母集団に持ち込める。**結べない行についてこの点検は何も言わない**は、
ここでは A 側 0行なので空である(**その旨を印字する**)。

**`2 = 乗算` は 6層の対応が 19,042層を決めている外挿**(供給側の回答)。
**だから「乗算を外した数」は、その対応表に依存する。** 依存しない帯のために
基盤と同じ三組を印字する: 合成0のみ / 乗算(値2)以外 / 合成を見ない。
"""
import collections
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from l15_occlusion import excluded_layers

INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
R8 = '/home/sh1/deepl/lineart/inbox/census_r8_layer_structure_20261027.csv'
ZCSV = '/home/sh1/deepl/lineart-fill-region/inbox/layer_z_order_priority65.csv'


def topf(anc):
    return (anc or '').strip('/').split('/')[0] if anc else ''


def structure_map(pages=None):
    """供給側の構造の表を `(page_path, layer_id) -> 行` で返す。

    **`composite` / `clip` / `is_folder` の唯一の入口**(T131: 選別の入口を2か所に置かない)。
    表は基盤の `inbox/`(git 管理外)に在る。**こちらの在庫には `composite` も `clip` も無い。**
    `pages` を渡すとその `page_path` だけに絞る(51MB を全部持たないため)。
    """
    out = {}
    for r in csv.DictReader(open(R8, encoding='utf-8-sig')):
        if pages is None or r['page_path'] in pages:
            out[(r['page_path'], r['layer_id'])] = r
    return out


# **2つの軸が在る。混ぜない**(2026-10-27 に気づいた):
#   軸A = **マスクのビット**(登録の腕。`O1` = 1の位=1 かつ 2の位=0 / `O2` = 1の位=1 の全部)
#   軸B = **合成モード**(作者の定義。乗算を外すか / 合成0だけか / 見ないか)
# `O_all` は**軸A では `O2`** である(= 2026-10-26 の実装)。**登録の主値は `O1` の側**なので、
# 作者の定義を当てた主値は **`O1_wide`**(マスクビット0 かつ 乗算でない)になる。
# T142(基盤から回送): **「マスクが在る」(`layer_type` の2の位、構造)と
# 「マスクが効いている」(`visibility` の2の位、状態)は別の量。** ここは**状態**で切る。
OCC_KINDS = ('legacy612', 'O1', 'O1_wide', 'O_all', 'O_wide', 'O_strict')


def ancestor_block(r8, page_path, layer_id):
    """祖先のフォルダーを根まで辿り、**描かれない理由**を返す(無ければ None)。

    T143(2026-10-27、基盤から回送。抽出側が踏んだ): 作者の定義は
    **「非表示レイヤーは含めない」**で、CSP では**非表示フォルダーの中身は描かれない**。
    **だから祖先を見るのが定義である。** こちらの候補の述語は層自身の `visibility` しか
    見ていなかった —— **向きは一方向で、遮蔽物を多く数える**(= 被覆を多く数える)。

    こちらの65頁では **741層の候補のうち 27層(3.64%)が非表示の祖先を持つ**
    (腕 O1 では 613層のうち 24層)。**辿れない形も理由別に返す**(規律9)。

    > **点検の値は定義から導く。表から計算しやすい形で出すと、正しい実装が落ちる。**
    > 抽出側の実装は「IC4 の数を合わせるため祖先を見ない」と書いていた ——
    > **実装者が「点検に合わせるため」と書いた一文は警報である。**
    """
    cur = r8.get((page_path, str(layer_id)))
    if cur is None:
        return '構造の表に無い (祖先を辿れない)'
    seen = set()
    while True:
        pid = (cur.get('parent_id') or '').strip()
        if not pid:
            return None
        if pid in seen:
            return '祖先が輪になっている (辿れない)'
        seen.add(pid)
        par = r8.get((page_path, pid))
        if par is None:
            return '祖先が構造の表に無い (辿れない)'
        v = (par.get('visibility') or '').strip()
        if v.replace('.', '', 1).isdigit() and int(float(v)) & 1 == 0:
            return '祖先のフォルダーが非表示 (CSP では中身が描かれない。T143)'
        o = (par.get('opacity') or '').strip()
        if o.replace('.', '', 1).isdigit() and int(float(o)) == 0:
            return '祖先のフォルダーの不透明度が 0'
        cur = par


def occ_predicate(kind, pages=None, ancestors=False):
    """遮蔽物の候補の述語を返す。**`kind` に既定を置かない**(T104)。

    | kind | 候補 | 役 |
    |---|---|---|
    | `legacy612` | **旧来の等値比較** `visibility == '1'` | **KC53。** 可視のトーンが落ちた側の再現 |
    | `O_all` | `visibility` の1の位=1 | 帯の上端。**T128 を直した後のいまの実装** |
    | `O_wide` | `O_all` から **乗算(composite=2)を外す** | **主値。作者の集合に最も近い** |
    | `O_strict` | `O_all` のうち **合成が 0 のみ** | 帯の下端 |

    作者の定義(2026-10-27 確定)は「表示 ∧ 不透明度>0 ∧ 乗算でない」。
    **不透明度 0 はこちらの候補に 0層**なので、どの kind でも結果は同じになる
    (**0層であることを数えた上で、門は置く**)。
    **クリップの下地との共通部分は、どの kind でも掛けていない** ——
    供給側に下地を辿るコードが無い。**向きは一方向で被覆を多く数える**(登録に書いた)。

    返すのは `keep(page_path, vis, anc, opacity) -> (bool, 理由)`。
    **理由の文字列は呼び手が数える**(規律9)。`O_all` の理由は
    `l15_zorder_occlusion` が 2026-10-26 に印字していたものと同じにしてある
    (**再現の基準を壊さないため**)。
    """
    if kind not in OCC_KINDS:
        raise ValueError(f'kind を書くこと。候補: {OCC_KINDS}')
    need = kind in ('O_wide', 'O_strict', 'O1_wide') or ancestors
    r8 = structure_map(pages) if need else {}

    def keep(page_path, layer_id, vis, anc, opacity):
        if not (vis or '').strip().isdigit():
            return False, 'visibility が数でない'
        if kind == 'legacy612':
            if (vis or '').strip() != '1':
                return False, 'visibility が 1 でない (**旧来の等値比較**)'
        elif int(vis) & 1 == 0:
            return False, 'visibility の1の位=0 (非表示)'
        elif kind in ('O1', 'O1_wide') and int(vis) & 2:
            return False, 'visibility の2の位=1 (マスクが効いている。腕 O1 は外す)'
        if topf(anc) == 'sketch set':
            return False, '/sketch set (下絵は入力画像。L12 の修正A2)'
        o = (opacity or '').strip()
        if o.isdigit() and int(o) == 0:
            return False, '不透明度 0 (作者: 0 でなければ遮蔽物)'
        if ancestors:
            # **祖先のフォルダーの表示・不透明度も見る。これが定義である**(T143)
            why = ancestor_block(r8, page_path, layer_id)
            if why:
                return False, why
        if kind in ('O_wide', 'O_strict', 'O1_wide'):
            r = r8.get((page_path, str(layer_id)))
            c = (r or {}).get('composite', '')
            c = int(float(c)) if c.strip().replace('.', '', 1).isdigit() else None
            if c is None:
                return False, '構造の表に無い (合成モードが分からない)'
            if kind in ('O_wide', 'O1_wide') and c == 2:
                return False, '乗算 (作者: 乗算は遮蔽物ではない)'
            if kind == 'O_strict' and c != 0:
                return False, '合成が通常でない (帯の下端は合成0だけ)'
        return True, ''

    return keep


def main():
    inv = list(csv.DictReader(open(INV, encoding='utf-8-sig')))
    pages = {r['page_path'] for r in inv}
    want = {(r['page_path'], r['layer_id']) for r in inv}
    r8 = {}
    for r in csv.DictReader(open(R8, encoding='utf-8-sig')):
        if r['page_path'] in pages and (r['page_path'], r['layer_id']) in want:
            r8[(r['page_path'], r['layer_id'])] = r
    print(f'在庫 {len(inv):,}層 / {len(pages)}頁   構造の表から結べた {len(r8):,}層')
    if len(r8) != len(inv):
        print(f'  **結べない {len(inv) - len(r8):,}層。この点検はその分について何も言わない**')
    else:
        print('  **A 側で結べない行は 0。覆いは 100%**'
              '(列の一致は t138_join_columns.py で別に確かめた: opacity/visibility/layer_type 不一致 0)')

    # ---- いまの候補の述語(l15_zorder_occlusion.main と同じ3つ)
    zz = {(r['page_id'], r['layer_id']) for r in
          csv.DictReader(open(ZCSV, encoding='utf-8-sig'))}
    vis = {(r['page_id'], r['layer_id']): r for r in inv}
    ex, _ = excluded_layers()
    cand, cut = [], collections.Counter()
    for k, v in ex.items():
        for lid, path, name, reason in v:
            r = vis.get((k[0], str(lid)))
            if r is None:
                cut['在庫に無い'] += 1; continue
            if not r['visibility'].strip().isdigit():
                cut['visibility が数でない'] += 1; continue
            if int(r['visibility']) & 1 == 0:
                cut['visibility の1の位=0 (非表示)'] += 1; continue
            if topf(r['ancestors']) == 'sketch set':
                cut['/sketch set'] += 1; continue
            if (k[0], str(lid)) not in zz:
                cut['z順に無い'] += 1; continue
            cand.append((r['page_path'], str(lid), r))
    print(f'\nいまの候補の述語で通した遮蔽物の候補 **{len(cand):,}層**')
    for k, v in cut.most_common():
        print(f'  落とした {v:5d}  {k}')

    # ---- 作者の定義の3つの門を、その候補に当てる
    def g(key, col):
        r = r8.get(key)
        return (r or {}).get(col, '')

    def ival(s, d=None):
        s = (s or '').strip()
        try:
            return int(float(s))
        except ValueError:
            return d

    mul = [c for c in cand if ival(g((c[0], c[1]), 'composite')) == 2]
    nz = [c for c in cand if (ival(g((c[0], c[1]), 'composite')) or 0) != 0]
    clip = [c for c in cand if str(g((c[0], c[1]), 'clip')).strip() in ('1', 'True', 'true')]
    op0 = [c for c in cand if ival(c[2].get('opacity')) == 0]
    fold = [c for c in cand if str(g((c[0], c[1]), 'is_folder')).strip() in ('1', 'True', 'true')]
    noj = [c for c in cand if (c[0], c[1]) not in r8]

    print('\n作者の定義を当てる(**こちらの候補のうち何が動くか**)')
    print(f'  **乗算(composite=2)        {len(mul):5d}層**  -> 作者の定義では**遮蔽物ではない。外す**')
    print(f'  通常以外すべて(composite!=0) {len(nz):5d}層   (内訳は下。帯の広い側)')
    print(f'  **クリップ(clip=1)         {len(clip):5d}層**  -> 遮蔽範囲は**下地との共通部分**。'
          '今は掛けていないので**被覆を多く数える**(向きは一方向)')
    print(f'  不透明度 0                 {len(op0):5d}層   -> 外す(作者: 0 でなければ遮蔽物)')
    print(f'  フォルダー                 {len(fold):5d}層   -> 候補に入っていれば誤り')
    print(f'  構造の表に無い             {len(noj):5d}層')
    comp = collections.Counter(ival(g((c[0], c[1]), 'composite')) for c in cand)
    print('  合成モードの分布(候補の中): '
          + ' / '.join(f'{k}:{v}' for k, v in sorted(comp.items(), key=lambda t: -t[1])))

    print('\n三組(**`2 = 乗算` の対応表に依存しない帯を取るため。基盤と同じ形**)')
    print(f"  O_strict 合成0のみ        {len(cand) - len(nz) - len(op0):5d}層")
    print(f"  O_wide   値2以外          {len(cand) - len(mul) - len(op0):5d}層")
    print(f"  O_all    合成を見ない      {len(cand) - len(op0):5d}層")
    print(f'  **いまの実装 = O_all から不透明度も外さない形 = {len(cand):,}層**')

    print('\n**この点検が言っていないこと**')
    print('  - マスクの有効フラグ: 在庫の `has_mask` は**持っているか**で、**有効かどうかではない**。'
          '供給側も「無効にした見本が無い」と答えている')
    print('  - クリップの下地: 供給側に**下地を辿るコードが無い**。'
          'だから共通部分は**こちらでも計算できない**(表に親は在るが、下地の規則は未定)')
    print('  - 乗算の同定: `2 = 乗算` は 6層の対応からの外挿である')


if __name__ == '__main__':
    main()
