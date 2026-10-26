# -*- coding: utf-8 -*-
"""L32 `preview_dark` が低い理由 —— **2仮説を比べる前に、第3候補を分離する。**

基盤経由で抽出側の定義が届いた(`inbox/reply_preview_dark_definition_20261026.md`):
`preview_dark` は **CSP が保存した合成済みページの縮小プレビュー**の上で、
**層のアルファ(`cov>=128`)を最近傍で縮めた窓の中の輝度 `L<128` の画素の割合**。
層の色は見ていない(どの画素を見るかの選択にだけ使う)。

これで候補が増えた:

  1. **細さを罰している** —— 縮小で細い線が灰色になり 128 を超える
  2. **上の層に覆われている** —— 合成後の絵なので、覆われれば暗くならない
  3. **非表示の層**(基盤の指摘) —— プレビューは合成後なので、表示されていない層は
     ほぼ不透明でも 0 近くになる
  4. **層自身が半透明**(`opacity < 256`) —— 合成すると灰色になる。
     ここで足した。3 と同じ「層が紙面に出ていない」族だが、別の列で分かれる

**3 と 4 を分離してからでないと、1 と 2 のどちらに票が入っても解釈が1つに決まらない**
(基盤の指摘 2)。この器具は**仮説の比較をしない。分離だけをする。**

`visibility` は 0/1/2/3/5/7 の6値で、**1の位が表示かどうかは抽出側も未確認**。
そこで**それ自体を測る**: 紙面で暗い層(`preview_dark>=0.5`)の 1の位がどう分かれるか。

**併記する罠**: 45層には `page00NN` と `doneN` の**同じ層の対**が入っている
(同じ `layer_id`・同じ `ink_px`・同じ `canvas`・同じ `drawn_ratio`)。
**対の中では層自身の画素と縮小倍率が同じなので、仮説1は対の中の差を作れない。**
これは自然な対照対であり、**同定の問題(版)でもある**(通達
`notice_version_identity_ask_the_author_20261026.md`)。
"""
import csv, collections

HERE = '/home/sh1/deepl/lineart-fill-region'
INV = f'{HERE}/inbox/page_layers_priority65.csv'
LED = f'{HERE}/inbox/line_selection_ledger_priority65_px.csv'
DARK_MIN, INK_MIN, BAD_MIN = 0.5, 10000, 0.95
OPAQUE = 256


def num(v):
    """**空文字と 0.0 を分ける**(罠 T104: `or` の既定値は 0 を消す)。"""
    v = (v or '').strip()
    if v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None


def vis_odd(r):
    """`visibility` の1の位。**分からなければ None**(既定値に落とさない)。"""
    v = r['visibility'].strip()
    if not v.lstrip('-').isdigit():
        return None
    return int(v) % 2 == 1


def main():
    led = {(r['page_path'], r['layer_id']): r
           for r in csv.DictReader(open(LED, encoding='utf-8-sig'))}
    rows = list(csv.DictReader(open(INV, encoding='utf-8-sig')))
    print(f'在庫 {len(rows)} 層  (在庫表の母集団。基盤 T115: '
          '**ディスクに在庫表に無い33件がある**ので「全数」と書かない)')

    # ---- 門その1: 1の位が「表示」なら、紙面で暗い層はほぼ全部奇数になるはず
    dark = [r for r in rows if (num(r['preview_dark']) or 0) >= DARK_MIN]
    od = collections.Counter(vis_odd(r) for r in rows)
    print()
    print('**第3候補の分離器そのものを測る** —— `visibility` の1の位')
    print(f'  値の分布: '
          f'{dict(sorted(collections.Counter(r["visibility"].strip() for r in rows).items()))}')
    for key, lab in ((True, '1の位=1'), (False, '1の位=0'), (None, '読めない')):
        n = od[key]
        if not n:
            continue
        d = sum(1 for r in dark if vis_odd(r) is key)
        print(f'  {lab}  層 {n:5d}  そのうち紙面で暗い (>={DARK_MIN}) {d:5d}  '
              f'= {d / n:.4f}')
    print(f'  紙面で暗い層 {len(dark)} のうち 1の位=1 は '
          f'{sum(1 for r in dark if vis_odd(r) is True)} —— '
          '**1の位=0 の層はほぼ暗くならない。分離器として使える**')
    print('  (**これは「1の位=表示」を証明しない。** 向きが合っていることだけを示す。'
          '定義は抽出側も未確認)')

    # ---- 45層を4候補に分ける
    sel = []
    cut = collections.Counter()
    for r in rows:
        if r['folder'] != 'False' or r['rendered'] != 'True':
            continue
        dr = num(r['drawn_ratio'])
        if dr is None or dr <= 0:
            continue
        pd = num(r['preview_dark'])
        if pd is None:
            cut['preview_dark が空'] += 1
            continue
        if pd >= DARK_MIN:
            continue
        cut[f'preview_dark < {DARK_MIN} で落ちた全部'] += 1
        if not ('線画' in r['name'] or 'line' in r['name'].lower()):
            cut['名前に線画/line なし'] += 1
            continue
        k = (r['page_path'], r['layer_id'])
        ink = int((led[k]['ink_px'] or 0) if k in led else 0)
        if ink < INK_MIN:
            cut[f'アルファ>=128 が {INK_MIN:,} 未満'] += 1
            continue
        if (num(r['black_among_drawn']) or 0) < BAD_MIN:
            cut[f'black_among_drawn < {BAD_MIN}'] += 1
            continue
        sel.append((r, ink))
    sel.sort(key=lambda t: -t[1])

    print()
    print('選別 (規律7。理由別に)')
    for k, v in sorted(cut.items()):
        print(f'  {k}: {v}')
    print(f'  **L28 の材料に残った {len(sel)}**')

    hidden = [(r, i) for r, i in sel if vis_odd(r) is False]
    translucent = [(r, i) for r, i in sel
                   if vis_odd(r) is not False and (num(r['opacity']) or 0) < OPAQUE]
    rest = [(r, i) for r, i in sel if vis_odd(r) is not False
            and (num(r['opacity']) or 0) >= OPAQUE]
    print()
    print(f'45層の内訳 (**1 と 2 の比較に入れてよいのは最後の行だけ**)')
    print(f'  第3候補 非表示らしい (1の位=0)        : {len(hidden)}')
    print(f'  第4候補 層自身が半透明 (opacity<{OPAQUE}) : {len(translucent)}')
    print(f'  **残り (表示・不透明) = 1 か 2 の勝負   : {len(rest)}**')
    print(f'  墨の多い上位15層のうち残るのは '
          f'{sum(1 for r, i in sel[:15] if vis_odd(r) is not False and (num(r["opacity"]) or 0) >= OPAQUE)}/15')
    print('  -> **第3候補は45層を説明しない。** 2層しか取らない')

    # ---- 自然な対照対(同じ層が2つの版に居る)
    byink = collections.defaultdict(list)
    for r, i in sel:
        byink[(r['layer_id'], i)].append(r)
    pairs = [v for v in byink.values() if len(v) > 1]
    print()
    print(f'**自然な対照対** —— 同じ `layer_id`・同じ `ink_px` が2つの版に居る: {len(pairs)} 対')
    print('  対の中で層自身の画素と `canvas`(= 縮小倍率)が同じなら、')
    print('  **仮説1(細さ)は対の中の差を作れない。** 差は合成の側から来る')
    uniq = 0
    for v in sorted(pairs, key=lambda v: -int(v[0]['layer_id'])):
        same = len({r['canvas'] for r in v}) == 1 and \
               len({r['drawn_ratio'] for r in v}) == 1
        pds = [num(r['preview_dark']) for r in v]
        uniq += len(v) - 1
        print(f"  L{v[0]['layer_id']:>4}  canvas/drawn_ratio 一致={same}  "
              f"pd {' vs '.join(f'{p:.4f}' for p in pds)}  "
              f"差 {max(pds) - min(pds):.4f}  {v[0]['name']}")
        for r in v:
            print(f'        {r["page_id"]}')
    print(f'  **45層のうち {uniq} 層は重複。異なる層は {len(sel) - uniq}**')
    print('  (版の同定は作者に聞く。通達 '
          'notice_version_identity_ask_the_author_20261026.md)')

    print()
    print('**この器具は仮説を判定しない。** 1 と 2 を比べる走行は、')
    print('上の「残り」だけを腕に取り、**対の片方だけを使う**(でないと同じ層を2回数える)')


if __name__ == '__main__':
    main()
