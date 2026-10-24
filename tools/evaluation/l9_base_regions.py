# -*- coding: utf-8 -*-
"""L9: 作者の言う「ベース面」が優先65ページの一覧に実在するか。

作者の回答(2026-10-24、規律6 により作者の判断):
  - キャラクターの仕上げを「ベース+濃淡」でやったモノクロ原稿は存在するが**ほんの僅か**
  - **3Dレンダリングから作成した背景**は「ベース領域の上に線画クリッピング」の構成が多い。
    キャラクター線画ではないので基本的に対象ではないが、**コマの排他的な部分を
    表すことが多いので、このベース面は使いようがあるかもしれない**
  - 背景(背面処理トーン)の配置にも同じ手法が使われている

事前登録(結果を見る前に書いた):

  KC 既知セル。page0028 の `bg base` は beta_manifest の excluded に
     category=light_fill / drawn_ratio=0.129759 で載っている。
     一覧から同じ値が出れば join は正しい。

  P4 一覧に rendering* / base / 背景 の一族が実在する。
     ako6 と ako7 の両方に出る(作者は作品を限定していない)。

  P5 この一族は納品ベタから系統的に外れており、外した軸は**色**である。
     ベース面は黒くないため。面積の床ではない(ベース面は大きいため)。
     L8 で面を外していた軸は面積の床だったので、**別の一族には別の軸が効く**
     という予測になる。

  P6 ベース面は納品ベタより**大きい**。コマの排他的な部分を表すなら、
     1層あたりの描画割合の中央値が納品ベタ(dr 中央)を上回る。

入力は回送された写し(md5 9cc040cd4bbe3e4d09144395b19912d7)。
"""
import csv, json, collections, statistics, os, re, sys

INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
MANIFESTS = ['/home/sh1/deepl/lineart/data/incoming_manuscript/ako6/fill/beta_manifest.json',
             '/home/sh1/deepl/lineart/data/incoming_manuscript/ako7/fill/beta_manifest.json']
OUT = '/home/sh1/deepl/lineart-fill-region/results/l9_base_regions'
CANVAS = 7016 * 6543

def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

rows = [r for r in csv.DictReader(open(INV, encoding='utf-8-sig'))]
inv = {(r['page_id'], int(r['layer_id'])): r for r in rows}
pids = {r['page_id'] for r in rows}

beta, exc = {}, {}
for mp in MANIFESTS:
    for p in json.load(open(mp, encoding='utf-8'))['pages']:
        if p['page'] not in pids:
            continue
        for b in (p.get('beta') or []):
            beta[(p['page'], b['layer_id'])] = b
        for e in (p.get('excluded') or []):
            exc[(p['page'], e['layer_id'])] = e

# ---- KC 既知セル ----
kc_key = ('018_2018_ako6_ako6_page0028', 157)
m = exc.get(kc_key)
r = inv.get(kc_key)
kc_ok = (m is not None and r is not None and m['category'] == 'light_fill'
         and abs(f(r['drawn_ratio']) - 0.129759) < 1e-6 and r['name'] == 'bg base')
print(f'KC 既知セル page0028 "bg base": manifest category={m["category"] if m else None} '
      f'dr={m["drawn_ratio"] if m else None} / 一覧 name={r["name"] if r else None} '
      f'dr={r["drawn_ratio"] if r else None}')
print('KC', 'PASS' if kc_ok else 'FAIL')
if not kc_ok:
    sys.exit(1)

# ---- P4 一族の同定 ----
# 作者の言葉から引いた語だけを使う。こちらが性質から選ばない
PAT = re.compile(r'rendering|\bbase\b|背景|背面|bg', re.I)
fam = [r for r in rows
       if r['folder'] == 'False' and r['rendered'] == 'True'
       and (f(r['drawn_ratio']) or 0) > 0
       and PAT.search((r['name'] or '') + ' ' + (r['ancestors'] or ''))]
print(f'\nP4 作者の語に当たる層(描画あり): {len(fam)}  '
      f'/ {len({r["page_id"] for r in fam})} ページ')
w = collections.Counter('ako6' if '_ako6_' in r['page_id'] else 'ako7' for r in fam)
print(f'   作品別: {dict(w)}')
print(f'   名前上位: {collections.Counter(r["name"] for r in fam).most_common(10)}')

# ---- P5 どの軸が外したか ----
def verdict(r):
    k = (r['page_id'], int(r['layer_id']))
    if k in beta:
        return '納品された'
    if k in exc:
        return 'excluded: ' + exc[k]['category']
    return '標準が見ていない'
print('\nP5 納品に対する立場')
for lab, n in collections.Counter(verdict(r) for r in fam).most_common():
    print(f'   {lab:28} {n:5d}')

print('\n   色の性質(納品ベタと並べる)')
def q(sel, lab):
    if not sel:
        print(f'   {lab:26} n=0')
        return
    gm = [f(r['drawn_gray_mean']) for r in sel if f(r['drawn_gray_mean']) is not None]
    bam = [f(r['black_among_drawn']) for r in sel if f(r['black_among_drawn']) is not None]
    dr = [f(r['drawn_ratio']) for r in sel if f(r['drawn_ratio']) is not None]
    print(f'   {lab:26} n={len(sel):5d}  gray中央 {statistics.median(gm):7.2f}  '
          f'bam中央 {statistics.median(bam):.4f}  dr中央 {statistics.median(dr):.5f} '
          f'({int(statistics.median(dr)*CANVAS):,}px)')
q([inv[k] for k in beta], '納品ベタ 496')
q(fam, '作者の語に当たる一族')
q([r for r in fam if verdict(r).startswith('excluded')], '  うち excluded')
q([r for r in fam if verdict(r) == '標準が見ていない'], '  うち標準が未見')
q([r for r in fam if verdict(r) == '納品された'], '  うち納品された')

# ---- P6 大きさ ----
dbeta = statistics.median([f(inv[k]['drawn_ratio']) for k in beta])
dfam = statistics.median([f(r['drawn_ratio']) for r in fam])
print(f'\nP6 大きさ: 一族 dr中央 {dfam:.5f} vs 納品ベタ {dbeta:.5f}  '
      f'→ {"一族のほうが大きい PASS" if dfam > dbeta else "一族のほうが小さい FAIL"}')

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'base_family.csv'), 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ['verdict'])
    wr.writeheader()
    for r in sorted(fam, key=lambda r: (r['page_id'], int(r['layer_id']))):
        d = dict(r); d['verdict'] = verdict(r); wr.writerow(d)
print(f'\n-> {OUT}/base_family.csv')
