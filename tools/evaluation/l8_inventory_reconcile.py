# -*- coding: utf-8 -*-
"""L8: 層の全数一覧(優先65ページ, 7,362層)と既存納品の突き合わせ。

撤回した L7 の偽陽性率は、標準(納品されたベタ層)が見ていない面を
「無い」と数えていた。一覧は選別前の全数なので、標準が何を見ていなかったかを
初めて層単位で数えられる。

事前登録(結果を見る前に書いた):

  P1 既知セル。beta_manifest.json が明記している選別規則
       black_among_drawn >= 0.95 / drawn_gray_mean <= 12.0
       0.002 <= drawn_ratio <= 0.9 / preview_dark >= 0.5
       名前の一族 nombre/balloon/text/furniture を外す
     を一覧に当てると、納品の beta 集合が (page_id, layer_id) で再現する。
     合格は両向き >= 0.98。名前の一族だけは向こうの分類器を持っていないので
     こちらの規則で代用し、食い違いは名前軸かどうかを見る。
     再現しなければ下流は読まない。

  P2 標準が一度も見ていない層 = 描画あり・フォルダーでない・
     beta にも excluded にも居ない。これを数える。
     撤回の根拠が正しければ、この集合に「紙面で暗い」層が実在する。

  P3 P2 の暗い層の black_among_drawn は 0.95 未満に集まり、
     名前は 線画 系が優勢。アンチエイリアスの線画層に塗られたベタなら
     ちょうどそこに落ちるため。

登録した選択(基盤の指摘を受けて明示する):
  - 層かフォルダーかは `folder` 列で決める。`layer_type` とは 141 行食い違うが、
    `folder` と `rendered` は 7,362 行すべてで一致するので軸として自己整合。
    `layer_type` は凡例(1/2/3)に収まらない生値を含む(0/256/800/1584/4098)
  - `用紙` は除く。全65ページで drawn_ratio=1.0 / black_among_drawn=1.0 と
    返るが紙面では白い。実害は無い(preview_dark 最大 0.4153 で暗い集合に
    一度も入らない)が、母数に混ぜない
  - `preview_dark` が空の層(31)は「暗いか判らない」層として別に数える。
    preview_dark>=0.5 で黙って落とさない

入力はすべて絶対パス参照(この木の data/ には何も置かない)。
"""
import csv, collections, json, os, statistics, sys

# 基盤の inbox は往復書簡の置き場で掃除の対象なので、この木に回送された写しを読む。
# md5 9cc040cd4bbe3e4d09144395b19912d7 で基盤側と同一であることを確認済み(2026-10-24)。
INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
MANIFESTS = ['/home/sh1/deepl/lineart/data/incoming_manuscript/ako6/fill/beta_manifest.json',
             '/home/sh1/deepl/lineart/data/incoming_manuscript/ako7/fill/beta_manifest.json']
OUT = '/home/sh1/deepl/lineart-fill-region/results/l8_inventory_reconcile'

# 向こうの名前の一族をこちらで代用する規則(向こうの分類器は持っていない)
FAMILY_CUT = ('nombre', 'ノンブル', 'balloon', 'フキダシ', '吹き出し',
              'text', 'テキスト', '描き文字', 'furniture', '枠', 'mask', 'white')

def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def name_cut(name, ancestors=''):
    # 向こうは祖先フォルダーまで見て一族を決めている(/p/描き文字/effect, /g/skin/MASK)。
    # 最初は層名だけを見て 13 層を拾いすぎ、うち 9 層が向こうの excluded に居た。
    low = ((name or '') + ' ' + (ancestors or '')).lower()
    return any(k.lower() in low for k in FAMILY_CUT)

rows = [r for r in csv.DictReader(open(INV, encoding='utf-8-sig'))]
pids = {r['page_id'] for r in rows}
inv = {(r['page_id'], int(r['layer_id'])): r for r in rows}

beta, excluded = {}, {}
for mp in MANIFESTS:
    man = json.load(open(mp, encoding='utf-8'))
    for p in man['pages']:
        if p['page'] not in pids:
            continue
        for b in p.get('beta') or []:
            beta[(p['page'], b['layer_id'])] = b
        for e in p.get('excluded') or []:
            excluded[(p['page'], e['layer_id'])] = e

print(f'一覧: {len(rows)} 層 / {len(pids)} ページ')
print(f'納品 beta(65ページ分): {len(beta)} 層   excluded: {len(excluded)} 層')

# ---- P1 既知セル ----
missing_in_inv = [k for k in beta if k not in inv]
print(f'\nP1-0 納品 beta のうち一覧に行が無いもの: {len(missing_in_inv)}')

def passes(r):
    bam, gm, dr, pd = (f(r['black_among_drawn']), f(r['drawn_gray_mean']),
                       f(r['drawn_ratio']), f(r['preview_dark']))
    if None in (bam, gm, dr, pd):
        return False
    return (bam >= 0.95 and gm <= 12.0 and 0.002 <= dr <= 0.9
            and pd >= 0.5 and not name_cut(r['name'], r['ancestors']))

repro = {k for k, r in inv.items()
         if r['folder'] == 'False' and r['rendered'] == 'True' and passes(r)}
bset = set(beta)
tp = len(repro & bset)
recall = tp / len(bset) if bset else 0
prec = tp / len(repro) if repro else 0
print(f'P1 規則の再現: 納品を取り戻せた {tp}/{len(bset)} = {recall:.4f}   '
      f'こちらが拾って納品に無い {len(repro - bset)} (精度 {prec:.4f})')
ok = recall >= 0.98 and prec >= 0.98
print('P1', 'PASS' if ok else 'FAIL')

for lab, keys in (('納品にあるがこちらが落とした', bset - repro),
                  ('こちらが拾ったが納品に無い', repro - bset)):
    if not keys:
        continue
    byname = collections.Counter(inv[k]['name'] for k in keys if k in inv)
    n_namecut = sum(1 for k in keys if k in inv and name_cut(inv[k]['name'], inv[k]['ancestors']))
    print(f'  {lab}: {len(keys)} 層  うち名前軸で説明がつく {n_namecut}')
    print(f'    名前上位: {byname.most_common(6)}')
    for k in sorted(keys)[:4]:
        r = inv.get(k)
        if r:
            print(f'    {k[0]} id={k[1]} "{r["name"]}" bam={r["black_among_drawn"]} '
                  f'gm={r["drawn_gray_mean"]} dr={r["drawn_ratio"]} pd={r["preview_dark"]}')

if not ok:
    print('\n既知セルが再現しない。下流は読まない。')
    sys.exit(1)

# ---- P2 標準が一度も見ていない層 ----
seen = bset | set(excluded)
unseen = [r for k, r in inv.items()
          if k not in seen and r['folder'] == 'False' and r['rendered'] == 'True'
          and (f(r['drawn_ratio']) or 0) > 0 and r['name'] != '用紙']
dark = [r for r in unseen if (f(r['preview_dark']) or 0) >= 0.5]
unknown = [r for r in unseen if f(r['preview_dark']) is None]
print(f'\nP2 標準が一度も見ていない層(描画あり): {len(unseen)}')
print(f'   うち紙面で暗い (preview_dark>=0.5): {len(dark)}')
print(f'   うち暗いか判らない (preview_dark が空): {len(unknown)}  '
      f'名前: {collections.Counter(r["name"] for r in unknown).most_common(4)}')
print(f'   参考: 納品 beta {len(bset)} / 候補まで来て切られた excluded {len(excluded)}')

# ---- P3 ----
print('\nP3 暗いのに標準が見ていない層の black_among_drawn')
for lo, hi, lab in ((0.95, 1.01, '>=0.95'), (0.80, 0.95, '0.80-0.95'),
                    (0.50, 0.80, '0.50-0.80'), (0.0, 0.50, '<0.50')):
    sel = [r for r in dark if lo <= (f(r['black_among_drawn']) or 0) < hi]
    area = sum(f(r['drawn_ratio']) or 0 for r in sel)
    print(f'   {lab:10s} {len(sel):5d} 層   紙面に占める描画の総和 {area*100:7.2f}%')
bam = [f(r['black_among_drawn']) or 0 for r in dark]
if bam:
    print(f'   中央 {statistics.median(bam):.4f}')
print('   名前上位:', collections.Counter(r['name'] for r in dark).most_common(10))
nm = sum(1 for r in dark if name_cut(r['name'], r['ancestors']))
print(f'   うち名前で調度と分かるもの(フキダシ/描き文字/ノンブル等): {nm}')

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'unseen_dark.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in sorted(dark, key=lambda r: (r['page_id'], int(r['layer_id']))):
        w.writerow(r)
print(f'\n-> {OUT}/unseen_dark.csv')
