# -*- coding: utf-8 -*-
"""L29: `preview_dark < 0.5` が落とした「線画」層を、作者に3択で聞く待ち行列。

**事前登録は要らない**(腕を採点する測定ではなく、作者の答を取る器具)。
ただし**出す層の一覧と門の位置は、走る前に記録する**。

母集団: **現行40頁のみ**(`done*` は作者が「作業途中の退避。背景や仕上げが抜けている」
と答えた 2026-10-26 ので外す。**背景が抜けた頁の線画層は別の問い**になる)。

  現行40頁で `preview_dark < 0.5` が落とした層
    名前に `線画` / `line` を含み (**名前は判定ではなく、確かめる理由**)
    アルファ>=128 の画素が 10,000 以上
    `black_among_drawn >= 0.95` (縁がぼけていない)
  -> 34 層

**門(既知セル)**: 作者が L25/L27 で既に **「線(完成の線画)」と答えた層**から5枚。
現行40頁・門を通った側(`preview_dark>=0.5`)・1万画素以上に限る。
**前の走行が記録した答**なので規律2 の形。**画面には印を出さない。**

**画面に出さないもの**: 層名 / `preview_dark` / `keep4` / `black_among_drawn`。
**「線画という名前なのに」が確かめる理由だが、理由は画面に出さない** ——
出すと答が名前に汚染される。L25/L27 の64問が汚染されていないのはこの作法による。

  python tools/ui/l29_build_preview_dark_queue.py
"""
import csv, json, os, random, sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/sh1/deepl/lineart-solid-fill/tools/evaluation')
sys.path.insert(0, os.path.join(HERE, '..', 'evaluation'))
import l25_build_layer_queue as Q

INV = '/home/sh1/deepl/lineart-fill-region/inbox/page_layers_priority65.csv'
LED = '/home/sh1/deepl/lineart-fill-region/inbox/line_selection_ledger_priority65_px.csv'
TRUTH = '/home/sh1/deepl/lineart-fill-region/results/l25_layer_ui/layer_truth.csv'
OUT = '/home/sh1/deepl/lineart-fill-region/results/l29_preview_dark_ui'
SEED = 20261026
N_GATE = 5
INK_MIN, BAD_MIN = 10000, 0.95

WORDS = ['完成の線画である',
         '別の役割の層である(T で一言)',
         'これだけでは判断できない']
ASK = 'このレイヤは完成の線画?'
HELP = ('1枚のレイヤの墨だけを原寸 1:1 で出している。C で頁のどこにあるか'
        '(赤枠が今見ている範囲)。**他のレイヤは描いていない。** '
        '「別の役割」なら T で一言だけ。迷ったら 3。')


def num(v):
    """**空文字と 0.0 を分ける**(罠 T104)。"""
    v = (v or '').strip()
    if v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None


def is_genuine(page_path):
    """現行の頁か。`done*` は作業途中の退避(作者 2026-10-26)。"""
    return not page_path.replace('\\', '/').split('/')[-2].startswith('done')


def main():
    os.makedirs(f'{OUT}/crops', exist_ok=True)
    led = {(r['page_path'], r['layer_id']): r
           for r in csv.DictReader(open(LED, encoding='utf-8-sig'))}
    inv = list(csv.DictReader(open(INV, encoding='utf-8-sig')))

    # ---- 選別。**落としたものを理由別に印字する**(規律7)
    cut, drop40, named, keep = {}, 0, 0, []
    def bump(k):
        cut[k] = cut.get(k, 0) + 1
    for r in inv:
        if r['folder'] != 'False' or r['rendered'] != 'True':
            continue
        dr = num(r['drawn_ratio'])
        if dr is None or dr <= 0:
            continue
        pd = num(r['preview_dark'])
        if pd is None or pd >= 0.5:
            continue
        bump('`preview_dark<0.5` が落とした層 (65頁)')
        if not is_genuine(r['page_path']):
            bump('  `done*` 由来 (作業途中。作者 2026-10-26)')
            continue
        drop40 += 1
        if not ('線画' in r['name'] or 'line' in r['name'].lower()):
            bump('  名前に 線画/line なし')
            continue
        named += 1
        ink = int(led[(r['page_path'], r['layer_id'])]['ink_px'] or 0)
        if ink < INK_MIN:
            bump(f'  アルファ>=128 が {INK_MIN:,} 未満')
            continue
        if (num(r['black_among_drawn']) or 0) < BAD_MIN:
            bump(f'  縁がぼけている (`bad` < {BAD_MIN})')
            continue
        keep.append((r, ink))
    print('選別 (規律7: 落としたものを理由別に)')
    for k, v in cut.items():
        print(f'  {k}: {v}')
    print(f'\n**現行40頁で `preview_dark<0.5` が落とした層: {drop40}**'
          f'  (65頁では {cut["`preview_dark<0.5` が落とした層 (65頁)"]})')
    print(f'**うち名前に 線画/line: {named}**  (65頁では 201)')
    print(f'**聞く層: {len(keep)}**')
    keep.sort(key=lambda t: -t[1])

    # ---- 門: 作者が既に「線(完成の線画)」と答えた層から N_GATE 枚
    invk = {(r['page_id'], r['layer_id']): r for r in inv}
    gate_pool = []
    for t in csv.DictReader(open(TRUTH)):
        if not t['ans'].startswith('線('):
            continue
        i = invk.get((t['pid'], t['lid']))
        if not i or not is_genuine(i['page_path']):
            continue
        if (num(i['preview_dark']) or 0) < 0.5 or int(t['area']) < INK_MIN:
            continue
        gate_pool.append(t)
    rng = random.Random(SEED)
    gate = rng.sample(gate_pool, N_GATE)
    print(f'\n門(既知セル): 作者が既に「線(完成の線画)」と答えた層 '
          f'{len(gate_pool)} から {N_GATE} 枚')

    jobs = [(r['page_id'], ('ako6' if '_ako6_' in r['page_id'] else 'ako7'),
             r['layer_id'], 'ask') for r, _ in keep]
    jobs += [(t['pid'], t['work'], t['lid'], 'gate') for t in gate]
    Q.OUT = OUT
    with Pool(4) as pool:
        recs = [x for x in pool.map(Q.render, jobs) if x]
    ask = [x for x in recs if x['why'] == 'ask']
    gat = [x for x in recs if x['why'] == 'gate']
    rng.shuffle(ask)
    # **門の位置は等間隔でなく、引いて記録する**(位置が読めると印になる)
    pos = sorted(rng.sample(range(len(ask) + len(gat)), len(gat)))
    out, gi, ai = [], 0, 0
    for k in range(len(ask) + len(gat)):
        if gi < len(gat) and k == pos[gi]:
            out.append(gat[gi]); gi += 1
        else:
            out.append(ask[ai]); ai += 1
    for k, r in enumerate(out):
        r['order'] = k
        # **起動の門が要求する鍵**(`make_rec` が全行で空打ちする。
        # 無いと `KeyError` でサーバが起動しない —— 2026-10-25 に72問を無言で
        # 捨てた事故の修理で足した門である。層の問いに面の鍵は無いので、
        # `comp=0` / `region=層の番号` で埋める。L25 の行列と同じ埋め方)
        r['cands'] = []
        r['comp'] = 0
        r['region'] = r['lid']
        r['word_share'] = {}
        r['top_word'] = ''
        r['purity'] = 0.0
        r['n_lay'] = 0
        r['n_lay90'] = 0
        r['auto'] = 0
    q = dict(n=len(out), n_ask=len(ask), n_check=len(gat), n_auto=0,
             n_all=len(out), words=WORDS, ask=ASK, help=HELP,
             others=[['free', '別の役割を入力', 'T']], queue=out)
    json.dump(q, open(f'{OUT}/queue.json', 'w'), ensure_ascii=False)
    # **走る前の記録**: 出した層と門の位置
    with open(f'{OUT}/registered.json', 'w') as fh:
        json.dump(dict(seed=SEED, n_ask=len(ask), n_gate=len(gat),
                       gate_positions=pos,
                       gate=[f'{t["pid"]}_L{t["lid"]}' for t in gate],
                       asked=[f'{x["pid"]}_L{x["lid"]}' for x in ask],
                       population='現行40頁 (done* を外した)',
                       drop40=drop40, named40=named), fh,
                  ensure_ascii=False, indent=1)
    print(f'\n-> {OUT}/queue.json  {len(out)} 問 (聞く {len(ask)} + 門 {len(gat)})')
    print(f'-> {OUT}/registered.json  **門の位置 {pos} を走る前に記録**')


if __name__ == '__main__':
    main()
