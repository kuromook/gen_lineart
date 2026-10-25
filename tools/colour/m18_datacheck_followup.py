"""M18: m17 が出した7点の深掘り。**工程(作者/抽出/こちらの門/不明)を切り分けるため**に、
それぞれ「どちらの側の性質か」を分けられる量を測る。腕は採点しない。

m17 の結果(2026-10-25、既知セル3つ PASS):
  S2 用紙が0枚のページ **11 / 872**
  S6 ベース不明のクリップ層 **16層 / 12ページ**(パイロット8ページでは0だった)
  S8 **`visibility` が 0/1 だけでなく 2,3,4,5,7 を取る**(3 が 15,156行)
  S9 `rendered` が False の行 **17,029 / 72,839**
  N5 層のメタデータ指紋を共有するページ対 **Jaccard>=0.5 が150組**
     (作品を跨ぐ組、`skima12chara` ↔ `skima12charaDone` J=0.990 を含む)
  Q2 `sat_hist` の和と `drawn_ratio × 面積` の相対誤差が5%超 **167層**
     (上位が全部 `dr=1e-06` = 有効桁1桁。**丸めでは説明できるか**を測る)
  S8 `has_gradation_fill` True **12,738 / 72,839**(色の列を信用できない層)

**走らせる前に書く期待**:
  F1 用紙0枚のページは、`layer_type 1584` が無いだけで描画層は持つはず
     (持たないなら抽出の取りこぼし)
  F2 ベース不明の16層は、**下に用紙しかない**のではないか。
     こちらの同定は用紙を兄弟から外しているので、**それならこちらの門**である
  F3 `rendered=False` はフォルダー行とほぼ一致するはず
     (フォルダー 16,263 に対し 17,029。差 766 が何かを数える)
  F4 Q2 の167層は、`drawn_ratio` の**有効桁の丸めで全部説明できる**はず。
     説明できない層が残れば、それは統計量が層と合っていない印である
  F5 N5 の重複は、**登録集合8ページと検定集合の間に跨いでいないか**。
     跨いでいれば分割が漏れている(Track L 1-1/1-2 と同じ費用)
  F6 `has_gradation_fill` が真の層のうち、**クリップのベース**(= 標準になる層)の数。
     標準は色の列を使わないので、効くのは falsifier と B1 だけのはず
  **F8 帰無値(N5 の指紋の重複に対して)**: 層の指紋をページに**無作為に振り直す**
     (ページごとの層数は保ったまま)。**期待: Jaccard>=0.5 の組は 0。**
     0 でなければ、150組のうちいくらかは偶然の一致である
  F7 `visibility` の値ごとに、他の列(`rendered` / `drawn_ratio` / `opacity`)が
     どう違うか。**意味は抽出側に聞く。こちらで決めない**
"""
import collections
import csv
import hashlib
import sys
from pathlib import Path

CSV872 = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_872.csv')
PAPER = '1584'
SAT = [f'sat_{i*8:03d}_{i*8+7:03d}' for i in range(32)]
REG8 = 8   # 登録集合 = v2 一覧の先頭8ページ


def num(s, d=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return d


def main():
    rows = list(csv.DictReader(open(CSV872, encoding='utf-8')))
    bypage = collections.defaultdict(list)
    for r in rows:
        bypage[r['page_path']].append(r)
    pages = list(dict.fromkeys(r['page_path'] for r in rows))
    reg = set(pages[:REG8])
    drawn = [r for r in rows if r['folder'] != 'True'
             and (num(r['drawn_ratio'], 0) or 0) > 0]

    print('== F1 用紙が0枚のページ ==')
    nopap = [p for p, v in bypage.items() if not any(
        r['layer_type'] == PAPER for r in v)]
    for p in nopap:
        v = bypage[p]
        d = [r for r in v if r['folder'] != 'True'
             and (num(r['drawn_ratio'], 0) or 0) > 0]
        tot = sum(num(r['drawn_ratio'], 0) for r in d)
        print(f'  {p.replace(chr(92), "/")[14:]:52s} 層{len(v):>4} '
              f'描画層{len(d):>4} 描画比の和 {tot:7.3f} '
              f'canvas {v[0]["canvas"]}')
    print(f'  -> 用紙0枚 {len(nopap)} ページ、うち描画層が1枚も無いページ '
          f'{sum(1 for p in nopap if not [r for r in bypage[p] if r["folder"] != "True" and (num(r["drawn_ratio"], 0) or 0) > 0])}')

    print('\n== F2 ベース不明のクリップ層の「下に何があるか」 ==')
    found = collections.Counter()
    detail = []
    for p, v in bypage.items():
        sib = collections.defaultdict(list)
        for r in v:
            if r['parent_id'] == '':
                continue
            sib[r['parent_id']].append(r)
        for par, g in sib.items():
            g.sort(key=lambda r: int(r['order_in_parent']))
            base = None
            for r in g:
                if r['layer_type'] == PAPER:
                    # m17/追補10 の同定は用紙を兄弟から外している
                    continue
                if r['clip'] == '0':
                    base = r
                elif base is None:
                    below = [x for x in g
                             if int(x['order_in_parent']) < int(r['order_in_parent'])]
                    kinds = {('用紙' if x['layer_type'] == PAPER else
                              'フォルダー' if x['folder'] == 'True' else
                              'クリップ層' if x['clip'] == '1' else '描画層')
                             for x in below}
                    key = ','.join(sorted(kinds)) if kinds else '**下に兄弟が無い**'
                    found[key] += 1
                    detail.append((p, r['layer_id'], r['name'], key))
    for k, c in found.most_common():
        print(f'  {k:34s} {c:>3} 層')
    for d in detail[:6]:
        print(f'    {Path(d[0].replace(chr(92),"/")).name:22s} L{d[1]:>4} '
              f'{d[2][:18]:20s} {d[3]}')

    print('\n== F3 rendered=False の中身 ==')
    rf = [r for r in rows if r['rendered'] == 'False']
    print(f'  rendered=False {len(rf)}  うちフォルダー '
          f'{sum(1 for r in rf if r["folder"] == "True")}  '
          f'非フォルダー {sum(1 for r in rf if r["folder"] != "True")}')
    nf = [r for r in rf if r['folder'] != 'True']
    print(f'  非フォルダーの内訳: layer_type '
          f'{dict(collections.Counter(r["layer_type"] for r in nf).most_common(5))}')
    print(f'    drawn_ratio 空 {sum(1 for r in nf if r["drawn_ratio"] == "")} / '
          f'0 {sum(1 for r in nf if num(r["drawn_ratio"], -1) == 0)} / '
          f'>0 {sum(1 for r in nf if (num(r["drawn_ratio"], 0) or 0) > 0)}')
    print(f'    visibility {dict(collections.Counter(r["visibility"] for r in nf))}')
    print(f'  フォルダー行で rendered=True のもの: '
          f'{sum(1 for r in rows if r["folder"] == "True" and r["rendered"] == "True")}')

    print('\n== F4 Q2 の5%超は drawn_ratio の丸めで説明できるか ==')
    unexplained = []
    over = 0
    for r in drawn:
        cv = r['canvas'].strip('[]').split(',')
        area = int(cv[0]) * int(cv[1])
        s = sum(int(r[c]) for c in SAT if r[c] != '')
        d = num(r['drawn_ratio'], 0)
        if abs(s - d * area) / (d * area) <= 0.05:
            continue
        over += 1
        # 印字された有効桁の丸めで、和から計算した比が同じ値になるか
        txt = r['drawn_ratio']
        sig = len(txt.split('e')[0].replace('.', '').replace('-', '').lstrip('0')) \
            if 'e' in txt else len(txt.replace('.', '').lstrip('0'))
        sig = max(1, sig)
        implied = s / area
        same = f'{implied:.{sig-1}e}' == f'{d:.{sig-1}e}'
        if not same:
            unexplained.append((abs(s - d * area) / (d * area), r, implied))
    unexplained.sort(key=lambda t: -t[0])
    print(f'  5%超 {over} 層、うち有効桁の丸めで説明できない {len(unexplained)} 層')
    # F4b 仮説: drawn_ratio は小数6桁に丸められている(min が 1.00e-06 だった)。
    # それなら round(和/面積, 6) が申告値と一致するはずである。全描画層で検定する。
    bad6 = []
    for r in drawn:
        cv = r['canvas'].strip('[]').split(',')
        area = int(cv[0]) * int(cv[1])
        s6 = sum(int(r[c]) for c in SAT if r[c] != '')
        if round(s6 / area, 6) != round(num(r['drawn_ratio'], 0), 6):
            bad6.append((r, s6 / area))
    print(f'  **F4b 仮説「drawn_ratio = round(和/面積, 6)」: 合わない層 '
          f'{len(bad6)} / {len(drawn)}**')
    for r, imp in bad6[:6]:
        print(f'      申告 {r["drawn_ratio"]} 和からの比 {imp:.8f} '
              f'{Path(r["page_path"].replace(chr(92),"/")).name} L{r["layer_id"]}')
    for e, r, imp in unexplained[:8]:
        print(f'    誤差 {e:.3f} 申告 {r["drawn_ratio"]} 和からの比 {imp:.3e}  '
              f'{Path(r["page_path"].replace(chr(92),"/")).name} L{r["layer_id"]} '
              f'{r["name"][:16]} grad={r["has_gradation_fill"]}')

    print('\n== F5 指紋の重複が登録集合と検定集合を跨いでいないか ==')
    def sig(r):
        return hashlib.md5('|'.join((
            r['name'], r['layer_type'], r['drawn_ratio'], r['sat_mean'],
            r['top_colour_share'], r['colours_q16'])).encode()).hexdigest()[:16]
    inv, per = collections.defaultdict(set), collections.defaultdict(set)
    for r in drawn:
        s = sig(r)
        inv[s].add(r['page_path'])
        per[r['page_path']].add(s)
    shared = collections.Counter()
    for s, ps in inv.items():
        if 1 < len(ps) <= 12:
            pl = sorted(ps)
            for i in range(len(pl)):
                for j in range(i + 1, len(pl)):
                    shared[(pl[i], pl[j])] += 1
    pairs = []
    for (a, b), c in shared.items():
        u = len(per[a]) + len(per[b]) - c
        if u and c / u >= 0.5:
            pairs.append((c / u, c, a, b))
    cross = [p for p in pairs if (p[2] in reg) != (p[3] in reg)]
    print(f'  Jaccard>=0.5 の組 {len(pairs)}  '
          f'**登録集合と検定集合を跨ぐ組 {len(cross)}**')
    for j, c, a, b in sorted(cross, reverse=True):
        ra = '登録' if a in reg else '検定'
        rb = '登録' if b in reg else '検定'
        print(f'    J={j:.3f} 共有{c:>4}  [{ra}] {Path(a.replace(chr(92),"/")).name}'
              f'  <->  [{rb}] {Path(b.replace(chr(92),"/")).name}')
    # 重複の塊の大きさ
    par = {p: p for p in pages}
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for _, _, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    cl = collections.Counter(find(p) for p in pages)
    big = sorted((c for c in cl.values() if c > 1), reverse=True)
    print(f'  重複の塊: {len(big)} 個、含まれるページ {sum(big)} / {len(pages)}  '
          f'大きさ {big[:12]}')
    cross_work = 0
    byroot = collections.defaultdict(list)
    for p in pages:
        byroot[find(p)].append(p)
    for root, ps in byroot.items():
        if len(ps) > 1 and len({tuple(p.replace('\\', '/').split('/')[2:4])
                                for p in ps}) > 1:
            cross_work += 1
    print(f'  うち**作品を跨ぐ塊 {cross_work}** 個')

    print('\n== F8 帰無値: 指紋をページに無作為に振り直す(層数は保つ) ==')
    import random
    allsig = [sig(r) for r in drawn]
    sizes = list(collections.Counter(
        r['page_path'] for r in drawn).items())
    for seed in (0, 1, 2):
        random.Random(seed).shuffle(allsig)
        per_n, i = {}, 0
        for p_, k in sizes:
            per_n[p_] = set(allsig[i:i + k]); i += k
        inv_n = collections.defaultdict(set)
        for p_, ss in per_n.items():
            for x in ss:
                inv_n[x].add(p_)
        sh = collections.Counter()
        for x, ps in inv_n.items():
            if 1 < len(ps) <= 12:
                pl = sorted(ps)
                for a_ in range(len(pl)):
                    for b_ in range(a_ + 1, len(pl)):
                        sh[(pl[a_], pl[b_])] += 1
        hi = sum(1 for (a_, b_), c in sh.items()
                 if c / (len(per_n[a_]) + len(per_n[b_]) - c) >= 0.5)
        mx = max((c for c in sh.values()), default=0)
        print(f'  seed {seed}: Jaccard>=0.5 の組 **{hi}**  '
              f'共有数の最大 {mx}  (実測は 150 組、共有最大 204)')

    print('\n== F6 has_gradation_fill とクリップのベース ==')
    sib = collections.defaultdict(list)
    for r in rows:
        if r['layer_type'] == PAPER or r['parent_id'] == '':
            continue
        sib[(r['page_path'], r['parent_id'])].append(r)
    bases = []
    for g in sib.values():
        g.sort(key=lambda r: int(r['order_in_parent']))
        base = None
        for r in g:
            if r['clip'] == '0':
                base = r
            elif base is not None and base not in bases:
                bases.append(base)
    gb = [r for r in bases if r['has_gradation_fill'] == 'True']
    print(f'  クリップのベース {len(bases)}  うち has_gradation_fill '
          f'**{len(gb)}** ({len(gb)/len(bases):.3f})')
    print(f'  ベースのうちフォルダー {sum(1 for r in bases if r["folder"] == "True")}  '
          f'描画0 {sum(1 for r in bases if num(r["drawn_ratio"], -1) == 0)}  '
          f'visibility!=1 {sum(1 for r in bases if r["visibility"] != "1")}')

    print('\n== F7 visibility の値ごとの性質(意味は抽出側に聞く) ==')
    print(f'  {"vis":>4} {"行":>7} {"フォルダー":>7} {"描画>0":>7} '
          f'{"rendered=F":>10} {"opacity!=256":>12} {"clip=1":>7}')
    for v, c in collections.Counter(r['visibility'] for r in rows).most_common():
        g = [r for r in rows if r['visibility'] == v]
        print(f'  {v:>4} {c:>7} '
              f'{sum(1 for r in g if r["folder"] == "True"):>7} '
              f'{sum(1 for r in g if (num(r["drawn_ratio"], 0) or 0) > 0):>7} '
              f'{sum(1 for r in g if r["rendered"] == "False"):>10} '
              f'{sum(1 for r in g if r["opacity"] not in ("256", "")):>12} '
              f'{sum(1 for r in g if r["clip"] == "1"):>7}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
