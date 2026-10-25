"""M17: **依頼3 の全数納品(872ページ)のデータチェック。** 検証・実験・器具づくりの前。

作者の指示(2026-10-25): Track H / L で**おかしな状態のデータ**が見つかっており、
**作者の制作工程によるものか、抽出によるものか、こちらの器具によるものかを
切り分けられないと混濁する**。層の状態・命名・統計量を先に点検する。

**この表は腕を採点しない。** 欄②(工程)の割り当ては `judge_*` にしか書かない。
Track L の起案 版2 と**同じ枠**で出す: **作者の制作工程 / 抽出 / こちらの門 / 不明**。

**既知セル(走らせる前に書いた。再現しなければ以降を読まない)**:
  KC1 先頭8ページの行が `known8.csv` の **1,005行 × 64列と完全一致**
      (抽出側の主張「全行・全列一致」をこちらで数え直す)
  KC2 ページ集合が依頼3 v2 の一覧(`request3_page_list_v2_20261024.csv`)の
      **872ページと一致**し、**順も一致**
  KC3 その8ページの中で、昨日の実測を再現: フォルダー行 **173** /
      描画画素0の層 **126** / 用紙 **8**(`drawn_ratio` 1.0)/
      クリップのベース **183**(フォルダー16・描画0が1)/ ベース不明の連鎖 **0**

**点検する項目(期待は「異常0」ではない。数と分母を出し、工程を後で割り当てる)**:
  層の状態 S1 ページの層数 / S2 用紙の枚数 / S3 親が行として存在しない /
           S4 root の数 / S5 同じ親で order_in_parent が重複 /
           S6 ベース不明のクリップ層 / S7 ページ内 layer_id の重複 /
           S8 visibility・opacity・has_mask・has_gradation_fill /
           S9 canvas がページで一意か・rendered
  命名     N1 空・制御文字・前後空白・異常長 / N2 ページ内の同名 /
           N3 語彙(パイロットの語彙と比べる) / N4 page_path の重複 /
           **N5 層のメタデータの指紋によるページ重複候補**
           (Track L の 1-1/1-2「同一頁の二版重複・作品を跨ぐ頁の共有」が
            カラー側にあるか。ラスタが無くても層の統計量で候補は出せる)
  統計量   Q1 drawn_ratio の範囲 / **Q2 sat_hist の和と drawn_ratio の整合** /
           Q3 top_colour_share・colours_q16・sat_mean の範囲 /
           Q4 形式2 の sat_mean / Q5 列ごとの空率 / Q6 完全重複行

**Q2 の根拠**: パイロットでは `sat_hist` の32箱の和が、用紙で
7016×6543 = 45,905,688 にちょうど一致し、別の層でも
`drawn_ratio × キャンバス面積` と丸めの範囲で合っていた。
**和が合わない層は「層の統計量がその層のものでない」ことの印**になる。
`drawn_ratio` は有効桁で丸めて配られているので、**相対誤差で見て分布を出す**。
"""
import collections
import csv
import hashlib
import sys
import unicodedata
from pathlib import Path

CSV872 = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_872.csv')
CSV8 = Path('/home/sh1/deepl/lineart/inbox/m_colour_page_layers_known8.csv')
V2 = Path('outbox/request3_page_list_v2_20261024.csv')
OUT = Path('results/m17_datacheck'); OUT.mkdir(parents=True, exist_ok=True)
PAPER = '1584'
SAT = [f'sat_{i*8:03d}_{i*8+7:03d}' for i in range(32)]
KC3 = {'folders': 173, 'zero_drawn': 126, 'paper': 8, 'bases': 183,
       'folder_bases': 16, 'orphan': 0}


def load(path):
    with open(path, encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def num(s, d=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return d


def verify_known_cells(rows, rows8):
    print('== 既知セル ==')
    pages8 = [r['page_path'] for r in rows8]
    first8 = list(dict.fromkeys(pages8))
    sub = [r for r in rows if r['page_path'] in set(first8)]
    ok1 = len(sub) == len(rows8) and all(a == b for a, b in zip(sub, rows8))
    print(f'  KC1 先頭8ページの行が known8 と一致: {len(sub)} / {len(rows8)} 行 '
          f'-> {"PASS" if ok1 else "FAIL"}')
    if not ok1 and len(sub) == len(rows8):
        for i, (a, b) in enumerate(zip(sub, rows8)):
            if a != b:
                d = [k for k in a if a[k] != b[k]]
                print(f'    最初の相違 行{i}: {d[:4]} '
                      f'{[(a[k], b[k]) for k in d[:2]]}')
                break
    want = [r['page_path'] for r in load(V2)] if V2.exists() else None
    pages = list(dict.fromkeys(r['page_path'] for r in rows))
    ok2 = want is not None and pages == want
    print(f'  KC2 ページ集合と順が v2 の一覧と一致: {len(pages)} ページ '
          f'(一覧 {len(want) if want else "?"}) -> {"PASS" if ok2 else "FAIL"}')
    if want and not ok2:
        print(f'    集合の差 {len(set(pages) ^ set(want))} / '
              f'順の最初の相違 '
              f'{next((i for i, (a, b) in enumerate(zip(pages, want)) if a != b), None)}')
    return ok1 and ok2, set(first8)


def clip_bases(rows):
    """追補10 §I3 の同定(納品の木の全行。用紙は兄弟から外す)。"""
    sib = collections.defaultdict(list)
    for r in rows:
        if r['layer_type'] == PAPER or r['parent_id'] == '':
            continue
        sib[r['parent_id']].append(r)
    bases, chains, orphan = {}, 0, 0
    for g in sib.values():
        g.sort(key=lambda r: int(r['order_in_parent']))
        base = None
        for r in g:
            if r['clip'] == '0':
                base = r
            else:
                chains += 1
                if base is None:
                    orphan += 1
                else:
                    bases[r['parent_id'] + '/' + base['layer_id']] = base
    return bases, chains, orphan


def main():
    rows = load(CSV872)
    rows8 = load(CSV8)
    ok, first8 = verify_known_cells(rows, rows8)

    bypage = collections.defaultdict(list)
    for r in rows:
        bypage[r['page_path']].append(r)
    sub8 = {p: v for p, v in bypage.items() if p in first8}
    kc3 = {'folders': sum(1 for v in sub8.values() for r in v
                          if r['folder'] == 'True'),
           'zero_drawn': sum(1 for v in sub8.values() for r in v
                             if r['folder'] != 'True' and r['drawn_ratio'] == ''),
           'paper': sum(1 for v in sub8.values() for r in v
                        if r['layer_type'] == PAPER)}
    b = [clip_bases(v) for v in sub8.values()]
    kc3['bases'] = sum(len(x[0]) for x in b)
    kc3['orphan'] = sum(x[2] for x in b)
    kc3['folder_bases'] = sum(1 for x in b for r in x[0].values()
                              if r['folder'] == 'True')
    bad3 = [k for k, v in KC3.items() if kc3.get(k) != v]
    print(f'  KC3 8ページの昨日の実測を再現: ' + ' '.join(
        f'{k} {kc3.get(k)}/{v}' for k, v in KC3.items()) +
        f' -> {"PASS" if not bad3 else "FAIL " + str(bad3)}')
    if not ok or bad3:
        print('\n既知セルが再現しない。以降の行は読まない。')
        return 1

    n, npg = len(rows), len(bypage)
    drawn = [r for r in rows if r['folder'] != 'True' and r['drawn_ratio'] != '']
    print(f'\n== 母集団 ==\n  行 {n}  ページ {npg}  '
          f'フォルダー行 {sum(1 for r in rows if r["folder"] == "True")}  '
          f'描画層 {len(drawn)}  '
          f'描画画素0の層 {sum(1 for r in rows if r["folder"] != "True" and r["drawn_ratio"] == "")}')

    print('\n== S 層の状態 ==')
    sizes = sorted(len(v) for v in bypage.values())
    print(f'  S1 ページの層数: min {sizes[0]} p10 {sizes[npg//10]} '
          f'中央 {sizes[npg//2]} p90 {sizes[npg*9//10]} max {sizes[-1]}  '
          f'(層1枚以下 {sum(1 for s in sizes if s <= 1)} ページ)')
    pap = collections.Counter(sum(1 for r in v if r['layer_type'] == PAPER)
                              for v in bypage.values())
    print(f'  S2 用紙の枚数ごとのページ数: {dict(sorted(pap.items()))}')
    dang = roots = dup_ord = dup_lid = 0
    dang_pages, ord_pages, lid_pages, root_pages = [], [], [], []
    for p, v in bypage.items():
        ids = {r['layer_id'] for r in v}
        d = [r for r in v if r['parent_id'] not in ids and r['parent_id'] != '']
        if d:
            dang += len(d); dang_pages.append(p)
        rt = [r for r in v if r['parent_id'] == '']
        roots += len(rt)
        if len(rt) != 1:
            root_pages.append((p, len(rt)))
        seen = collections.Counter((r['parent_id'], r['order_in_parent'])
                                  for r in v if r['parent_id'] != '')
        dd = sum(c - 1 for c in seen.values() if c > 1)
        if dd:
            dup_ord += dd; ord_pages.append((p, dd))
        c = collections.Counter(r['layer_id'] for r in v)
        ddl = sum(x - 1 for x in c.values() if x > 1)
        if ddl:
            dup_lid += ddl; lid_pages.append((p, ddl))
    print(f'  S3 親が行として存在しない層: {dang} 層 / {len(dang_pages)} ページ')
    print(f'  S4 root(parent_id 空)の数: 合計 {roots} / {npg} ページ  '
          f'1つでないページ {len(root_pages)} {root_pages[:4]}')
    print(f'  S5 同じ親で order_in_parent が重複: {dup_ord} 層 / '
          f'{len(ord_pages)} ページ {ord_pages[:4]}')
    print(f'  S7 ページ内 layer_id の重複: {dup_lid} 層 / {len(lid_pages)} ページ '
          f'{lid_pages[:4]}')
    tb = [clip_bases(v) for v in bypage.values()]
    orph = sum(x[2] for x in tb)
    orph_pages = sum(1 for x in tb if x[2])
    print(f'  S6 ベース不明のクリップ層: {orph} 層 / {orph_pages} ページ  '
          f'(クリップ層 {sum(x[1] for x in tb)}、ベース {sum(len(x[0]) for x in tb)})')
    for col, want in (('visibility', '0'), ('has_mask', 'True'),
                      ('has_gradation_fill', 'True')):
        c = collections.Counter(r[col] for r in rows)
        print(f'  S8 {col:20s} {dict(sorted(c.items(), key=lambda kv: -kv[1]))}')
    op = collections.Counter(r['opacity'] for r in rows)
    print(f'  S8 opacity != 256 の層: '
          f'{sum(v for k, v in op.items() if k not in ("256", ""))} / {n}'
          f'  (空 {op.get("", 0)})')
    cv = collections.Counter(
        len({r['canvas'] for r in v}) for v in bypage.values())
    print(f'  S9 canvas の異なり数ごとのページ数: {dict(cv)}  '
          f'rendered: {dict(collections.Counter(r["rendered"] for r in rows))}')

    print('\n== N 命名 ==')
    def weird(s):
        return any(unicodedata.category(ch)[0] == 'C' for ch in s)
    empty = [r for r in rows if r['name'] == '']
    ctrl = [r for r in rows if weird(r['name'])]
    sp = [r for r in rows if r['name'] != r['name'].strip()]
    long_ = [r for r in rows if len(r['name']) > 60]
    print(f'  N1 名前が空 {len(empty)}(うちフォルダー '
          f'{sum(1 for r in empty if r["folder"] == "True")})  '
          f'制御文字 {len(ctrl)}  前後に空白 {len(sp)}  60字超 {len(long_)}')
    dupn = sum(1 for v in bypage.values()
               for k, c in collections.Counter(r['name'] for r in v).items()
               if c > 1 for _ in range(c - 1))
    print(f'  N2 ページ内の同名層(重複分): {dupn} / {n}')
    voc = collections.Counter(r['name'] for r in rows if r['name'])
    print(f'  N3 異なり名 {len(voc)}  上位: '
          f'{[(k, v) for k, v in voc.most_common(10)]}')
    pp = collections.Counter(r['page_path'] for r in rows)
    base = collections.Counter(Path(p.replace("\\", "/")).name
                              for p in bypage)
    print(f'  N4 page_path の異なり {len(pp)} / 一覧 {npg}  '
          f'同じファイル名が複数の場所に: '
          f'{sum(1 for k, v in base.items() if v > 1)} 名 '
          f'{[kv for kv in base.most_common(3) if kv[1] > 1]}')

    # N5 層のメタデータ指紋によるページ重複候補
    def sig(r):
        return hashlib.md5('|'.join((
            r['name'], r['layer_type'], r['drawn_ratio'], r['sat_mean'],
            r['top_colour_share'], r['colours_q16'])).encode()).hexdigest()[:16]
    for label, thr in (('全描画層', 0.0), ('drawn_ratio>=1e-4 の層', 1e-4)):
        inv = collections.defaultdict(set)
        per = collections.defaultdict(set)
        for r in drawn:
            if num(r['drawn_ratio'], 0) < thr:
                continue
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
        for (a, bb), c in shared.items():
            u = len(per[a]) + len(per[bb]) - c
            if u:
                pairs.append((c / u, c, a, bb))
        pairs.sort(reverse=True)
        hi = [p for p in pairs if p[0] >= 0.5]
        mid = [p for p in pairs if 0.2 <= p[0] < 0.5]
        print(f'  N5 {label}: 指紋を共有するページ対 Jaccard>=0.5 **{len(hi)}** / '
              f'0.2〜0.5 {len(mid)}  (比較した対 {len(pairs)})')
        for j, c, a, bb in hi[:8]:
            wa = a.replace('\\', '/').split('/')[2:4]
            wb = bb.replace('\\', '/').split('/')[2:4]
            print(f'      J={j:.3f} 共有{c:>3}  {"/".join(wa)} … '
                  f'{Path(a.replace(chr(92),"/")).name}  <->  {"/".join(wb)} … '
                  f'{Path(bb.replace(chr(92),"/")).name}'
                  f'{"   **作品を跨ぐ**" if wa != wb else ""}')

    print('\n== Q 統計量 ==')
    dr = [num(r['drawn_ratio']) for r in drawn]
    bad_dr = [r for r, x in zip(drawn, dr) if x is None or x < 0 or x > 1]
    print(f'  Q1 drawn_ratio: 範囲外・読めない {len(bad_dr)} / {len(drawn)}  '
          f'min {min(x for x in dr if x is not None):.2e} '
          f'max {max(x for x in dr if x is not None)}')
    rel, zero_sum, nocanvas = [], 0, 0
    for r in drawn:
        cv = r['canvas'].strip('[]').split(',')
        try:
            area = int(cv[0]) * int(cv[1])
        except (ValueError, IndexError):
            nocanvas += 1
            continue
        s = sum(int(r[c]) for c in SAT if r[c] not in ('', None))
        d = num(r['drawn_ratio'], 0)
        if s == 0:
            zero_sum += 1
            continue
        if d > 0:
            rel.append((abs(s - d * area) / (d * area), r))
    rel.sort(key=lambda t: -t[0])
    over = [t for t in rel if t[0] > 0.05]
    print(f'  Q2 sat_hist の和 vs drawn_ratio×面積: 相対誤差 '
          f'中央 {rel[len(rel)//2][0]:.2e} p90 {rel[len(rel)//10][0]:.2e} '
          f'max {rel[0][0]:.3f}   **5%超 {len(over)} / {len(rel)}**')
    for e, r in over[:6]:
        print(f'      誤差 {e:.3f}  dr={r["drawn_ratio"]} '
              f'{Path(r["page_path"].replace(chr(92),"/")).name} L{r["layer_id"]} '
              f'{r["name"][:20]} grad={r["has_gradation_fill"]}')
    print(f'      (和が0の描画層 {zero_sum}、canvas が読めない行 {nocanvas})')
    for col, lo, hi2 in (('top_colour_share', 0.0, 1.0),
                         ('sat_mean', 0.0, 255.0), ('colours_q16', 1, 4096)):
        vs = [num(r[col]) for r in drawn if r[col] != '']
        oob = [v for v in vs if v is not None and (v < lo or v > hi2)]
        print(f'  Q3 {col:18s} n={len(vs)} 範囲外 {len(oob)}  '
              f'min {min(vs):.4g} max {max(vs):.4g}')
    f2 = [r for r in drawn if r['formats'] == '2']
    f2n0 = [r for r in f2 if num(r['sat_mean'], 0) != 0]
    print(f'  Q4 形式2 の層 {len(f2)}  うち sat_mean != 0: **{len(f2n0)}** '
          f'({1 - len(f2n0)/len(f2):.4f} が 0)')
    print('  Q5 列ごとの空率(描画層のみ、空が1%以上の列):')
    for c in rows[0]:
        e = sum(1 for r in drawn if r[c] == '')
        if e / len(drawn) >= 0.01:
            print(f'      {c:22s} {e/len(drawn):.3f}')
    seen = collections.Counter(tuple(r.values()) for r in rows)
    print(f'  Q6 完全重複行: {sum(c-1 for c in seen.values() if c > 1)} / {n}')

    with open(OUT / 'summary.txt', 'w', encoding='utf-8') as fh:
        fh.write(f'rows {n} pages {npg} drawn {len(drawn)}\n')
    print(f'\n工程(作者 / 抽出 / こちらの門 / 不明)の割り当てはこの器具ではしない。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
