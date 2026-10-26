# -*- coding: utf-8 -*-
"""M56: **「この層は何ですか」の行列**(44頁 / 307層)。**単位は層。頁ではない。**

ユーザー(作者)の指示(追補41。**時計を読んでいないので時刻を書かない**):
  **「複数のレイヤが合成されたものが表示されているようだが、それぞれをこたえるべきなのか、
  どれか一つのレイヤをこたえてほしいのかわからない。もし合成されたものなら、
  左端に合成前の各レイヤを並べて、どれについて答えてほしいかを問うてほしい」**
時間の承認(追補42): **「３６分でOKだ　やる」** → **307層すべてを聞く。絞らない。**

**走らせる前に登録する(以下はすべて走行前に書いた)**

母集団: `results/m55_request5/request5_layer_list.csv` の 307行
  = 44頁の **`rendered` 真 / `drawn_ratio` > 0 / `visibility` = 1** の層。
  **フォルダー(423)と未描画の層は入っていない。**
  44頁 = 追補40 の行列(未回答40 + 既知セル4)。**母集団は変えない。**

絵(**作者の指示どおり**):
  **左端 = その頁の合成前の各層を z 順に並べた帯。問われている層に枠。**
  `crop`(既定) = **帯 + 問われている層を大きく**
  `ctx`(`C`)   = **帯 + 頁の合成プレビューを大きく**
  **帯は両方の絵で同じ位置・同じ並び**(規則8: 位置を固定する。覚えた指が使える)
  **透明は市松模様で出す**(追補43 §B で直した。**白で出していたら、100% 純白の層4件が
  消えて、作者が「白紙」と答えた** —— 消していたのはこちらだった)。
  **白いインクの割合を印字する**(門ではない。読むときの但し書き)。

問い: **「この層は何ですか」**(層を名指しする。**単位を画面に書く** = 追補41 §F)

選択肢(**すべて作者自身の言葉から**。帳簿の語彙は使わない。規則5):
  1 下絵・ラフ          <- 作者「下絵」「ラフ」
  2 線画                <- 作者「線画」
  3 色・ベタ(塗り)
  4 仕上がった絵(統合済み) <- 作者「完成済みカラーイラストの合成レイヤ」
  5 資料用の画像         <- 作者「資料」(追補28 の問い N)
  6 セリフ・文字         <- 作者「セリフ」
  T その他(一言)        <- **出口。必須**(規則6)
  S わからない
  **`T`/`S` の使用率を印字する。** **m53 では 3/3 = 100% で、選択肢ではなく
  問いの単位が間違っていた** —— **今回 100% なら、単位がまだ違う。**

層の名前を画面に出すか: **出す**(作者自身の札であり、自分のファイルの中で層を同定するのに要る)。
  **危険として登録**: **名前が答を誘導しうる**(`Sketch` → 「下絵」)。
  **パスと同じ扱い**(追補40 §G1)。**読むときの但し書きにする。**

門(どれが当たり、どれが当たらないかを理由つきで書く。T110):
  **門(0') 層の絵が空 —— 当たる。** 書き出された PNG に**不透明な画素が1つも無い**なら落とす。
    `drawn_ratio` は台帳の値で、**書き出しの結果とは別**(T-M3 の族: 別の量)
  **門(1)(2)(マゼンタ) —— 当たらない。** 成分の輪郭を描かない
  **門(0) `m27.flat` —— 当たらない。** 追補40 §C で量が問いに対応していないと示した
  **門(3) 層の PNG が納品に無い —— 当たる。** 絵が出せないものは出さない

既知セル(規律2)。**今日の3件の答が、そのまま層の単位の門になる**:
  **撤回したのは問いであって答ではない**(追補41 §C)。作者は頁について次を述べた:

    `4th page0002`       「**下絵　最下段だけ線画がはいってる**」
      -> その頁の層の答に **`下絵・ラフ` が在り、`線画` が少なくとも1つ在る**はず
    `reitaisai page0016` 「**資料　ラフ　セリフ**」
      -> **`資料用の画像` と `下絵・ラフ` と `セリフ・文字` が在る**はず
    `gran page0016`      「**完成済みカラーイラストの合成レイヤ**」
      -> **`仕上がった絵(統合済み)` が少なくとも1つ在る**はず
    `koa\\koakuma` `koa\\koakuma_check`(作者が5つのフルパスを名指しで「全部下絵」= 決定)
      -> **`仕上がった絵` と答えられた層が1つも無い**はず

  **これは「層ごとの期待値」ではなく「頁の層の集合への期待値」である。**
  **1つの層の答だけでは判定できない。** 頁の層が全部返ってから読む。
  **外れたときの読み(走る前に登録)**:
    `gran` / `4th` / `reitaisai` の3件は**今日その言葉で答えたばかり**なので、
      **外れたら器具を疑う**(選択肢が作者の言葉を写せていない、など)
    `koa` の2件は**頁について「全部下絵」**なので、**層の1つが `線画` でも矛盾しない**
      (下絵の工程に線が含まれうる)。**`仕上がった絵` だけが矛盾する。**
  **既知セルは残りの答を捨てる門ではない。**

順序: **頁ごとにまとめる**(頁を見るのは1回で済む)。頁の順は SEED の並べ替え、
  **頁の中は z 順**。**既知セルの頁も混ぜる。画面に印は出さない。**

時間(規則9): **307問。1問 7秒なら 35.8分だが、頁ごとにまとめるので短くなるはず。**
  **宣言: 36分。実測を出す。超過は欠陥。**

出さなかった問いは理由別に数える(規則10)。
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REQ = Path('results/m55_request5/request5_layer_list.csv')
OUT = Path('results/m56_ui'); (OUT / 'crops').mkdir(parents=True, exist_ok=True)
SEED = 20261026
WIN = 1400              # 絵ぜんたいの長辺
STRIP = 190             # 左端の帯の幅
CELL_MAX = 240          # 帯の1枚の高さの上限
MAXUP = 8               # 主の絵の拡大の上限(超えると何も足さない)
PAD = 6
WORDS = ['下絵・ラフ', '線画', '色・ベタ(塗り)', '仕上がった絵(統合済み)',
         '資料用の画像', 'セリフ・文字']
OTHERS = [['free', 'その他(一言)', 'T'], ['unknown', 'わからない', 'S']]
ASK = 'この層は何ですか。'
HELP = ('**左端がこの頁の層を重なり順に並べたもので、いま聞いている層に枠が付いています。**'
        '透明なところは白で出しています(紙と同じ)。'
        'C で頁ぜんたいの絵に切り替わります(左端の並びは動きません)。'
        '当てはまる言葉が無ければ T に一言、迷ったら S。B で前へ、N で次へ。')
EXPECT_PAGES = 45
EXPECT_LAYERS = 315


def checker(h, w, cell=14, a=242, b=214):
    """**透明の下地。白のインクを白で消さないため。**

    追補43 §B: 透明を**白**で出していたところ、作者が 4件に
    **「白紙」**と答えた。**その4層は 100% 純白のインク**(不透明 46,956 と
    166,816 画素、BGR 全画素 255、アルファ 255)だった ——
    **作者は見せられた絵を正しく読んでいた。消していたのはこちらである。**
    **規則4(既定値を主張にしない)の最も痛い形**: 白い下地は「何も無い」という主張を、
    何も測らずに出していた。
    """
    y, x = np.mgrid[0:h, 0:w]
    m = (((y // cell) + (x // cell)) % 2).astype(bool)
    out = np.full((h, w, 3), a, np.uint8)
    out[m] = b
    return out


def over_checker(img):
    """RGBA なら市松模様に重ねて BGR にする。BGR はそのまま返す。"""
    if img.ndim == 3 and img.shape[2] == 4:
        al = img[..., 3:4].astype(np.float32) / 255.0
        bgc = checker(img.shape[0], img.shape[1]).astype(np.float32)
        return (img[..., :3].astype(np.float32) * al
                + bgc * (1.0 - al)).round().astype(np.uint8)
    return img


def flatten(path, bg=255):
    """アルファ付き PNG を**墨の囲み箱で切って RGBA のまま**返す。

    返すのは `(RGBA の絵, 不透明な画素数, 元の画布に対する比, 囲み箱)`。
    **下地に重ねるのは最終の大きさになってから**(`over_checker`)——
    先に重ねると、縮小で市松模様がつぶれる。

    **切らないと帯も主の絵もほぼ白になる**(追補42 §C-2 の組版試験で確認:
    22層の頁で、どの札も中身が見えず、主の絵も数本の線だけだった)。
    **m53 で頁に対して直したのと同じことを、層に対しても直す。**

    **代価(登録しておく)**: **札ごとに切る倍率が違うので、札の間で絵の大きさを比べられない。**
    層の大小は札では分からない —— **主の絵の倍率(画面に出る)と `C`(頁ぜんたい)で見る。**

    **層の PNG は原寸**(例 7016x6543)で、**プレビューは 1/縮小率** ——
    **同じ式で倍率を出すと嘘になる**(追補42 §C-1 で踏んだ)。**画像ごとに比を持たせる。**
    **窓より大きい絵は切った直後に縮める** —— 22層を原寸で持つと 4GB になる。
    """
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None, 0, 1.0, None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        a = im[..., 3]
        opaque = int((a > 0).sum())
        if opaque == 0:
            return None, 0, 1.0, None
        ys, xs = np.nonzero(a > 0)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pad = max(1, round(0.01 * max(y1 - y0, x1 - x0)))
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        y1, x1 = min(im.shape[0], y1 + pad), min(im.shape[1], x1 + pad)
        out = im[y0:y1, x0:x1].copy()        # **RGBA のまま返す**
        box = [int(y1 - y0), int(x1 - x0)]
    else:
        out = im[..., :3]
        opaque = int(im.shape[0] * im.shape[1])
        box = [int(im.shape[0]), int(im.shape[1])]
    del im
    h, w = out.shape[:2]
    r = min(1.0, WIN / max(h, w))
    if r < 1.0:
        out = cv2.resize(out, (max(1, round(w * r)), max(1, round(h * r))),
                         interpolation=cv2.INTER_AREA)
    return out, opaque, r, box


def fit(img, box):
    """`box=(h, w)` に収める(拡大はしない)。余白は白。"""
    bh, bw = box
    h, w = img.shape[:2]
    s = min(bh / h, bw / w, 1.0)
    im = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                    interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    im = over_checker(im)
    canvas = np.full((bh, bw, 3), 255, np.uint8)
    y, x = (bh - im.shape[0]) // 2, (bw - im.shape[1]) // 2
    canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
    return canvas, round(s, 5)


def compose(strip_imgs, main, mark):
    """返すのは `(絵, 主の絵の縮小比)`。**元の画布に対する比は呼ぶ側が掛ける。**"""
    """左端の帯 + 主の絵。**帯は両方の絵で同じ並び・同じ位置。**"""
    # **1枚あたりの高さに上限を置く。** 層が2枚しかない頁で窓を等分すると、
    # 1枚が 700px の枠に小さな絵が浮く形になる(追補42 §C-2 の目視で見つけた)。
    # **頁の枠(位置の対応)は残す** —— 中身は主の絵で見る
    n = len(strip_imgs)
    cell = min(CELL_MAX, max(40, (WIN - PAD * (n + 1)) // max(1, n)))
    tiles = []
    for i, im in enumerate(strip_imgs):
        t, _ = fit(im, (cell, STRIP - 2 * PAD))
        t = cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                               value=(255, 0, 255) if i == mark else (200, 200, 200))
        tiles.append(t)
    H = sum(t.shape[0] for t in tiles) + PAD * (n + 1)
    H = max(H, WIN)
    left = np.full((H, STRIP, 3), 255, np.uint8)
    y = PAD
    for t in tiles:
        left[y:y + t.shape[0], PAD:PAD + t.shape[1]] = t
        y += t.shape[0] + PAD
    # **小さい層は拡大する。拡大したことは `interp` で画面に出す。**
    # 13:25 の目視: ノンブル(箱 38x15)を 1:1 で出すと、組んだ絵が
    # **1476x219 の細長い帯**になり、21枚の帯が 190px 幅に潰れた
    # (`max-width:100%` なので、ブラウザは 219px のまま出す)。
    # **拡大しないと版面が崩れ、拡大すると無い解像度を出す** ——
    # **拡大して、倍率を画面に書く**ほうを採る(規則4: 既定値を主張にしない。
    # **黙って原寸と書くのが禁じ手であって、拡大そのものではない**)。
    bh, bw = main.shape[:2]
    sc = min(H / bh, WIN / bw, MAXUP)
    big = over_checker(
        cv2.resize(main, (max(1, round(bw * sc)), max(1, round(bh * sc))),
                   interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_NEAREST))
    pane = np.full((H, WIN, 3), 255, np.uint8)
    y, x = (H - big.shape[0]) // 2, (WIN - big.shape[1]) // 2
    pane[y:y + big.shape[0], x:x + big.shape[1]] = big
    return np.hstack([left, pane]), round(sc, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deliv', default='/home/sh1/deepl/lineart/data/'
                    'incoming_manuscript/colour_request5',
                    help='依頼5 の納品')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--partial', action='store_true',
                    help='**層が揃っている頁だけで組む(通しの試験用)。本番では使わない**')
    a = ap.parse_args()

    # **行列の素は納品の manifest**(= 事前登録した規則の出力そのもの)。
    # **台帳との結合は `source` + `layer_id`。ファイル名では結合しない**(m48 の教訓)。
    deliv = Path(a.deliv)
    man = deliv / 'colour_layers_pages.jsonl'
    if not man.exists():
        print(f'**納品が無い: {man}。行列を書かない**')
        return 1
    dpages = [json.loads(l) for l in open(man, encoding='utf-8') if l.strip()]
    led = collections.defaultdict(dict)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'colour'))
    import m19_version_resolution as m19                   # noqa: E402
    for r in csv.DictReader(open(m19.CSV872, encoding='utf-8')):
        led[r['page_path']][str(r['layer_id'])] = r
    rows, paper = [], []
    for p in dpages:
        for l in p['layers']:
            r = led[p['source']].get(str(l['layer_id']))
            if r is None:
                print(f'  **台帳に無い層**: {p["source"]} L{l["layer_id"]}')
                continue
            row = dict(page_path=p['source'], layer_id=str(l['layer_id']),
                       name=(r.get('name') or ''), formats=(r.get('formats') or ''))
            # **用紙層を外す(ユーザーの裁定)**: `formats = paper` かつ
            # **全面・各ch 255 の純白** = **画布の紙であって、作者が描いた物ではない。**
            # **名前(「用紙」)では決めていない** —— **ファイル形式の構造の値で決めている**
            # (どの層がベースかを名前で決めない、という track の規律と同じ扱い)。
            # **外した一覧は印字し、CSV にも書く**(異議があれば戻せるように)。
            if row['formats'] == 'paper':
                paper.append(row); continue
            rows.append(row)
    print(f'== 既知セル(母集団)==')
    print(f'  納品の頁 {len(dpages)} / 期待 {EXPECT_PAGES}  '
          f'{"PASS" if len(dpages) == EXPECT_PAGES else "**FAIL**"}')
    print(f'  納品の層 {sum(len(p["layers"]) for p in dpages)} / 期待 {EXPECT_LAYERS}  '
          f'{"PASS" if sum(len(p["layers"]) for p in dpages) == EXPECT_LAYERS else "**FAIL**"}')
    if len(dpages) != EXPECT_PAGES or sum(len(p['layers']) for p in dpages) != EXPECT_LAYERS:
        print('**納品が期待と合わない。行列を書かない。**')
        return 1
    print(f'\n== 母集団の差分(理由別。規律9)==')
    print(f'  納品の層                         {EXPECT_LAYERS}')
    print(f'  **用紙層を外した(ユーザーの裁定)**    -{len(paper)}'
          f'  (`formats = paper` / 全面 / 各ch 255 の純白)')
    print(f'  **聞く層**                        **{len(rows)}**')
    with open(OUT / 'excluded_paper_layers.csv', 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['page_path', 'layer_id', 'name',
                                          'formats'])
        w.writeheader(); w.writerows(paper)
    print(f'  外した一覧: {OUT / "excluded_paper_layers.csv"}')

    # 絵の在り処: 依頼5 の木 -> 無ければ依頼4(7頁12層が在る)
    trees = [deliv,
             Path('/home/sh1/deepl/lineart/data/incoming_manuscript/'
                  'colour_request4_rest')]
    prev = {}
    for t in trees:
        f = t / 'colour_layers_pages.jsonl'
        if f.exists():
            for l in open(f, encoding='utf-8'):
                if l.strip():
                    p = json.loads(l)
                    prev.setdefault(p['source'], (t, p))

    def layer_file(src, lid):
        for t in trees:
            if not t.exists():
                continue
            tt, p = prev.get(src, (None, None))
            if p:
                for l in p['layers']:
                    if str(l['layer_id']) == str(lid) and (tt / l['file']).exists():
                        return tt / l['file']
        return None

    drop = collections.Counter()
    bypage = collections.defaultdict(list)
    for r in rows:
        bypage[r['page_path']].append(r)

    pages, skipped = [], []
    for src, rs in bypage.items():
        if src not in prev:
            drop['門(3) 頁が納品の一覧に無い'] += len(rs); skipped.append((src, 'no_page')); continue
        files = [(r, layer_file(src, r['layer_id'])) for r in rs]
        missing = [r for r, f in files if f is None]
        if missing:
            drop['門(3) 層の PNG が納品に無い'] += len(missing)
            if not a.partial:
                skipped.append((src, f'{len(missing)}/{len(rs)} 層が無い')); continue
        got = [(r, f) for r, f in files if f is not None]
        if not got:
            skipped.append((src, '層が1つも無い')); continue
        imgs, keep, ratios, boxes, whites = [], [], [], [], []
        for r, f in got:
            im, opaque, ratio, box = flatten(f)
            if im is None and opaque == 0:
                drop["門(0') 層の絵が空(不透明な画素が0)"] += 1; continue
            if im is None:
                drop['門(3) 層の PNG が読めない'] += 1; continue
            al = im[..., 3] > 0 if im.shape[2] == 4 else np.ones(im.shape[:2], bool)
            near_white = float((im[..., :3][al] >= 240).all(axis=1).mean()) if al.any() else 0.0
            imgs.append(im); keep.append(r); ratios.append(ratio); boxes.append(box)
            whites.append(round(near_white, 4))
        if not keep:
            skipped.append((src, 'すべて空')); continue
        tt, p = prev[src]
        comp = cv2.imread(str(tt / p['preview']['file']), cv2.IMREAD_COLOR)
        pages.append(dict(src=src, page=p['page'], rows=keep, imgs=imgs,
                          ratios=ratios, boxes=boxes, whites=whites, comp=comp,
                          red=float(p['preview']['縮小率'])))

    print(f'\n  絵が組める頁 {len(pages)} / {len(bypage)}  '
          f'層 {sum(len(x["rows"]) for x in pages)} / {len(rows)}')
    print('\n== 出さなかった問い(理由別。規則10)==')
    for k, v in drop.most_common():
        print(f'  {k:44s} {v:>4}')
    for src, why in skipped:
        print(f'    **出さない頁**: {src}  ({why})')

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pages))
    q = []
    for pi in order:
        x = pages[pi]
        for i, r in enumerate(x['rows']):
            n = len(q)
            base = f'l{n:03d}_{x["page"]}_L{r["layer_id"]}'
            im_l, s_l = compose(x['imgs'], x['imgs'][i], i)
            im_c, s_c = compose(x['imgs'], x['comp'], i)
            if not a.dry:
                cv2.imwrite(str(OUT / 'crops' / f'{base}_layer.png'), im_l)
                cv2.imwrite(str(OUT / 'crops' / f'{base}_page.png'), im_c)
            q.append({'order': n, 'crop': f'{base}_layer.png',
                      'ctx': f'{base}_page.png',
                      'work': f'{x["page"]}  層 {r["layer_id"]}'
                              f'「{r["name"] or "(名前なし)"}」'
                              f'  {i + 1}/{len(x["rows"])}',
                      'pid': x['src'],
                      # **層は原寸、プレビューは 1/縮小率。比は画像ごとに掛ける**
                      'scale': round(s_l * x['ratios'][i], 5),
                      'ctx_scale': round(s_c / x['red'], 5),
                      'interp': round(max(1.0, s_l), 3),
                      'box_px': x['boxes'][i],
                      'white': x['whites'][i],
                      'cands': [], 'word_share': {},
                      'top_word': '', 'purity': 0.0, 'n_lay': len(x['rows']),
                      'n_lay90': i + 1, 'auto': 0, 'comp': 0,
                      'region': int(r['layer_id']), 'lid': int(r['layer_id']),
                      'area': None, 'why': 'ask'})

    nw = [it for it in q if it['white'] >= 0.99]
    print(f'\n== ほぼ全部が白いインクの層 {len(nw)} / {len(q)}'
          f'(**市松模様の下地で出す。白地なら消える**)==')
    for it in nw:
        print(f'  問{it["order"]:<4} 白 {it["white"]:.1%}  {it["work"]}')
    print(f'\n== 行列 ==\n  問 {len(q)}  頁 {len(pages)}  '
          f'1頁あたり中央 {int(np.median([len(x["rows"]) for x in pages]))} 層')
    print(f'  **宣言する時間: 36分**(307問。頁ごとにまとめるので短くなるはず。'
          f'**超過は欠陥**)')
    if a.dry:
        print('\n(--dry: 絵も行列も書いていない)')
        return 0
    meta = dict(n=len(q), n_ask=len(q), n_check=0, n_auto=0, n_all=len(q),
                words=list(WORDS), others=[list(o) for o in OTHERS],
                ask=ASK, help=HELP, queue=q)
    (OUT / 'queue.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT / "queue.json"}  ({len(q)} 問)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
