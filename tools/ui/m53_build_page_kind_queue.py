# -*- coding: utf-8 -*-
"""M53: **「この頁は何ですか」の行列**(41頁 + 既知セル4件)。

ユーザーの裁定(2026-10-26、基盤経由): **「41ページもこちらが見て判断する。
UI を作ってそれに回答する方式で track M で実装・立ち上げる」/ ファイルは「残す(いま消さない)」**。

骨子 `doc/ponytail_skeleton.md`(基盤、`be53f47`)の最初の適用。

**走らせる前に登録する(以下はすべて走行前に書いた)**

母集団:
  `results/m38_rough_sieve/rough_candidates_negative.csv` の 50行
    = **仕上がった絵が無い頁の候補**(追補28 §A で名前を直した篩)
  **描画比の和 == 0.000000 の9行 = 作者が既に答えた群**(追補28 の「N の9件」。
    **ただし札は弱い** —— 「上から数件で結構です」に対する運用の説明で、**個別判定ではない**)
  **残り41行 = 未回答。これを出す。**

絵: **納品のプレビュー(頁ぜんたい)**。`data/incoming_manuscript/colour_request4_rest/`。
  **原寸は無い**(この50頁はアルファを7頁12層ぶんしか請求していない)。
  **縮小するので `scale` に実際の倍率を入れる**(規則4: 既定値を主張にしない)。
  倍率 = (1/納品の縮小率) x (窓に収めた比) —— **元の画布に対する比**。

門(**どれが当たり、どれが当たらないかを理由つきで書く。黙って呼ばないのが T110**):
  **門(0) 窓が単色 —— `m27.flat` は当たらない。理由を先に書く:**
    `m27.flat` は「share >= 0.05 の色が2つ未満」で、**窓が対象で埋まっている前提**の門である。
    **頁ぜんたいは大半が紙**なので、この量は「読めるか」ではなく**「頁が疎か」**を測る。
    実際に当てたら **41頁中36頁と既知セル4件中3件が落ちた**(走行 `--dry`。追補40 §C)。
    **既知セルが門の誤りを捕まえた** —— 落ちた3件は作者が答えた頁そのものである。
    → **量を言い直す: 「最多色でない画素の割合」が 0 なら落とす**(= 何も描かれていない)。
      **閾値は 0。意味から決めた値で、出方を見て動かす余地が無い。**
      **測った結果: 49頁すべてで 0 ではない**(最小 0.001144 / 中央 0.062882 / 最大 0.463585)。
      → **門(0) はこの母集団で1件も落とさない。** 落とすべき頁が無いので落とさない
  **門(1) 輪郭が窓の10%超 —— 当たらない。** この UI は成分の輪郭を描かない(マゼンタが無い)
  **門(2) 窓の内側にマゼンタ無し —— 当たらない。** 同上
  **門(3) プレビューの実体が無い / 納品の一覧に無い —— 当たる。** 絵が出せないものは出さない

**絵の出し方(門の代わりにこちらを直した)**:
  疎な頁は「落とす」のではなく**「寄る」**。**墨の囲み箱を主の絵にする。**
    `crop` = **頁ぜんたい**(**既定。問いが「この頁は何ですか」なので、既定は頁でなければならない**)
    `ctx`  = **墨の塊に寄った絵**(`C` で切り替え)。疎な頁のための補助
    **寄った絵を既定にしない** —— 断片を見て頁を判定されると、問いと答がずれる(規則5 の族)
  **拡大は元の画布の解像度を超えない**(上限 = 納品の縮小率)。
    **超えて出せば、無い解像度を主張することになる**(規則4 / 縫い目2)。
  **`scale` / `ctx_scale` には元の画布に対する真の比を入れる。**
    **プレビューは既に 1/縮小率 なので、拡大したぶんは補間でぼける** ——
    **その旨を画面に出す**(`m54` が `info()` を差し替える)。

既知セル(規律2)は**2種類ある。札の強さが違う**:

  **強い既知セル(2件): `koa`。41件の中に既に入っている。**
    作者は **5つのフルパスを名指しで「全部下絵」**と答えた(追補25・追補26 §C)= **決定**。
    そのうち **`koa\koakuma.clip` と `koa\koakuma_check.clip` が、この41件に残っている**
    (この CSV は `koa` の裁定より前の時点の写し)。
    **読み(走る前に登録)**: **2件とも `下絵だけ` が返るはず。**
    **外れたら器具を疑う** —— 札が決定なので、外れを「札が弱かった」で逃がせない。
    **この2件は41件から抜かない**(抜けば母集団が変わる。聞くのに14秒しかかからない)。

  **弱い既知セル(4件): 回答済み9件から、4つの作品から各1件**
  (`gran` / `kacho` / `illust20172` / `toramusume2`)。**画面に印は出さない。**
  **読み(走る前に登録)**:
    4件とも `資料用の画像` → **器具は通った**(ただし強い既知セルの方が証拠として重い)
    1〜2件が外れる → **器具の欠陥とは言えない。** 9件の札は個別判定ではないので、
      **外れは「9件の中に別の種類が混ざっていた」とも読める**(追補28 の札の弱さ)
    **4件とも外れる** → **器具を疑う**(絵・問い・選択肢のどれかが働いていない)
  **どちらに転んでも、41件の答は読む**(既知セルは41件の答を捨てる門ではない)

位置: 既知セルは **order 2 / 13 / 26 / 38** に固定、残りは SEED の並べ替え。
  **走行ログにだけ印字する**(作者は見ない)。

選択肢(作者自身の言葉から。**帳簿の語彙を出さない**。規則5):
  1 資料用の画像(絵を描くための参考に入れたもの)
  2 下絵だけ
  3 仕上がった絵
  T その他(一言)   <- **出口。必須**(規則6)
  S わからない
  **`T` と `S` の使用率を後で印字する** —— 高ければ選択肢が間違っている

パス: **画面に出す**(基盤の見立て: 作者は自分のファイルを認識して答えるので精度が上がる)。
  **危険として明記**: **フォルダー名が答を誘導しうる**(`illust20172` → 「イラスト」)。
  **読むときの但し書きにする。**

時間(規則9: 先に宣言し、後で実測を出す): **45問 x 1問 7秒 = 5〜6分**と宣言する。
  (m47 の実測が1問あたり中央 7.1秒。**超過は欠陥として記録する**)

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'colour'))
import m27_author_crops as m27                            # noqa: E402

CSV = Path('results/m38_rough_sieve/rough_candidates_negative.csv')
REST = Path('/home/sh1/deepl/lineart/data/incoming_manuscript/colour_request4_rest')
OUT = Path('results/m53_ui'); (OUT / 'crops').mkdir(parents=True, exist_ok=True)
SEED = 20261026
WIN = 1400
KNOWN_AT = [2, 13, 26, 38]
WORDS = ['資料用の画像(絵を描くための参考に入れたもの)', '下絵だけ', '仕上がった絵']
OTHERS = [['free', 'その他(一言)', 'T'], ['unknown', 'わからない', 'S']]
ASK = 'この頁は何ですか。'
HELP = ('絵ぜんたいを縮小して出しています(倍率は上の行に出ます)。'
        '当てはまる言葉が無ければ T に一言、迷ったら S。'
        'B で前へ、N で次へ戻れます。')
EXPECT = {'候補': 50, '回答済み': 9, '未回答': 41}


def work_of(src):
    """作品の名札。**結合には使わない**(結合は `source` のみ)。"""
    parts = src.replace('/', '\\').split('\\')
    return parts[3] if len(parts) > 4 else parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true', help='絵を書かずに数だけ出す')
    a = ap.parse_args()

    rows = list(csv.DictReader(open(CSV, encoding='utf-8')))
    answered = [r for r in rows if float(r['テンプレート外の描画比の和']) == 0.0]
    rest = [r for r in rows if float(r['テンプレート外の描画比の和']) > 0.0]
    got = {'候補': len(rows), '回答済み': len(answered), '未回答': len(rest)}
    print('== 既知セル(母集団)==')
    for k, v in EXPECT.items():
        print(f'  {k:8s} {got[k]:>4} / 期待 {v:>4}  '
              f'{"PASS" if got[k] == v else "**FAIL**"}')
    if got != EXPECT:
        print('**母集団が合わない。行列を書かない。**')
        return 1

    pages = {p['source']: p for p in
             (json.loads(l) for l in
              open(REST / 'colour_layers_pages.jsonl', encoding='utf-8')
              if l.strip())}

    drop = collections.Counter()
    dropped = []

    def prep(r, why):
        """1件を絵にする。**門を通らなければ None**(理由を数える)。"""
        src = r['page_path']
        p = pages.get(src)
        if p is None:
            drop['**納品の一覧に無い(こちらの2つの一覧で綴りが違う)**'] += 1
            dropped.append((src, 'not_in_manifest')); return None
        f = REST / p['preview']['file']
        if not f.exists():
            drop['門(3) プレビューの実体が無い'] += 1
            dropped.append((src, 'no_preview_file')); return None
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            drop['門(3) プレビューが読めない'] += 1
            dropped.append((src, 'unreadable')); return None
        h, w = img.shape[:2]
        # 墨 = 最多色でない画素(量子化は m27 と同じ 32階調)
        qq = (img // 32).astype(np.int32)
        key = qq[..., 0] * 1024 + qq[..., 1] * 32 + qq[..., 2]
        u, c = np.unique(key, return_counts=True)
        dom = u[c.argmax()]
        ink = key != dom
        share = float(ink.mean())
        if share == 0.0:
            drop['門(0) 最多色でない画素が1つも無い'] += 1
            dropped.append((src, 'no_ink')); return None
        # **囲み箱を墨の最小・最大で取ると、トンボ(頁の四隅と中央の線)で
        # 頁ぜんたいに広がる**(追補40 §D の `--dry` で見つけた: `ntrlaw page0024` は
        # 墨 0.1% なのに箱が 1635x1522 = ほぼ頁ぜんたいになった)。
        # → **墨を 1% ふくらませて塊に分け、墨の画素がいちばん多い塊を取る。**
        # r は「同じ絵か」の意味から決めた値で、出方を見て動かさない。
        # **塊が頁の墨の何割を持つかを印字する**(寄る先を間違えた危険が見えるように)。
        r = max(1, round(0.01 * max(h, w)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        grown = cv2.dilate(ink.astype(np.uint8), k)
        nlab, lab = cv2.connectedComponents(grown, connectivity=8)
        best, best_n = 0, -1
        for i in range(1, nlab):
            cnt = int((ink & (lab == i)).sum())
            if cnt > best_n:
                best, best_n = i, cnt
        sel = ink & (lab == best)
        clus = best_n / max(1, int(ink.sum()))
        ys, xs = np.nonzero(sel)
        pad = max(1, round(0.02 * max(h, w)))
        y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
        x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
        box = img[y0:y1, x0:x1]
        bh, bw = box.shape[:2]
        red = float(p['preview']['縮小率'])
        # **拡大は元の画布の解像度まで**(u <= 縮小率)。超えれば無い解像度の主張になる
        uz = min(WIN / max(bh, bw), red)
        crop = cv2.resize(box, (max(1, round(bw * uz)), max(1, round(bh * uz))),
                          interpolation=(cv2.INTER_AREA if uz < 1
                                         else cv2.INTER_LANCZOS4))
        fit = min(1.0, WIN / max(h, w))
        ctx = (img if fit == 1.0 else
               cv2.resize(img, (max(1, round(w * fit)), max(1, round(h * fit))),
                          interpolation=cv2.INTER_AREA))
        return dict(row=r, page=p, crop=crop, ctx=ctx,
                    scale=round(uz / red, 5),      # 寄った絵(= ctx 側に回す)
                    ctx_scale=round(fit / red, 5),  # 頁ぜんたい(= crop 側に回す)
                    interp=round(max(1.0, uz), 3), ink=round(share, 6),
                    clus=round(clus, 4),
                    box_px=[bh, bw], why=why, src=src, canvas=p['canvas'])

    print()
    print('== 絵を作る(門(0) と 門(3) だけが当たる。'
          '門(1)(2) はマゼンタを描かないので**当たらない**)==')
    ask_items, known_items = [], []
    for r in rest:
        x = prep(r, 'ask')
        if x: ask_items.append(x)
    # 既知セル: 4つの作品から各1件(作品名の昇順で先頭。**無作為にしない = 再現する**)
    bywork = collections.defaultdict(list)
    for r in answered:
        bywork[work_of(r['page_path'])].append(r)
    for wk in sorted(bywork):
        x = prep(sorted(bywork[wk], key=lambda r: r['page_path'])[0], 'known')
        if x: known_items.append(x)

    print(f'  未回答から絵になった  {len(ask_items):>3} / {len(rest)}')
    print(f'  既知セル(4作品から各1) {len(known_items):>3} / 4  '
          f'作品 {sorted(bywork)}')
    print()
    print('== 出さなかった問い(理由別。規則10)==')
    if not drop:
        print('  なし')
    for k, v in drop.most_common():
        print(f'  {k:52s} {v:>3}')
    for src, why in dropped:
        print(f'    **落とした**: {src}  ({why})')

    # 位置: 既知セルを固定の order へ、残りは SEED の並べ替え
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(ask_items))
    shuffled = [ask_items[i] for i in order]
    n_all = len(shuffled) + len(known_items)
    slots = [s for s in KNOWN_AT if s < n_all][:len(known_items)]
    seq = [None] * n_all
    for s, x in zip(slots, known_items):
        seq[s] = x
    it = iter(shuffled + known_items[len(slots):])
    for i in range(n_all):
        if seq[i] is None:
            seq[i] = next(it)

    q = []
    for i, x in enumerate(seq):
        base = f'p{i:03d}_{x["page"]["page"]}'
        if not a.dry:
            cv2.imwrite(str(OUT / 'crops' / f'{base}_page.png'), x['ctx'])
            cv2.imwrite(str(OUT / 'crops' / f'{base}_ink.png'), x['crop'])
        q.append({'order': i, 'crop': f'{base}_page.png',
                  'ctx': f'{base}_ink.png',
                  'work': x['page']['page'], 'pid': x['src'],
                  'scale': x['ctx_scale'], 'ctx_scale': x['scale'],
                  'interp': x['interp'], 'ink': x['ink'], 'box_px': x['box_px'],
                  'clus': x['clus'],
                  'why': x['why'], 'cands': [], 'word_share': {},
                  'top_word': '', 'purity': 0.0, 'n_lay': 0, 'n_lay90': 0,
                  'auto': 0, 'comp': 0, 'region': 0, 'lid': 0, 'area': None,
                  'canvas': x['canvas']})

    print()
    print('== 絵の読みやすさ(**門ではない。読むときの但し書きにする**)==')
    print('  %-3s %-9s %-7s %-11s %-13s %-9s %s'
          % ('問', '墨の割合', '塊の占有', '囲み箱px', '表示倍率(画布比)', '補間', '頁'))
    for it, x in zip(q, seq):
        print('  %-3d %9.6f %7.1f%% %5dx%-5d %13.4f %8.2fx %s'
              % (it['order'], it['ink'], 100 * it['clus'], it['box_px'][0],
                 it['box_px'][1], it['scale'], it['interp'], it['work']))
    low = [it for it in q if it['clus'] < 0.5]
    print(f'  **塊が頁の墨の半分未満の問 {len(low)} / {len(q)}** '
          f'(寄る先を間違えた危険。`C` で頁ぜんたいが出る)')
    for it in low:
        print(f'    問{it["order"]:<3} 占有 {it["clus"]:.1%}  {it["work"]}')
    nint = sum(1 for it in q if it['interp'] > 1.0)
    print(f'  **補間して拡大した問 {nint} / {len(q)}** '
          f'(プレビューの画素より細かくは出せない。画面にその旨を出す)')

    print()
    print('== 既知セルの位置(**作者には見せない**)==')
    print('  order ' + ' / '.join(str(i) for i, x in enumerate(seq)
                                  if x['why'] == 'known'))
    print()
    print('== 行列 ==')
    print(f'  問 {n_all}  (未回答 {len(shuffled)} + 既知セル {len(known_items)})')
    print(f'  **宣言する時間: 1問 7秒 x {n_all} = {n_all * 7 / 60:.1f}分**'
          f'(m47 の実測 中央 7.1秒 から。**超過は欠陥**)')
    sc = [it['scale'] for it in q]
    print(f'  表示倍率(元の画布に対する比): 最小 {min(sc):.4f} / '
          f'中央 {sorted(sc)[len(sc)//2]:.4f} / 最大 {max(sc):.4f}  '
          f'**すべて `scale` に入れた**')
    if a.dry:
        print('\n(--dry: 絵も行列も書いていない)')
        return 0

    meta = dict(n=n_all, n_ask=n_all, n_check=0, n_auto=0, n_all=n_all,
                words=list(WORDS), others=[list(o) for o in OTHERS],
                ask=ASK, help=HELP, queue=q)
    (OUT / 'queue.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwrote {OUT / "queue.json"}  ({n_all} 問)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
