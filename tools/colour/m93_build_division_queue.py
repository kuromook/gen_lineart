#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m93: **3案を原寸で描いて行列にする**(追補144)。器具②-a。

**登録の正本は `doc/prereg_author_division_20261030.md`(修正1・修正2 を含む)。**
窓は `m92` が選んだ `results/m92_windows/windows.csv` をそのまま読む。**選び直さない。**

**問いの単位を画面に書く**(条例 / `feedback_state_the_unit_of_the_question`):
  **「この窓の中の絵を『面』に分けるとしたら、A・B・C のどれがあなたの分け方に最も近いですか」**

3案(**どれが A/B/C になるかは問いごとに引き直す。NC2: 並びを固定するのは誤り**):
  **標準** = `standard_visible` の島(作者の単位。8頁で 692)
  **A0**   = 一様量子化の等色8連結(学習なし)
  **頁全体** = 窓ぜんたいを1面(= 内側に線を引かない)

**描き方の規律**:
- **原寸 1024x1024 のまま。縮小しない**(NC3: 縮小して見せるのは誤り)。
  **3案は別々の PNG** —— 横に並べると画面幅に収まらず、UI が縮小してしまう
- **境界は 1 画素のマゼンタ。3案すべて同じ描き方**(片方だけ太いと比較にならない)
- **A0 の境界は 4 画素の段のまま。滑らかにしない**(修正1-A)。
  preview の側で出た A0 を **`np.repeat` で原寸へ**(最近傍。補間しない)
- **下地は用紙の色**(`_preview.png` から読む)。**「透明は水色」は効かない** ——
  用紙が全画布を覆うので透明な画素が残らない。**水色が出た割合を測って印字する**
- **合成は `m27.composite` では書けない** —— あれは `clip` を見ないので、
  **クリップ層が全画布を塗り直す**(この木の7頁では見えている層の **48% が `clip=1`**)。
  クリップ層 → 下地の対応は **`m87.clip_base_map`**、フォルダーの範囲は
  **`m87.subtree_alpha`** を import(**写さない**)。`m27` は無改変
- **`opacity` を効かせる**(満タンでない層が 68 枚在る)
- **色調補正レイヤー(`layer_type & 4096`)は塗らない** —— 下の層を調整する層で、
  書き出されたラスタを乗せると絵が壊れる(ある頁では**1枚で誤差を +47% 増やした**)
- **マスク持ちの層(`layer_type & 2`)は飛ばさない** —— 書き出しの時点でマスクが
  アルファに入っている。**飛ばすと誤差が増える**(0.41 → 0.76)
- **合成モード(`composite` != 0)は再現しない** —— 符号の対応表を持っていないので
  **推測せず、数を数えて画面の但し書きに出す**(102 枚)。**IC6 で誤差を測る**
- **合成は頁ぜんたいを1回だけ**作って窓を切る。**窓ごとに合成すると同じ PNG を
  4回開く**(1頁 133 層なら 532 回)。**画素は1画素も変わらない**(アルファ合成は画素ごと)

**正解の対応表は行列に入れない。** `key.csv` に別に書く
—— 行列は UI がそのままブラウザへ送るので、**入れたら作者に見えてしまう**。

**登録に書いていないので器具が決めたこと**(印字もする):
- IC1 に複製する 2 窓の**選び方**(28 窓から一様。登録は枚数しか言っていない)
- 複製を**離す距離**(登録は「離して置く」とだけ言う)→ **10 問以上**。達成した距離を印字
- 提示の順と A/B/C の割り当ての**種**(窓の種 20261030 とは別にする。`SEED_SHOW`)
"""
import collections, csv, json, pathlib, sys, time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                     # noqa: E402
import m16_score_corrected_standard as m16           # noqa: E402
import m27_author_crops as m27                       # noqa: E402
import m87_split_standard_by_occlusion as m87         # noqa: E402
import m77_standard_visible as m77                   # noqa: E402

WINDOWS = pathlib.Path("results/m92_windows/windows.csv")
CACHE = pathlib.Path(
    "/tmp/claude-1000/-home-sh1-deepl-lineart-colour-manuscripts/"
    "56c20ca8-6335-4886-8b3b-961887500cde/scratchpad/m92_labels")
OUT = pathlib.Path("results/m93_ui")
WIN = 1024
SEED_SHOW = 20261031      # **提示の種。窓の種 20261030 とは別**
N_IC1 = 2                 # 登録: 複製は 2 窓まで(既知セルは1経路に1つ)
MIN_APART = 10            # **器具が決めた距離**(登録は「離して置く」とだけ言う)
AQUA = (240, 210, 150)    # BGR = 水色。**透明をここで見せる**
LINE = (255, 0, 255)      # マゼンタ 1px。**3案すべて同じ描き方**
PLANS = ("標準", "A0", "頁全体")
CORRECTION = 4096         # **台帳のビット表: 4096 = 色調補正レイヤー**(全数で例外 0)
IC6_MEAN = 3.0            # **工程の外から来る棒**(抽出側の検証 0.31〜2.42/255)
IC6_FRAC = 0.05

ASK = "この窓の中の絵を「面」に分けるとしたら、A・B・C のどれがあなたの分け方に最も近いですか。"
HELP = ("1画面 = 原稿の 1024x1024 画素そのまま(縮小していません)。"
        "A / B / C を押すと同じ窓の別の分け方に切り替わります。R で線なしの絵。"
        "水色は透明(その窓にその層が無い)。マゼンタの線が「面の境目」の案です。")


def edges1(lab):
    """ラベルが変わる所に **1 画素**の境界。**3案すべてこの関数を通す**。"""
    e = np.zeros(lab.shape, bool)
    e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    e[:-1, :] |= lab[:-1, :] != lab[1:, :]
    return e


def mark(base, lab):
    img = base.copy()
    if lab is not None:
        img[edges1(lab)] = LINE
    return img


def clip_mask(r, cbase, rowby, kids, files, grid, cache, stats):
    """**クリップの連鎖を親までたどった**マスク。無ければ None。

    クリップ層は下地の形にだけ乗る。**親のフォルダーがクリップされていれば中身も
    クリップされる**ので、`parent_id` を上へたどって重ねる。
    """
    m, cur, seen = None, r, set()
    while cur is not None:
        lid = str(cur["layer_id"])
        if lid in seen:
            break
        seen.add(lid)
        if cur.get("clip") == "1" and lid in cbase:
            b = cbase[lid]
            mk = (m87.subtree_alpha(b["layer_id"], kids, files, grid, cache)
                  if b.get("folder") == "True"
                  else m87.alpha_of(b["layer_id"], files, grid, cache))
            if mk is None:
                stats["**クリップの下地にラスタが無い**"] += 1
            else:
                m = mk.copy() if m is None else (m & mk)
        cur = rowby.get(str(cur.get("parent_id") or ""))
    return m


def compose_page(p, rows_vis, grid, stats, preview=None):
    """**クリップ・不透明度を効かせた原寸の合成**(下から上へ)。

    **用紙層(`layer_type` 1584)のラスタは使わない。** この納品では用紙のラスタが
    **純白**なのに、CLIP 自身の `_preview.png` の下地は**淡い青**だった ——
    **用紙の色は層のラスタに入っていない**(器具の都合ではなく納品の性質)。
    そこで**用紙の色を `_preview.png` から読む**: 用紙以外が何も乗らなかった画素を
    preview の側で拾い、その中央値を下地にする。**読めなければ白にして、そう印字する。**

    **「透明は水色」は効かない** —— 用紙が全画布を覆うので、透明な画素は残らない。
    **水色が出た割合を測って印字する**(備えが働いたと思い込まないため)。
    """
    files = {l["layer_id"]: l["file"] for l in p["layers"] if l.get("file")}
    roots, kids = m16.build_tree(rows_vis)
    z = m16.z_order(roots, kids)
    cbase = m87.clip_base_map(rows_vis)
    rowby = {str(r["layer_id"]): r for r in rows_vis}
    cache = {}
    gh, gw = grid
    C = np.zeros((gh, gw, 3), np.float32)      # 用紙を除いた積み上げ
    A = np.zeros((gh, gw), np.float32)         # その不透明度
    order = sorted((r for r in rows_vis if int(r["layer_id"]) in files),
                   key=lambda r: z.get(int(r["layer_id"]), 0))
    for r in order:
        t = int(r.get("layer_type") or 0)
        if t == int(m16.PAPER_TYPE):
            stats["用紙層(ラスタは使わず、色を preview から読む)"] += 1
            continue
        if t & CORRECTION:
            # **色調補正レイヤーは塗らない。** 下の層を調整する層で、書き出された
            # ラスタを乗せると絵が壊れる —— ある頁では**1枚で誤差を +47% 増やした**。
            # **`4096` = 色調補正レイヤーは台帳のビット表(全数で例外 0)**
            stats["色調補正レイヤー(塗らない。下の層を調整する層)"] += 1
            continue
        lid = int(r["layer_id"])
        im = cv2.imread(str(m12.D / files[lid]), cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            stats["**ラスタが読めない / RGBA でない層**"] += 1
            continue
        sub = im[:gh, :gw]
        a = sub[..., 3].astype(np.float32) / 255.0
        op = int(r.get("opacity") or 256)
        if op != 256:
            a *= min(1.0, op / 256.0)
            stats["不透明度を効かせた層"] += 1
        if r.get("clip") == "1":
            mk = clip_mask(r, cbase, rowby, kids, files, grid, cache, stats)
            if mk is None:
                stats["**クリップ層だが下地が決まらない(そのまま乗せた)**"] += 1
            else:
                a = a * mk
                stats["クリップを効かせた層"] += 1
        if str(r.get("composite") or "0") != "0":
            stats["**合成モードを再現できない層(そのまま乗せた)**"] += 1
        a3 = a[..., None]
        C = C * (1 - a3) + sub[..., :3].astype(np.float32) * a3
        A = A * (1 - a) + a
        del im, sub, a, a3
    cache.clear()

    paper = np.array([255.0, 255.0, 255.0], np.float32)
    if preview is not None:
        As = cv2.resize(A, (preview.shape[1], preview.shape[0]),
                        interpolation=cv2.INTER_AREA)
        bare = As < 0.01
        if bare.mean() >= 0.005:
            paper = np.median(preview[bare].reshape(-1, 3),
                              axis=0).astype(np.float32)
            stats[f"用紙の色を preview から読んだ BGR "
                  f"{tuple(int(v) for v in paper)}(用紙が見えている "
                  f"{bare.mean():.1%})"] += 1
        else:
            stats[f"**用紙が見えている画素が {bare.mean():.2%} しか無く、"
                  f"用紙の色を読めない(白にした)**"] += 1
    out = C + paper * (1 - A)[..., None]
    return np.ascontiguousarray(out.round().clip(0, 255).astype(np.uint8))


def page_stack(p, byp):
    """**見えている層**を z 順(下から)に並べた id と、その書き出しファイル。

    **向きは `z` の昇順。** `m16` は `pr = zmax - z` の**小さい方が勝つ**ので
    **z が大きい方が上**であり、`m27` も `key=lambda t: t[1]`(昇順)で渡している。
    **降順で渡すと、一番下の全画布の不透明な層を最後に乗せることになり、
    窓が単色になる** —— 初版はそれで 30/31 窓が単色だった(`m27.flat` が捕まえた)。
    """
    rows_vis, _ = m77.visible_rows(byp[p["source"]])
    files = {l["layer_id"]: l["file"] for l in p["layers"]}
    roots, kids = m16.build_tree(rows_vis)
    z = m16.z_order(roots, kids)
    ids = sorted((i for i in files if i in z), key=lambda i: z[i])
    return ids, files


def main():
    # **`--check-only`: 絵を描き直さず点検だけ回す。** 名前と順は種から決まるので同じ
    check_only = "--check-only" in sys.argv
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(WINDOWS.open(encoding="utf-8")))
    byp = m16.page_rows()
    pages = {p["page"]: p for p in
             json.load(open(m12.MANIFEST, encoding="utf-8"))["pages"]}
    rng = np.random.default_rng(SEED_SHOW)
    t0 = time.time()

    print(f"窓 {len(rows)} 行を {WINDOWS} から読んだ(**選び直さない**)")
    print(f"提示の種 {SEED_SHOW} / 複製 {N_IC1} / 離す距離 {MIN_APART} 問以上\n")

    # ---- IC1: 28 の「問う窓」から 2 窓を複製(**器具が決めた選び方**)----
    asks = [r for r in rows if r["role"] == "ask"]
    ic2s = [r for r in rows if r["role"] == "IC2"]
    dup_i = sorted(int(v) for v in rng.choice(len(asks), N_IC1, replace=False))
    items = []
    for i, r in enumerate(asks):
        items.append(dict(r, qrole="ask", twin=""))
    for i in dup_i:
        items.append(dict(asks[i], qrole="IC1", twin=asks[i]["item_id"]))
        items[i]["twin"] = asks[i]["item_id"]
    for r in ic2s:
        items.append(dict(r, qrole="IC2", twin=""))
    print(f"IC1 に複製した窓: "
          + ", ".join(f"{asks[i]['item_id']}({asks[i]['alias']})" for i in dup_i))

    # ---- 提示の順(複製が MIN_APART 以上離れるまで引き直す)----
    tries, order = 0, None
    while tries < 10000:
        tries += 1
        cand = list(rng.permutation(len(items)))
        pos = {items[j]["item_id"] + items[j]["qrole"]: k
               for k, j in enumerate(cand)}
        d = []
        for i in dup_i:
            a = pos[asks[i]["item_id"] + "ask"]
            b = pos[asks[i]["item_id"] + "IC1"]
            d.append(abs(a - b))
        if min(d) >= MIN_APART:
            order, dists = cand, d
            break
    if order is None:
        print(f"**{MIN_APART} 問以上離れる並びが {tries} 回で見つからない。出さない**")
        return 1
    print(f"提示の順: 引き {tries} 回 / 複製の間隔 {dists}(**{MIN_APART} 問以上**)\n")

    # ---- 絵を描く(頁ごとにまとめる。合成とラベルの読み込みは頁1回)----
    q, key, checks, stats = [], [], collections.Counter(), collections.Counter()
    a0diff, ic6 = [], []
    bypage = collections.defaultdict(list)
    for k, j in enumerate(order):
        bypage[items[j]["page"]].append((k, items[j]))

    for page in sorted(bypage):
        p = pages[page]
        s = int(round(p["preview"]["縮小率"]))
        lab_i = np.load(CACHE / f"{page}.std.npy")
        a0_small = np.load(CACHE / f"{page}.a0.npy")
        # **原寸へは `np.repeat`(最近傍)。4 画素の段を残す**(修正1-A)
        a0_full = np.repeat(np.repeat(a0_small, s, 0), s, 1)
        if a0_full.shape != lab_i.shape:
            print(f"**A0 を原寸にしたら標準と形が合わない {a0_full.shape} "
                  f"!= {lab_i.shape}。出さない**")
            return 1
        ids, files = page_stack(p, byp)
        H, W = lab_i.shape
        full = None
        if not check_only:
            rows_vis, _ = m77.visible_rows(byp[p["source"]])
            pv = cv2.imread(str(m12.D / p["preview"]["file"]), cv2.IMREAD_COLOR)
            full = compose_page(p, rows_vis, (H, W), stats, preview=pv)
            # ---- IC6: **CLIP 自身の合成(`_preview.png`)と突き合わせる** ----
            small = cv2.resize(full, (pv.shape[1], pv.shape[0]),
                              interpolation=cv2.INTER_AREA)
            d = np.abs(small.astype(np.int16) - pv.astype(np.int16))
            ic6.append((bypage[page][0][1]["alias"], float(d.mean()),
                        float(np.percentile(d, 99)),
                        float((d.max(axis=2) > 32).mean())))
        stats[f"{bypage[page][0][1]['alias']}: 合成した層"] = len(ids)
        missing = [i for i in ids if not (m12.D / files[i]).exists()]
        if missing:
            stats[f"{bypage[page][0][1]['alias']}: **書き出しが無い層**"] = len(missing)

        for k, it in bypage[page]:
            y, x = int(it["y"]), int(it["x"])
            base = (None if full is None
                    else np.ascontiguousarray(full[y:y + WIN, x:x + WIN]))
            lw = {"標準": lab_i[y:y + WIN, x:x + WIN],
                  "A0": a0_full[y:y + WIN, x:x + WIN],
                  "頁全体": None}
            # **A0 の面の数を原寸の揃った位置で数え直す**(m92 は y//s で寄せていた)
            na = int(np.count_nonzero(np.bincount(
                np.asarray(lw["A0"], np.int64).ravel())[1:]))
            if na != int(it["n_a0_areas"]):
                checks["A0 の面の数が m92 と違う(位置の寄せ方。棒には効かない)"] += 1
                a0diff.append((it["item_id"], int(it["n_a0_areas"]), na))
            slots = list(rng.permutation(3))
            base_name = f"q{k:02d}"
            imgs = {}
            for sl, pi in enumerate(slots):
                plan = PLANS[pi]
                nm = f"{base_name}_{'ABC'[sl]}.png"
                imgs['ABC'[sl]] = nm
                if not check_only:
                    img = mark(base, lw[plan])
                    cv2.imwrite(str(OUT / "crops" / nm), img)
                    if img.shape != (WIN, WIN, 3):
                        checks["**1024x1024 でない絵**"] += 1
                key.append({"order": k, "slot": "ABC"[sl], "plan": plan,
                            "item_id": it["item_id"], "qrole": it["qrole"],
                            "alias": it["alias"], "stratum": it["stratum"],
                            "twin": it["twin"], "file": nm})
            raw = f"{base_name}_raw.png"
            if not check_only:
                cv2.imwrite(str(OUT / "crops" / raw), base)
            q.append({
                "order": k, "crop": imgs["A"], "ctx": raw,
                "alt": [imgs["A"], imgs["B"], imgs["C"], raw],
                "work": f"窓 {k + 1}/{len(items)}  原寸 1024x1024",
                "pid": it["alias"], "scale": 1.0, "ctx_scale": 1.0,
                "interp": 1.0, "box_px": [y, x, WIN, WIN],
                "n_lay": len(ids), "white": 0.0,
                "cands": [], "word_share": {}, "top_word": "", "purity": 0.0,
                "n_lay90": 0, "auto": 0, "comp": 0, "area": None,
                "region": k, "lid": k, "why": "ask"})
            stats["描いた問"] += 1
        del lab_i, a0_small, a0_full, full
        print(f"  {bypage[page][0][1]['alias']}  問 {len(bypage[page]):2d}  "
              f"合成した層 {len(ids):3d}  描いた絵 {len(bypage[page]) * 4}", flush=True)

    q.sort(key=lambda r: r["order"])

    # ---- 器具の点検 ----
    print(f"\n== IC3(**原寸 1:1**。書いた PNG を読み直して当てる)==")
    bad, n_png = [], 0
    for f in sorted((OUT / "crops").glob("*.png")):
        im = cv2.imread(str(f), cv2.IMREAD_COLOR)
        n_png += 1
        if im is None or im.shape != (WIN, WIN, 3):
            bad.append(f.name)
    print(f"  PNG {n_png} / 1024x1024 でない: {len(bad)}  "
          f"→ {'**PASS**' if not bad else '**FAIL。出さない** ' + str(bad[:5])}")

    print(f"\n== IC4(**3案の下地が同じ絵であること**)==")
    # A/B/C の3枚は、マゼンタの画素を除けば1画素も違わないはず
    diff = []
    for it in q[:len(q)]:
        a = [cv2.imread(str(OUT / "crops" / n), cv2.IMREAD_COLOR)
             for n in (it["alt"][0], it["alt"][1], it["alt"][2])]
        m = np.ones((WIN, WIN), bool)
        for im in a:
            m &= ~np.all(im == np.array(LINE, np.uint8), axis=2)
        if not (np.array_equal(a[0][m], a[1][m]) and np.array_equal(a[0][m], a[2][m])):
            diff.append(it["order"])
    print(f"  問 {len(q)} / 下地が違う: {len(diff)}  "
          f"→ {'**PASS**' if not diff else '**FAIL。出さない** ' + str(diff[:5])}")

    print(f"\n== 観察: **窓にどれだけ絵が入っているか**(規則は足さない)==")
    print(f"  `m27` には「窓が単色なら捨てて次の候補へ」が在ったが、**今回の登録の"
          f"引き直し条件は「標準の面が0」の1つだけ**なので、**規則は足さない。数えるだけ**")
    # **単色かどうかは `m27.flat` を import する**(判定を2か所に書かない)。
    # **透明の割合も別に数える** —— 「透明は水色」が実際に効いたかは測らないと分からない
    ink = []
    for it in q:
        im = cv2.imread(str(OUT / "crops" / it["ctx"]), cv2.IMREAD_COLOR)
        if im is None:
            continue
        bg = float(np.all(im == np.array(AQUA, np.uint8), axis=2).mean())
        u, c = np.unique(m27.quant(im), return_counts=True)
        ink.append((it["order"], it["pid"], float(c.max() / c.sum()),
                    bool(m27.flat(im)), bg))
    if ink:
        v = sorted(r[2] for r in ink)
        print(f"  最頻色の占める割合: 中央 {v[len(v)//2]:.1%}  "
              f"範囲 {v[0]:.1%}〜{v[-1]:.1%}  窓 {len(ink)}")
        print(f"  **`m27.flat` が単色と言う窓: "
              f"{sum(1 for r in ink if r[3])} / {len(ink)}**")
        thin = sorted((r for r in ink if r[2] >= 0.90), key=lambda r: -r[2])
        print(f"  最頻色が 90% 以上の窓 {len(thin)}:")
        for o, pid, fr, fl, bgf in thin:
            print(f"    問{o:2d} {pid}  最頻色 {fr:.1%}"
                  f"{'  **m27.flat = 単色**' if fl else ''}"
                  f"  → **出す。登録に無い規則で窓を外さない**")
        n_flat = sum(1 for r in ink if r[3])
        print(f"\n  == IC5(**合成が壊れていないこと**。修正3)==")
        print(f"    期待値(先に書く): **単色の窓は半分未満。** 初版は z の向きを逆に"
              f"重ねて **30/31 が単色**だった(**この門が無かったので絵が出てしまった**)")
        print(f"    単色の窓 {n_flat} / {len(ink)}  → "
              + ("**PASS**" if n_flat * 2 < len(ink)
                 else "**FAIL。合成が壊れている。出さない**"))
        if n_flat * 2 >= len(ink):
            checks["**IC5 FAIL: 単色の窓が半分以上**"] += 1
        tr = sorted(r[4] for r in ink)
        print(f"  透明(水色が出た)画素の割合: 中央 {tr[len(tr)//2]:.2%}  "
              f"最大 {tr[-1]:.2%}")
        if tr[-1] == 0.0:
            print(f"    → **水色は1画素も出ていない。** 一番下に全画布の不透明な層が"
                  f"在るため。**「透明は水色」の備えは効いていない**(害も無い)")

    if ic6:
        print(f"\n== IC6(**CLIP 自身の合成と突き合わせる**。修正4)==")
        print(f"  **この値は工程の外から来る** —— `_preview.png` は CLIP が"
              f"クリップも不透明度も合成モードも効かせて書き出した像である。"
              f"こちらの原寸の合成を同じ大きさに縮めて引き算する")
        print(f"  期待値(先に書く): **平均の差が {IC6_MEAN}/255 以下、かつ 32 階調より"
              f"大きく違う画素が {IC6_FRAC:.0%} 未満**。**超えた頁は絵を出さない**")
        print(f"  **棒の出所は工程の外**: 抽出側が通常合成だけで組み直して"
              f"**4ページで平均差 0.31〜2.42/255**(台帳 C7)。"
              f"こちらもその帯に入らなければ、合成が壊れている")
        print(f"   別名   平均の差  99%点   差>32 の画素  判定")
        for al_, mu, p99, frac in ic6:
            ok = frac < IC6_FRAC and mu <= IC6_MEAN
            print(f"  {al_:6s} {mu:9.2f} {p99:7.1f} {frac:12.2%}  "
                  f"{'PASS' if ok else '**FAIL**'}")
            if not ok:
                checks[f"**IC6 FAIL: {al_} の合成が CLIP と違いすぎる**"] += 1

    print(f"\n== 観察: A0 の面の数(位置の寄せ方)==")
    print(f"  `m92` は窓の左上を `y//s` で preview 側に寄せて数えていた。"
          f"**ここは原寸で揃えて数え直した**(ずれは最大 s-1 画素)")
    print(f"  違った窓 {len(a0diff)} / {len(q)}")
    for iid, old, new in a0diff:
        print(f"    {iid}  m92 {old:4d} → 原寸で揃えて {new:4d}  (差 {new - old:+d})")
    print(f"  **どの棒もこの数に依っていない** —— 窓の引き直しは標準の面の数だけで決まる")

    print(f"\n== 行列 ==")
    print(f"  問 {len(q)} = 問う窓 {sum(1 for i in items if i['qrole']=='ask')}"
          f" + IC1 複製 {sum(1 for i in items if i['qrole']=='IC1')}"
          f" + IC2 {sum(1 for i in items if i['qrole']=='IC2')}")
    print(f"  層: 差分 {sum(1 for i in items if i['stratum']=='差分')} / "
          f"非差分 {sum(1 for i in items if i['stratum']=='非差分')}")
    print(f"  **A/B/C の割り当ては問いごとに引き直した**(NC2)。"
          f"案ごとの出た回数: "
          + ", ".join(f"{p}={sum(1 for r in key if r['plan']==p)}" for p in PLANS))
    for sl in "ABC":
        print(f"    位置 {sl} に入った案: "
              + ", ".join(f"{p}={sum(1 for r in key if r['slot']==sl and r['plan']==p)}"
                          for p in PLANS))

    print(f"\n== 但し書き(読むときに画面に出す)==")
    for k, v in stats.most_common():
        print(f"  {v:6d}  {k}")
    print(f"\n== 観察 ==")
    for k, v in checks.most_common():
        print(f"  {v:6d}  {k}")
    if not checks:
        print("  なし")

    if bad or diff or any("IC5 FAIL" in k or "IC6 FAIL" in k for k in checks):
        # **門が落ちたら古い行列を消す。** 残しておくと、壊れた絵を指した
        # `queue.json` を `m94` がそのまま作者に出せてしまう(**沈黙が正常動作**)
        gone = []
        for nm in ("queue.json", "key.csv"):
            if (OUT / nm).exists():
                (OUT / nm).unlink()
                gone.append(nm)
        print(f"\n**門が落ちたので行列を消した**: {gone or 'なし'}  "
              f"(絵は `crops/` に残るが、指す行列が無いので UI は立ち上がらない)")
        return 1
    if check_only:
        print("\n(--check-only: 行列と絵は書き換えていない)")
        return 0
    meta = dict(n=len(q), n_ask=len(q), n_check=0, n_auto=0, n_all=len(q),
                words=["A", "B", "C", "=同じくらい"], others=[],
                ask=ASK, help=HELP, queue=q)
    (OUT / "queue.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(OUT / "key.csv", "w", encoding="utf-8") as fh:
        cols = ["order", "slot", "plan", "item_id", "qrole", "alias", "stratum",
                "twin", "file"]
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(key)
    print(f"\n書いた: {OUT/'queue.json'}({len(q)} 問)  "
          f"{OUT/'key.csv'}({len(key)} 行)  {OUT/'crops'}({n_png} 枚)")
    print(f"**`key.csv` は行列に入れていない** —— 行列はそのままブラウザへ行くので、"
          f"入れたら作者に正解が見える")
    print(f"{time.time()-t0:.0f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
