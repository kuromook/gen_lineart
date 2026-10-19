#!/usr/bin/env python
"""K19 作者への 2 度目の依頼の用紙(2026-10-19 事前登録)。

隙間 ≤8px の**未判定で聞ける箱 B の対の全数 28 件**(抽出ではない)に、
反復 8 / 錨 8 / **前回との突き合わせ 8** を混ぜて 52 タイル。
**作者には種類を知らせない。**対応表は key.json(用紙には出さない)。

K10 からの変更は 2 点だけで、どちらも登録に理由が書いてある:
  - **錨は箱 A の ≤8px から**引く(本番が短いので、長さだけで錨を見分けられないように)
  - **前回答えた 8 件を混ぜる**(作者が今回の分析を読んでいるので、基準の移動を測る)
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import k2_box_b_gray as K
import k18_gap_cost as K18

SEED, N_REPEAT, N_ANCHOR, N_CROSS = 20261019, 8, 8, 8
MAX_GAP = 8.0            # 本番の帯。起案 §5 の ≤8px
TILE, ZOOM, PER_PAGE, NCOL = 150, 3, 8, 4
# **K10 の 2 倍・赤丸では駄目だった。**本番の隙間は中央値 5.92px で K10(10.8)の半分なので、
# 半径 9 の丸が**隙間そのものを覆う**。定数は前回の値ではなく前回の理由から引き継ぐ:
#   - 拡大は 3 倍(最短 2.24px でも 7px に開く)
#   - tip は**線に直交する短い棒**で示す(隙間の上に何も置かない)
#   - 外側の細い円は「ここを見よ」だけを言う
OPTS = ["(1) 飛ばし線", "(2) かすれ", "(3) 切れていない", "(4) 判断できない"]


def okey(r):
    """同定の鍵。**出力の中の位置から振らない**(K10 訂正 4 の一般形)。"""
    return (int(r["panel"]), int(r["s0"]), int(r["s1"]), int(r["e0"]), int(r["e1"]),
            round(float(r.get("gap_sk", r.get("gap"))), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(K.TRACK_H / "labels/state.json"))
    ap.add_argument("--contig", default="results/k8_contiguity_20261018/per_pair_contig.csv")
    ap.add_argument("--k18", default="results/k18_gap_cost_20261019/per_pair.csv")
    ap.add_argument("--k10", default="results/k10_author_request_20261018")
    ap.add_argument("--out", default="results/k19_author_request_20261019")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    contig = [r for r in csv.DictReader(open(ROOT / a.contig)) if r["missing"] == "0"]
    answerable = {(r["panel"], r["s0"], r["s1"], r["e0"], r["e1"])
                  for r in contig
                  if r["box"] == "B" and not (r["contiguous"] == "1" and r["comp_same"] == "0")}
    thread = {(r["panel"], r["s0"], r["s1"], r["e0"], r["e1"])
              for r in contig if r["box"] == "B" and r["contiguous"] == "1"}
    per = list(csv.DictReader(open(ROOT / a.k18)))

    # --- 本番: 抽出せず全数 ---
    main_items = [dict(r, kind="main", gap=float(r["gap_sk"]),
                       thread=int((r["panel"], r["s0"], r["s1"], r["e0"], r["e1"]) in thread))
                  for r in per
                  if float(r["gap_sk"]) <= MAX_GAP and r["answer"] == "0"
                  and (r["panel"], r["s0"], r["s1"], r["e0"], r["e1"]) in answerable]
    mg = np.sort([r["gap"] for r in main_items])
    print(f"本番 {len(main_items)} 件(全数)  隙間 {mg.min():.2f}-{mg.max():.2f} "
          f"中央値 {np.median(mg):.2f}")

    # --- 反復: 本番から 8 件 ---
    rep = [dict(main_items[i], kind="repeat")
           for i in rng.permutation(len(main_items))[:N_REPEAT]]

    # --- 錨: 箱 A の ≤8px から、本番の隙間分位に対応させて(登録どおり K10 と逆向き) ---
    pool = sorted((r for r in contig if r["box"] == "A" and float(r["gap"]) <= MAX_GAP),
                  key=lambda r: float(r["gap"]))
    targets = [float(np.quantile(mg, (k + .5) / N_ANCHOR)) for k in range(N_ANCHOR)]
    used_pairs, anchors = set(), []
    for t in targets:
        best = min((k for k in range(len(pool))
                    if (pool[k]["panel"], pool[k]["s0"], pool[k]["s1"]) not in used_pairs),
                   key=lambda k: abs(float(pool[k]["gap"]) - t))
        used_pairs.add((pool[best]["panel"], pool[best]["s0"], pool[best]["s1"]))
        anchors.append(dict(pool[best], kind="anchor", gap=float(pool[best]["gap"]), thread=0))
    print("錨の隙間:", [round(r["gap"], 1) for r in anchors],
          " (本番の分位", [round(t, 1) for t in targets], ")")
    assert len({(r["panel"], r["s0"], r["s1"]) for r in anchors}) == N_ANCHOR

    # --- 前回との突き合わせ: K10 の本番 40 件から、隙間の全域に散らして 8 件 ---
    k10key = json.load(open(ROOT / a.k10 / "key.json"))["key"]
    k10ans = json.load(open(ROOT / a.k10 / "answers.json"))["answers"]
    prev = []
    for did, it in k10key.items():
        if it["kind"] != "main":
            continue
        m = [r for r in per if int(r["panel"]) == int(it["panel"])
             and int(r["s0"]) == int(it["s0"]) and int(r["s1"]) == int(it["s1"])
             and abs(float(r["gap_sk"]) - float(it["gap"])) < 0.02]
        if m:
            prev.append(dict(m[0], kind="cross", gap=float(m[0]["gap_sk"]),
                             prev_answer=int(k10ans[did]),
                             thread=int((m[0]["panel"], m[0]["s0"], m[0]["s1"],
                                         m[0]["e0"], m[0]["e1"]) in thread)))
    # **前回の答えで層化する**(隙間で散らすと「判断できない」に偏り、
    # 測りたい「飛ばし線 <-> かすれ の境界が動いたか」に効かない)。層内は隙間で散らす。
    prev.sort(key=lambda r: r["gap"])
    want = {1: 3, 2: 3, 4: 2}                 # 飛ばし線 3 / かすれ 3 / 判断できない 2
    cross = []
    for ansv, k_ in want.items():
        st = [r for r in prev if r["prev_answer"] == ansv]
        if not st:
            continue
        idx = np.linspace(0, len(st) - 1, min(k_, len(st)))
        cross += [dict(st[int(round(q))]) for q in idx]
    print(f"  前回の答えの在庫: " +
          " ".join(f"{v}={sum(1 for r in prev if r['prev_answer'] == v)}" for v in (1, 2, 3, 4)))
    print(f"突き合わせ {len(cross)} 件  隙間 {min(r['gap'] for r in cross):.1f}-"
          f"{max(r['gap'] for r in cross):.1f}  前回の答え "
          f"{sorted(r['prev_answer'] for r in cross)}")

    items = main_items + rep + anchors + cross
    # --- 並べ方: 反復の 2 回目は初出から PER_PAGE*2 以上離す(K10 訂正 3) ---
    for _ in range(8000):
        order = rng.permutation(len(items))
        cand = [items[i] for i in order]
        pos, ok = {}, True
        for n, it in enumerate(cand):
            k_ = okey(it)
            if k_ in pos and abs(n - pos[k_]) < PER_PAGE * 2:
                ok = False; break
            pos[k_] = n
        if ok:
            items = cand; break
    else:
        raise SystemExit("反復を離して並べられない")
    for n, it in enumerate(items):
        it["display_id"] = n + 1

    sys.path.insert(0, str(K.TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    P = G.S["panels"]
    byp = defaultdict(list)
    for it in items:
        byp[int(it["panel"])].append(it)
    tiles = {}
    for pid, its in byp.items():
        g = cv2.imread(str(K.DATASET / P[pid]["source"] / "line" / P[pid]["name"]), 0)
        Q = K.pipeline(g)
        for it in its:
            i, j, ei, ej = int(it["s0"]), int(it["s1"]), int(it["e0"]), int(it["e1"])
            if i not in Q["tips"] or j not in Q["tips"]:
                print(f"  **tip なし** コマ{pid} {i},{j}"); continue
            pa, pb = Q["tips"][i][ei], Q["tips"][j][ej]
            c = (pa + pb) / 2.0
            r0 = int(np.clip(c[0] - TILE // 2, 0, g.shape[0] - TILE))
            c0 = int(np.clip(c[1] - TILE // 2, 0, g.shape[1] - TILE))
            im = cv2.cvtColor(g[r0:r0 + TILE, c0:c0 + TILE], cv2.COLOR_GRAY2BGR)
            im = cv2.resize(im, (TILE * ZOOM, TILE * ZOOM), interpolation=cv2.INTER_NEAREST)
            qa = ((pa[1] - c0) * ZOOM, (pa[0] - r0) * ZOOM)
            qb = ((pb[1] - c0) * ZOOM, (pb[0] - r0) * ZOOM)
            mid = ((qa[0] + qb[0]) // 2, (qa[1] + qb[1]) // 2)
            cv2.circle(im, (int(mid[0]), int(mid[1])), 26, (0, 165, 255), 1)
            for p_, sp_, e_ in ((qa, i, ei), (qb, j, ej)):
                u = K18.outward_tangent(Q, sp_, e_)
                if u is None:
                    cv2.circle(im, (int(p_[0]), int(p_[1])), 5, (30, 30, 210), 2); continue
                n_ = np.array([-u[1], u[0]]) * 7.0 * ZOOM / 2.0     # 線に直交する棒
                a_ = (int(p_[0] - n_[1]), int(p_[1] - n_[0]))
                b_ = (int(p_[0] + n_[1]), int(p_[1] + n_[0]))
                cv2.line(im, a_, b_, (30, 30, 210), 2, cv2.LINE_AA)
            head = np.full((34, im.shape[1], 3), 255, np.uint8)
            cv2.putText(head, f"#{it['display_id']}", (6, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2, cv2.LINE_AA)
            tiles[it["display_id"]] = np.pad(np.concatenate([head, im], 0),
                                             ((3, 3), (3, 3), (0, 0)), constant_values=120)
    ids = sorted(tiles)
    pages = []
    for p0 in range(0, len(ids), PER_PAGE):
        chunk = [tiles[k] for k in ids[p0:p0 + PER_PAGE]]
        while len(chunk) % NCOL:
            chunk.append(np.full_like(chunk[0], 250))
        rowsim = [np.concatenate(chunk[k:k + NCOL], 1) for k in range(0, len(chunk), NCOL)]
        p = out / f"sheet_{len(pages) + 1:02d}.png"
        cv2.imwrite(str(p), np.concatenate(rowsim, 0)); pages.append(str(p))
    key = {str(it["display_id"]): dict(kind=it["kind"], panel=it["panel"], order=it.get("order"),
                                       label=it.get("label"), box=("A" if it["kind"] == "anchor"
                                                                   else "B"),
                                       gap=round(it["gap"], 2), thread=it["thread"],
                                       s0=it["s0"], s1=it["s1"], e0=it["e0"], e1=it["e1"],
                                       prev_answer=it.get("prev_answer"))
           for it in items}
    json.dump(dict(registered=dict(seed=SEED, max_gap=MAX_GAP,
                                   n_main=len(main_items), n_repeat=N_REPEAT,
                                   n_anchor=N_ANCHOR, n_cross=N_CROSS,
                                   main_is_census="抽出ではなく ≤8px の未判定で聞ける対の全数",
                                   options=OPTS,
                                   bands=dict(repeat="一致 >= 6/8 かつ偶然一致を上回る。"
                                                     "1 つの選択肢が 7 割以上なら判定不能",
                                              anchor="(3) >= 6/8 かつ (1) <= 1/8",
                                              cross="反復の帯を当てる(一致 >= 6/8)。"
                                                    "前回の答えで層化(飛ばし線 3 / かすれ 3 / "
                                                    "判断できない 2)、層ごとにも印字"),
                                   anchor_rule="箱 A の <=8px から本番の隙間分位に対応。"
                                               "K10 とは逆向き(本番が短いため)",
                                   reading="n=42: CP95 下限>0.50(要 28/42)→越えた / "
                                           "点推定>=0.50 かつ下限<=0.50 →判定保留 / "
                                           "CP95 上限<0.50(要 <=14/42)→軸が死ぬ",
                                   power="真の精度 0.625 のとき 越える 0.350 / "
                                         "判定保留 0.650 / 軸が死ぬ 0.000。"
                                         "**判定保留が最も起こりやすい。材料の上限の表示である**",
                                   null_value=0.225, null_statistic="精度",
                                   denominator="判断できないを含める",
                                   blinding="**盲検ではない。作者は『短い隙間ほどかすれ、"
                                            "精度 0.625』という今回の分析を読んでいる。**"
                                            "錨が同じ短い帯に居るのがその直接の検査"),
                   pages=pages, key=key),
              open(out / "key.json", "w"), indent=1, ensure_ascii=False)
    req = out / "request.md"
    req.write_text(
        "# 線の切れ目についての判定(2 度目)\n\n"
        f"タイル {len(ids)} 枚、{len(pages)} 枚の用紙。各タイルは線画を 2 倍に拡大した切り出しで、\n"
        "**橙の細い円**が見るべき場所、その中の**赤い短い棒 2 本**が、\n"
        "こちらの処理が「線の端」と判断した 2 点です(棒は線に直交しています)。\n"
        "**2 本の棒のあいだには何も描いていません。**そこを見てください。\n\n"
        "**問い: その 2 点のあいだは、もともとどうなっていましたか。**\n\n"
        + "\n".join(f"- **{o}**" for o in OPTS) + "\n\n"
        "- (1) 飛ばし線 = **意図的に切って描いた**(処理が繋がなかったのは正しい)\n"
        "- (2) かすれ = **描いたが出力で飛んだ**(処理が繋ぐべきだった)\n"
        "- (3) 切れていない = **そもそも切れて見えない**(繋がっている)\n"
        "- (4) 判断できない\n\n"
        "**答え方**: `#1:1 #2:4 #3:1 ...` のように番号で返してください。\n"
        "**迷ったら (4) で構いません。**(4) は捨てずに分母に入れて数えます。\n\n"
        "用紙:\n" + "\n".join(f"- `{p}`" for p in pages) + "\n")
    print(f"\n用紙 {len(pages)} 枚 / タイル {len(ids)} 枚")
    for p in pages:
        print(" ", p)
    print(" ", req)
    print(" ", out / "key.json", "(**作者には見せない**)")


if __name__ == "__main__":
    main()
