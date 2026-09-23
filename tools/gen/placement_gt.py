#!/usr/bin/env python
"""GT 配置分析+単語の信頼性の見極め(2026-09-23 事前登録)。生成モデルは使わない。

層1 抽出面の信頼性(語ごと): X1 出現 / X2 細部不変 / X3 決定性 D / X4 形の凝集
    (Track F tools/stroke/word_eval.py の定義を語単位にしたもの)
層2 配置面の信頼性(語ごと): 位置×scale 64 セルの分布の特異性(JS 対 周辺分布)を
    half A / B で独立に並べ替え検定 → 再現あり / 配置中立 / 再現なし / 判定不能
層3 語ペアの相対配置: b−a を a の大きさで正規化した 8方向×4距離リング。帰無は
    b を同じコマの別のまとまりに置き換え。half A / B の両方で有意なら有意

--check: 道具確認のみ(X2/D の全体値が Track F 実測に一致 / 層2・層3 の偽陽性率 ≈5%)
--smoke: 語 20・ペア 50・並べ替え 200 回
"""
import argparse, csv, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trackf"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CODEBOOK, CORPUS, N_WORDS                           # noqa: E402
from train_codebook import load                                        # noqa: E402
from train_codebook2 import Codebook2                                  # noqa: E402
from render_real import CLUSTERS                                       # noqa: E402

HERE = Path(__file__).resolve().parent
N_DIR, RINGS = 8, np.array([1.0, 2.0, 4.0])    # 距離リング境界(a の大きさ単位)→ 4 リング
N_CELL3 = N_DIR * (len(RINGS) + 1)
A_SIZE = 28.0                                   # 復号原型の枠半径(±28)。a の大きさ = 28/scale_a


# ---------------------------------------------------------------- データ
def load_data(dev):
    cb = torch.load(CODEBOOK, map_location=dev, weights_only=False)
    cm = Codebook2(levels=tuple(cb["levels"]), coord_bins=cb["args"].get("coord_bins", 0)).to(dev)
    cm.load_state_dict(cb["model"]); cm.eval(); cm.rfsq.stages = cb["args"].get("stages", 3)
    Ps, Ws, Ms, ws, rows = [], [], [], [], []
    with torch.no_grad():
        for split in ("train", "test"):
            P, W, M, rr = load(CLUSTERS, split, dev)
            for i in range(0, len(P), 4096):
                ws.append(cm.encode(P[i:i + 4096].float(), W[i:i + 4096].float(), M[i:i + 4096], 1)[1].cpu())
            Ps.append(P); Ws.append(W); Ms.append(M); rows += rr
    P, W, M = torch.cat(Ps), torch.cat(Ws), torch.cat(Ms)
    words = torch.cat(ws).numpy()
    sg = json.load(open(HERE / "series_groups.json"))
    w2g = {w: g for g, ws_ in sg["groups"].items() for w in ws_}
    half = {g: 0 for g in sg["half_A"]} | {g: 1 for g in sg["half_B"]}
    D = dict(
        word=words, panel=np.array([int(r["panel"]) for r in rows]),
        scale=np.array([float(r["scale"]) for r in rows]),
        # Track F meta の列名は逆: cy 列 = x、cx 列 = y(2026-09-23 確認。strokes は (x, y) 順)
        x=np.array([float(r["cy"]) for r in rows]), y=np.array([float(r["cx"]) for r in rows]),
        nstk=np.array([int(r["n"]) for r in rows]),
        group=np.array([w2g[r["work"]] for r in rows]), work=np.array([r["work"] for r in rows]),
    )
    D["half"] = np.array([half[g] for g in D["group"]])
    dims = np.load(CORPUS)["dims"]
    D["H"] = dims[D["panel"], 0].astype(float); D["Wd"] = dims[D["panel"], 1].astype(float)
    return cm, P, W, M, D


def dedupe(D):
    """同一群内の重複コマ(クラスタ位置/8 と本数の一致がコマの過半)を検出し、
    作品名の辞書順で後の方のコマを除く。返る: keep マスク, 除去コマ数"""
    key_pan = defaultdict(set)
    size = Counter(D["panel"].tolist())
    pw = {}
    for i in range(len(D["panel"])):
        p = int(D["panel"][i]); pw[p] = (D["group"][i], D["work"][i])
        key_pan[(round(D["x"][i] / 8), round(D["y"][i] / 8), int(D["nstk"][i]))].add(p)
    share = Counter()
    for ps in key_pan.values():
        if 1 < len(ps) < 30:
            ps = sorted(ps)
            for x in range(len(ps)):
                for y in range(x + 1, len(ps)):
                    share[(ps[x], ps[y])] += 1
    drop = set()
    for (a, b), c in share.items():
        if pw[a][1] != pw[b][1] and pw[a][0] == pw[b][0] and c / min(size[a], size[b]) > 0.5:
            drop.add(a if pw[a][1] > pw[b][1] else b)
    return ~np.isin(D["panel"], list(drop)), len(drop)


# ---------------------------------------------------------------- 層1
def perturb_words(cm, P, W, M, idx, rng):
    """Track F word_eval と同じ手続き: 最長の線 1 本+4番目以下から最大2本をそれぞれ抜く。
    返る: [(i, changed_long, [changed_minor...])](本数 < 4 は除外)"""
    out = []
    with torch.no_grad():
        for i in idx:
            pp, ww, mk = P[i].float(), W[i].float(), M[i]
            n = int(mk.sum())
            if n < 4:
                continue
            arc = (pp[:n, 1:] - pp[:n, :-1]).norm(dim=-1).sum(-1)
            order = torch.argsort(arc, descending=True).cpu().numpy()
            picks = [order[0]] + list(rng.choice(order[3:], min(2, n - 3), replace=False))
            Pk = pp[None].repeat(len(picks) + 1, 1, 1, 1); Wk = ww[None].repeat(len(picks) + 1, 1)
            Mk = mk[None].repeat(len(picks) + 1, 1)
            for k, s in enumerate(picks):
                Mk[k + 1, s] = False
            wds = cm.encode(Pk, Wk, Mk, 1)[1]
            out.append((int(i), int(wds[1] != wds[0]), [int(wds[k + 1] != wds[0]) for k in range(1, len(picks))]))
    return out


def chamfer(a, b):
    d = torch.cdist(a, b)
    return float(0.5 * (d.min(1).values.mean() + d.min(0).values.mean()))


def boot_ci(fn, n, rng, B=1000):
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def layer1(cm, P, W, M, D, words_eval, rng, n_inst=200, n_pair=100):
    by_word = defaultdict(list)
    for i, w in enumerate(D["word"]):
        by_word[int(w)].append(i)
    res = {}
    for w in words_eval:
        pool = np.array(by_word.get(w, []))
        r = dict(word=int(w), count=len(pool), series=len(set(D["group"][pool])) if len(pool) else 0)
        r["X1"] = bool(r["count"] >= 50 and r["series"] >= 3)
        if len(pool) >= 2:
            sel = rng.choice(pool, min(n_inst, len(pool)), replace=False)
            pr = perturb_words(cm, P, W, M, sel, rng)
            r["n_eval"] = len(pr)
            if len(pr) >= 30:
                cl = np.array([x[1] for x in pr], float)
                cmn = [np.array(x[2], float) for x in pr]
                pm = lambda ix: float(np.mean(np.concatenate([cmn[j] for j in ix])))
                r["p_long"], r["p_minor"] = float(cl.mean()), pm(range(len(pr)))
                r["D"] = r["p_long"] - r["p_minor"]
                r["D_lo"], r["D_hi"] = boot_ci(lambda ix: float(cl[ix].mean()) - pm(ix), len(pr), rng)
                r["X2"] = bool(r["p_minor"] <= 0.25)
                r["X3"] = bool(r["D_lo"] > 0)
            # X4 purity
            same, diff = [], []
            for _ in range(n_pair):
                a, b = rng.choice(pool, 2, replace=False)
                c = rng.integers(0, len(D["word"]))
                pa = P[a][M[a]].float().reshape(-1, 2)
                same.append(chamfer(pa, P[b][M[b]].float().reshape(-1, 2)))
                diff.append(chamfer(pa, P[c][M[c]].float().reshape(-1, 2)))
            same, diff = np.array(same), np.array(diff)
            r["purity"] = float(np.median(same) / np.median(diff))
            r["pur_lo"], r["pur_hi"] = boot_ci(
                lambda ix: float(np.median(same[ix]) / np.median(diff[ix])), n_pair, rng)
            r["X4"] = bool(r["pur_hi"] < 1)
        r["extract_ok"] = bool(r["X1"] and r.get("X2", False) and r.get("X3", False) and r.get("X4", False))
        res[int(w)] = r
    return res


# ---------------------------------------------------------------- 層2
def cells64(D, sq):
    y = np.clip((D["y"] / D["H"] * 4).astype(int), 0, 3)
    x = np.clip((D["x"] / D["Wd"] * 4).astype(int), 0, 3)
    s = np.clip(np.searchsorted(sq, np.log(D["scale"])), 0, 3)
    return (y * 4 + x) * 4 + s


def js_rows(H, m):
    """H: (..., V, 64) の各行と周辺 m (64) の JS 距離(bits)。行和 0 は nan"""
    tot = H.sum(-1, keepdims=True)
    p = np.where(tot > 0, H / np.maximum(tot, 1), 0)
    mm = 0.5 * (p + m)
    kl = lambda a, b: np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0).sum(-1)
    return 0.5 * kl(p, mm) + 0.5 * kl(m, mm)


def layer2_half(word, cell, words_eval, n_perm, rng, min_n=20):
    V = N_WORDS
    H = np.bincount(word * 64 + cell, minlength=V * 64).reshape(V, 64).astype(float)
    m = H.sum(0) / H.sum()
    real = js_rows(H[words_eval], m)
    null = np.zeros((n_perm, len(words_eval)))
    for k in range(n_perm):
        Hp = np.bincount(rng.permutation(word) * 64 + cell, minlength=V * 64).reshape(V, 64).astype(float)
        null[k] = js_rows(Hp[words_eval], m)
    thr = np.percentile(null, 95, axis=0)
    cnt = H[words_eval].sum(1)
    return dict(js=real, thr=thr, sig=real > thr, n=cnt, ok=cnt >= min_n)


def layer2(D, keep, words_eval, n_perm, rng, shuffle_words=False):
    sq = np.quantile(np.log(D["scale"][keep]), [0.25, 0.5, 0.75])
    cell = cells64(D, sq)
    out = {}
    for h in (0, 1):
        sel = keep & (D["half"] == h)
        w = D["word"][sel].astype(int)
        if shuffle_words:                       # 道具確認: 語ラベルを無意味化
            w = rng.permutation(w)
        out[h] = layer2_half(w, cell[sel], words_eval, n_perm, rng)
    res = {}
    for j, w in enumerate(words_eval):
        a, b = out[0], out[1]
        if not (a["ok"][j] and b["ok"][j]):
            v = "判定不能"
        elif a["sig"][j] and b["sig"][j]:
            v = "再現あり"
        elif not a["sig"][j] and not b["sig"][j]:
            v = "配置中立"
        else:
            v = "再現なし"
        res[int(w)] = dict(verdict=v, jsA=float(a["js"][j]), thrA=float(a["thr"][j]), nA=int(a["n"][j]),
                           jsB=float(b["js"][j]), thrB=float(b["thr"][j]), nB=int(b["n"][j]))
    fp = {h: float(out[h]["sig"][out[h]["ok"]].mean()) for h in (0, 1)}
    return res, fp


# ---------------------------------------------------------------- 層3
def rel_cell(dy, dx, size_a):
    ang = (np.arctan2(dy, dx) + math.pi) / (2 * math.pi) * N_DIR
    d = np.clip(ang.astype(int), 0, N_DIR - 1)
    r = np.searchsorted(RINGS, np.hypot(dy, dx) / size_a)
    return r * N_DIR + d


def enumerate_pairs(D, keep):
    """全コマの順序対 (i→j, i≠j) の相対セルを列挙。返る: 対の配列と、各 i の帰無確率 (N,32)"""
    idx = np.flatnonzero(keep)
    by_p = defaultdict(list)
    for i in idx:
        by_p[int(D["panel"][i])].append(i)
    null = np.zeros((len(D["word"]), N_CELL3))
    A, Bj, C, Pn = [], [], [], []
    for p, ii in by_p.items():
        ii = np.array(ii)
        if len(ii) < 2:
            continue
        a, b = np.meshgrid(ii, ii, indexing="ij")
        off = a != b
        a, b = a[off], b[off]
        c = rel_cell(D["y"][b] - D["y"][a], D["x"][b] - D["x"][a], A_SIZE / D["scale"][a])
        np.add.at(null, (a, c), 1.0)
        A.append(a); Bj.append(b); C.append(c); Pn.append(np.full(len(a), p))
    null /= np.maximum(null.sum(1, keepdims=True), 1)
    return np.concatenate(A), np.concatenate(Bj), np.concatenate(C), np.concatenate(Pn), null


def pair_stats(D, pairs, null, sel_rows, n_perm, dev, fake=False, gen=None):
    """1 ペア・1 half 分: 行 = (a, b, cell, panel)。各コマの重み合計 1。
    返る: KL(実 || 帰無期待) と帰無並べ替えの 95 パーセンタイル"""
    a, c, pn = pairs[0][sel_rows], pairs[2][sel_rows], pairs[3][sel_rows]
    _, inv, cnt = np.unique(pn, return_inverse=True, return_counts=True)
    wt = 1.0 / cnt[inv]
    probs = torch.tensor(null[a], dtype=torch.float32, device=dev)
    wt_t = torch.tensor(wt, dtype=torch.float32, device=dev)
    exp = (probs * wt_t[:, None]).sum(0)
    exp = (exp + 1e-3) / (exp + 1e-3).sum()
    if fake:                                    # 道具確認: 実セルを帰無からの1回抽出で置換
        c = torch.multinomial(probs, 1, generator=gen)[:, 0].cpu().numpy()
    real = np.bincount(c, weights=wt, minlength=N_CELL3)
    real = torch.tensor(real, dtype=torch.float32, device=dev)
    kl = lambda h: ((h + 1e-3) / (h + 1e-3).sum(-1, keepdim=True)
                    * torch.log2(((h + 1e-3) / (h + 1e-3).sum(-1, keepdim=True)) / exp)).sum(-1)
    samp = torch.multinomial(probs, n_perm, replacement=True, generator=gen).T      # (n_perm, m)
    hs = torch.zeros(n_perm, N_CELL3, device=dev).scatter_add_(1, samp, wt_t.expand(n_perm, -1))
    return float(kl(real[None])[0]), float(torch.quantile(kl(hs), 0.95)), real.cpu().numpy(), exp.cpu().numpy()


def layer3(D, keep, n_perm, dev, max_pairs=0, fake=False, seed=0, min_panels=30, min_series=3, min_half=10):
    A, Bj, C, Pn, null = enumerate_pairs(D, keep)
    wa, wb = D["word"][A].astype(int), D["word"][Bj].astype(int)
    canon = (wa < wb) | ((wa == wb) & (A < Bj))          # 無順序ペアは語ID小を a に
    A, Bj, C, Pn, wa, wb = A[canon], Bj[canon], C[canon], Pn[canon], wa[canon], wb[canon]
    key = wa * N_WORDS + wb
    order = np.argsort(key, kind="stable")
    A, Bj, C, Pn, key = A[order], Bj[order], C[order], Pn[order], key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    res = []
    cand = []
    for s, e in zip(starts, ends):
        pan = np.unique(Pn[s:e])
        if len(pan) < min_panels:
            continue
        ser = set(D["group"][A[s:e]])
        if len(ser) < min_series:
            continue
        cand.append((s, e, len(pan), len(ser)))
    cand.sort(key=lambda t: -t[2])
    if max_pairs:
        cand = cand[:max_pairs]
    for s, e, npan, nser in cand:
        k = int(key[s]); r = dict(a=k // N_WORDS, b=k % N_WORDS, panels=npan, series=nser)
        rows = np.arange(s, e)
        ok = True
        for h, nm in ((0, "A"), (1, "B")):
            rr = rows[D["half"][A[rows]] == h]
            r[f"n{nm}"] = int(len(np.unique(Pn[rr])))
            if r[f"n{nm}"] < min_half:
                ok = False
                continue
            kl, thr, real, exp = pair_stats(D, (A, Bj, C, Pn), null, rr, n_perm, dev, fake, gen)
            r[f"kl{nm}"], r[f"thr{nm}"] = kl, thr
            r[f"sig{nm}"] = bool(kl > thr)
            r[f"real{nm}"], r[f"exp{nm}"] = real.tolist(), exp.tolist()
        r["verdict"] = ("判定不能" if not ok else "有意" if r["sigA"] and r["sigB"]
                        else "片側のみ" if r["sigA"] or r["sigB"] else "非有意")
        res.append(r)
    return res, (A, Bj, C, Pn)


# ---------------------------------------------------------------- 描画
def draw_inst(canvas, P, M, ci, x, y, sc, color, th=1):
    """pts は (x, y) 順・マスクは線単位 (S,)。座標はそのまま cv2 の (x, y)"""
    pts = P[ci].float().cpu().numpy() / sc + np.array([x, y])
    m = M[ci].cpu().numpy()
    for s in np.flatnonzero(m):
        cv2.polylines(canvas, [np.round(pts[s]).astype(np.int32)], False, color, th, cv2.LINE_AA)


def word_grid(P, M, D, w, rng, n=16, cellpx=90):
    pool = np.flatnonzero(D["word"] == w)
    sel = rng.choice(pool, min(n, len(pool)), replace=False) if len(pool) else []
    g = np.full((cellpx * 2, cellpx * 8, 3), 255, np.uint8)
    for k, ci in enumerate(sel):
        y0, x0 = (k // 8) * cellpx + cellpx / 2, (k % 8) * cellpx + cellpx / 2
        draw_inst(g, P, M, ci, x0, y0, A_SIZE * 2 / (cellpx * 0.9), (40, 40, 40))
    cv2.rectangle(g, (0, 0), (g.shape[1] - 1, g.shape[0] - 1), (200, 200, 200), 1)
    return g


def pair_overlay(P, M, D, pairs, r, rng, n=40, px=360):
    """a を中心・一定の大きさに正規化し、b の実インスタンスを相対位置に重ねる(灰=b、黒=a 1個)"""
    A, Bj, _C, Pn = pairs
    k = r["a"] * N_WORDS + r["b"]
    rows = np.flatnonzero((D["word"][A] == r["a"]) & (D["word"][Bj] == r["b"]))
    pan_first = {}
    for i in rng.permutation(rows):
        pan_first.setdefault(int(Pn[i]), i)
    rows = list(pan_first.values())[:n]
    g = np.full((px, px, 3), 255, np.uint8)
    ref = A_SIZE / (px / 10)                      # a の大きさ = 画像幅の 1/10
    c0 = px / 2
    for i in rows:
        a, b = A[i], Bj[i]
        k_ = D["scale"][a] / ref
        draw_inst(g, P, M, b, c0 + (D["x"][b] - D["x"][a]) * k_, c0 + (D["y"][b] - D["y"][a]) * k_,
                  D["scale"][b] / k_, (150, 150, 150))
    if rows:
        draw_inst(g, P, M, A[rows[0]], c0, c0, ref, (0, 0, 0), 2)
    for rr in RINGS:
        cv2.circle(g, (int(c0), int(c0)), int(rr * px / 10), (220, 200, 200), 1)
    cv2.putText(g, f"a={r['a']} b={r['b']} n={r['panels']}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
    return g


# ---------------------------------------------------------------- 語の総合区分
def word_tier(l1, l2):
    ex = l1["extract_ok"]
    v = l2["verdict"] if l2 else "判定不能"
    if ex and v in ("再現あり", "配置中立"):
        return "両面信頼"
    if ex:
        return "抽出のみ信頼"
    if v == "再現あり":
        return "配置のみ(語彙改善候補)"
    return "不信頼"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/placement_gt_20260923")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=20260923)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cm, P, W, M, D = load_data(dev)
    keep, n_drop = dedupe(D)
    print(f"clusters {len(D['word'])} | 重複コマ除去 {n_drop} コマ ({(~keep).sum()} まとまり) | "
          f"half A {len(set(D['panel'][keep & (D['half'] == 0)]))} / B {len(set(D['panel'][keep & (D['half'] == 1)]))} コマ",
          flush=True)
    n_perm = 200 if a.smoke else 1000

    if a.check:
        # T1: Track F word_eval の全体値の再現(test 2,000 まとまり、seed 0)
        rng0 = np.random.default_rng(0)
        n_train = sum(1 for r in csv.DictReader(open(CLUSTERS / "meta.csv")) if r["split"] == "train")
        n_test = len(D["word"]) - n_train
        idx = n_train + rng0.choice(n_test, min(2000, n_test), replace=False)
        pr = perturb_words(cm, P, W, M, idx, rng0)
        pl = float(np.mean([x[1] for x in pr])); pm = float(np.mean([c for x in pr for c in x[2]]))
        print(f"T1 Track F 再現: minor {pm:.4f} (実測 0.232) / D {pl - pm:.4f} (実測 0.166)", flush=True)
        # T2: 層2 偽陽性率(語ラベル無意味化)
        rng = np.random.default_rng(a.seed)
        _, fp2 = layer2(D, keep, np.arange(N_WORDS), n_perm, rng, shuffle_words=True)
        print(f"T2 層2 偽陽性率: A {fp2[0]:.3f} / B {fp2[1]:.3f}(期待 ≈0.05)", flush=True)
        # T3: 層3 偽陽性率(実セルを帰無からの抽出で置換)
        r3, _ = layer3(D, keep, n_perm, dev, max_pairs=300, fake=True, seed=a.seed)
        fa = np.mean([r["sigA"] for r in r3 if "sigA" in r]); fb = np.mean([r["sigB"] for r in r3 if "sigB" in r])
        print(f"T3 層3 偽陽性率(300 ペア): A {fa:.3f} / B {fb:.3f}(期待 ≈0.05)", flush=True)
        json.dump(dict(T1=dict(minor=pm, D=pl - pm, n=len(pr)), T2=fp2, T3=dict(A=float(fa), B=float(fb)),
                       dedupe_panels=n_drop), open(out / "check.json", "w"), indent=1)
        return

    rng = np.random.default_rng(a.seed)
    cnt = np.bincount(D["word"][keep].astype(int), minlength=N_WORDS)
    words_eval = np.argsort(-cnt)[:20] if a.smoke else np.flatnonzero(cnt > 0)
    # 層1
    l1 = layer1(cm, P, W, M, D, [int(w) for w in words_eval], rng)
    print(f"層1 抽出面で信頼: {sum(r['extract_ok'] for r in l1.values())}/{len(l1)} 語", flush=True)
    # 層2
    l2, _ = layer2(D, keep, words_eval, n_perm, rng)
    print("層2:", dict(Counter(r["verdict"] for r in l2.values())), flush=True)
    tier = {w: word_tier(l1[w], l2.get(w)) for w in l1}
    print("語の区分:", dict(Counter(tier.values())), flush=True)
    with open(out / "word_reliability.csv", "w", newline="") as f:
        cols = ["word", "tier", "count", "series", "X1", "n_eval", "p_minor", "X2", "D", "D_lo", "X3",
                "purity", "pur_hi", "X4", "extract_ok", "verdict", "jsA", "thrA", "nA", "jsB", "thrB", "nB"]
        wr = csv.DictWriter(f, cols, extrasaction="ignore"); wr.writeheader()
        for w in l1:
            wr.writerow({**l1[w], **l2.get(w, {}), "tier": tier[w]})
    # 層3
    r3, pairs = layer3(D, keep, n_perm, dev, max_pairs=50 if a.smoke else 0, seed=a.seed)
    for r in r3:
        ta, tb = tier.get(r["a"], "不信頼"), tier.get(r["b"], "不信頼")
        r["tier"] = "主(両語とも両面信頼)" if ta == tb == "両面信頼" else "参考"
    print(f"層3 対象ペア {len(r3)}:", dict(Counter((r['tier'], r['verdict']) for r in r3)), flush=True)
    json.dump(r3, open(out / "pairs.json", "w"))
    with open(out / "pairs.csv", "w", newline="") as f:
        cols = ["a", "b", "tier", "verdict", "panels", "series", "nA", "nB", "klA", "thrA", "klB", "thrB"]
        wr = csv.DictWriter(f, cols, extrasaction="ignore"); wr.writeheader()
        for r in sorted(r3, key=lambda r: -min(r.get("klA", 0), r.get("klB", 0))):
            wr.writerow(r)
    # 図: 層ごとの語グリッド / 層3 重ね描き
    rng_f = np.random.default_rng(1)
    for t in sorted(set(tier.values())):
        ws_t = [w for w in l1 if tier[w] == t]
        ws_t = sorted(ws_t, key=lambda w: -l1[w]["count"])[:8]
        if not ws_t:
            continue
        rows = []
        for w in ws_t:
            g = word_grid(P, M, D, w, rng_f)
            cv2.putText(g, f"w{w} n={l1[w]['count']}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
            rows.append(np.pad(g, ((0, 6), (0, 0), (0, 0)), constant_values=200))
        cv2.imwrite(str(out / f"words_{t}.png"), np.vstack(rows))
    for t in ("主(両語とも両面信頼)", "参考"):
        sig = sorted([r for r in r3 if r["tier"] == t and r["verdict"] == "有意"],
                     key=lambda r: -min(r["klA"], r["klB"]))[:12]
        if not sig:
            continue
        ims = [pair_overlay(P, M, D, pairs, r, rng_f) for r in sig]
        ims += [np.full_like(ims[0], 255)] * (-len(ims) % 4)
        grid = np.vstack([np.hstack([np.pad(im, ((0, 4), (0, 4), (0, 0)), constant_values=200)
                                     for im in ims[i:i + 4]]) for i in range(0, len(ims), 4)])
        cv2.imwrite(str(out / f"pairs_{t.split('(')[0]}.png"), grid)
    print("out ->", out, flush=True)


if __name__ == "__main__":
    main()
