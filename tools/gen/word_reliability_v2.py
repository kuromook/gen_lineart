#!/usr/bin/env python
"""層1(抽出面の信頼性)第2版(2026-09-24 事前登録)。

第1回(placement_gt.py layer1)との違い: 全インスタンスで X2/X3 を測り、X2/X3/X4 を
95% ブートストラップ区間で 通過/不通過/境界 の3値判定する。閾値は第1回と同じ。

--check: 一括版で Track F 全体値(test 2,000、細部変化率 0.232 / D 0.166)を再計算
既定: seed 3 つで全語を測り、seed 間の通過↔不通過の反転を数える
"""
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from placement_gt import load_data, word_grid, CLUSTERS                 # noqa: E402


@torch.no_grad()
def perturb_all(cm, P, W, M, idx, gen, dev, chunk=1024):
    """idx の各まとまり(本数 ≥4 のみ)について 最長1本 / 4番目以下から最大2本 を抜いた語の変化。
    返る: inst (K,), changed_long (K,), minor_sum (K,), minor_n (K,)"""
    out_i, out_l, out_s, out_n = [], [], [], []
    idx = torch.as_tensor(idx, device=dev)
    for c in range(0, len(idx), chunk):
        ii = idx[c:c + chunk]
        pp, ww, mk = P[ii].float(), W[ii].float(), M[ii]
        n = mk.sum(1)
        ok = n >= 4
        ii, pp, ww, mk, n = ii[ok], pp[ok], ww[ok], mk[ok], n[ok]
        B, S = mk.shape
        if B == 0:
            continue
        arc = (pp[:, :, 1:] - pp[:, :, :-1]).norm(dim=-1).sum(-1)
        arc = torch.where(mk, arc, torch.full_like(arc, -1.0))
        order = torch.argsort(arc, dim=1, descending=True)             # 有効な線が先頭 n 個
        rank = torch.arange(S, device=dev)[None].expand(B, -1)
        key = torch.rand(B, S, device=dev, generator=gen)
        key = torch.where((rank >= 3) & (rank < n[:, None]), key, torch.full_like(key, -1.0))
        pick = torch.argsort(key, dim=1, descending=True)[:, :2]       # 4番目以下(順位 3..n-1)から最大2
        m1 = torch.gather(order, 1, pick[:, :1])[:, 0]
        m2 = torch.gather(order, 1, pick[:, 1:2])[:, 0]
        has2 = (n - 3) >= 2
        drops = [order[:, 0], m1, m2]
        Pk = pp.repeat(4, 1, 1, 1); Wk = ww.repeat(4, 1); Mk = mk.repeat(4, 1).clone()
        ar = torch.arange(B, device=dev)
        for k, d in enumerate(drops):
            Mk[(k + 1) * B + ar, d] = False
        wds = cm.encode(Pk, Wk, Mk, 1)[1].view(4, B)
        ch = wds[1:] != wds[0][None]
        out_i.append(ii); out_l.append(ch[0])
        out_s.append(ch[1].float() + torch.where(has2, ch[2].float(), torch.zeros_like(ch[2].float())))
        out_n.append(1.0 + has2.float())
    cat = lambda xs: torch.cat(xs).cpu().numpy()
    return cat(out_i), cat(out_l).astype(float), cat(out_s), cat(out_n)


def ci3(lo, hi, pass_if, fail_if):
    return "通過" if pass_if(lo, hi) else "不通過" if fail_if(lo, hi) else "境界"


@torch.no_grad()
def chamfer_batch(P, M, ia, ib, dev, chunk=200):
    """点集合(線マスク内の全点)どうしの chamfer を一括計算"""
    out = []
    for c in range(0, len(ia), chunk):
        a, b = torch.as_tensor(ia[c:c + chunk], device=dev), torch.as_tensor(ib[c:c + chunk], device=dev)
        pa = P[a].float().flatten(1, 2); pb = P[b].float().flatten(1, 2)          # (B, 512, 2)
        ma = M[a][:, :, None].expand(-1, -1, P.shape[2]).flatten(1)
        mb = M[b][:, :, None].expand(-1, -1, P.shape[2]).flatten(1)
        d = torch.cdist(pa, pb)
        big = torch.tensor(1e9, device=dev)
        d = torch.where(mb[:, None, :], d, big)
        da = d.min(2).values; da = (da * ma).sum(1) / ma.sum(1)
        d = torch.where(ma[:, :, None], d, big)
        db = d.min(1).values; db = (db * mb).sum(1) / mb.sum(1)
        out.append(0.5 * (da + db))
    return torch.cat(out).cpu().numpy()


def measure(cm, P, W, M, D, words, seed, dev, n_pair=400, B=1000):
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    by_w = {w: np.flatnonzero(D["word"] == w) for w in words}
    allidx = np.concatenate([by_w[w] for w in words])
    inst, cl, ms, mn = perturb_all(cm, P, W, M, allidx, gen, dev)
    wi = D["word"][inst]
    res = {}
    for w in words:
        pool = by_w[w]
        r = dict(word=int(w), count=len(pool), series=len(set(D["group"][pool])))
        r["X1"] = "通過" if r["count"] >= 50 and r["series"] >= 3 else "不通過"
        sel = wi == w
        k = int(sel.sum()); r["n_eval"] = k
        if k >= 30:
            l, s, n = cl[sel], ms[sel], mn[sel]
            r["p_long"], r["p_minor"] = float(l.mean()), float(s.sum() / n.sum())
            r["D"] = r["p_long"] - r["p_minor"]
            bi = rng.integers(0, k, (B, k))
            pm_b = s[bi].sum(1) / n[bi].sum(1); d_b = l[bi].mean(1) - pm_b
            r["pm_lo"], r["pm_hi"] = np.percentile(pm_b, [2.5, 97.5]).tolist()
            r["D_lo"], r["D_hi"] = np.percentile(d_b, [2.5, 97.5]).tolist()
            r["X2"] = ci3(r["pm_lo"], r["pm_hi"], lambda lo, hi: hi <= 0.25, lambda lo, hi: lo > 0.25)
            r["X3"] = ci3(r["D_lo"], r["D_hi"], lambda lo, hi: lo > 0, lambda lo, hi: hi < 0)
        else:
            r["X2"] = r["X3"] = "不通過"                # 評価数不足(第1回と同じ扱い)
        if len(pool) >= 2:
            ia = np.array([rng.choice(pool, 2, replace=False) for _ in range(n_pair)])
            ic = rng.integers(0, len(D["word"]), n_pair)
            same = chamfer_batch(P, M, ia[:, 0], ia[:, 1], dev)
            diff = chamfer_batch(P, M, ia[:, 0], ic, dev)
            r["purity"] = float(np.median(same) / np.median(diff))
            bi = rng.integers(0, n_pair, (B, n_pair))
            pb = np.median(same[bi], 1) / np.median(diff[bi], 1)
            r["pur_lo"], r["pur_hi"] = np.percentile(pb, [2.5, 97.5]).tolist()
            r["X4"] = ci3(r["pur_lo"], r["pur_hi"], lambda lo, hi: hi < 1, lambda lo, hi: lo >= 1)
        else:
            r["X4"] = "不通過"
        vs = [r["X1"], r["X2"], r["X3"], r["X4"]]
        r["verdict"] = "不信頼" if "不通過" in vs else "信頼" if all(v == "通過" for v in vs) else "境界"
        res[int(w)] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/word_reliability_v2_20260924")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seeds", default="20260924,1,2")
    ap.add_argument("--limit_words", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cm, P, W, M, D = load_data(dev)

    if a.check:
        n_train = sum(1 for r in csv.DictReader(open(CLUSTERS / "meta.csv")) if r["split"] == "train")
        rng0 = np.random.default_rng(0)
        idx = n_train + rng0.choice(len(D["word"]) - n_train, 2000, replace=False)
        gen = torch.Generator(device=dev); gen.manual_seed(0)
        _i, cl, ms, mn = perturb_all(cm, P, W, M, idx, gen, dev)
        pm = ms.sum() / mn.sum()
        print(f"一括版 Track F 再計算: minor {pm:.4f} (実測 0.232) / D {cl.mean() - pm:.4f} (実測 0.166) / 評価 {len(cl)}")
        json.dump(dict(minor=float(pm), D=float(cl.mean() - pm), n=int(len(cl))), open(out / "check.json", "w"))
        return

    cnt = np.bincount(D["word"].astype(int), minlength=500)
    words = np.flatnonzero(cnt > 0)
    if a.limit_words:
        words = np.argsort(-cnt)[:a.limit_words]
    seeds = [int(s) for s in a.seeds.split(",")]
    runs = {}
    for s in seeds:
        runs[s] = measure(cm, P, W, M, D, [int(w) for w in words], s, dev)
        print(f"seed {s}:", dict(Counter(r["verdict"] for r in runs[s].values())), flush=True)
    # seed 間の安定性
    flips, border_moves, table = [], 0, []
    for w in words:
        w = int(w); vs = [runs[s][w]["verdict"] for s in seeds]
        per = {k: [runs[s][w][k] for s in seeds] for k in ("X2", "X3", "X4")}
        hard = any(("通過" in v and "不通過" in v) for v in [vs] + list(per.values()))
        if hard:
            flips.append(w)
        elif len(set(vs)) > 1:
            border_moves += 1
        table.append(dict(word=w, **{f"s{s}": runs[s][w]["verdict"] for s in seeds}))
    print(f"安定性: 通過↔不通過の反転 {len(flips)} 語 {flips[:20]} / 境界との出入り {border_moves} 語", flush=True)
    base = runs[seeds[0]]
    cols = ["word", "verdict", "count", "series", "X1", "n_eval", "p_minor", "pm_lo", "pm_hi", "X2",
            "D", "D_lo", "D_hi", "X3", "purity", "pur_lo", "pur_hi", "X4"]
    with open(out / "word_reliability_v2.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, cols, extrasaction="ignore"); wr.writeheader()
        for w in words:
            wr.writerow(base[int(w)])
    with open(out / "seed_agreement.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, list(table[0])); wr.writeheader(); wr.writerows(table)
    json.dump(dict(seeds=seeds, flips=flips, border_moves=border_moves,
                   counts={s: dict(Counter(r["verdict"] for r in runs[s].values())) for s in seeds}),
              open(out / "stability.json", "w"), indent=1, ensure_ascii=False)
    rng_f = np.random.default_rng(1)
    for v in ("信頼", "境界", "不信頼"):
        ws_v = sorted([int(w) for w in words if base[int(w)]["verdict"] == v], key=lambda w: -base[w]["count"])[:8]
        rows = []
        for w in ws_v:
            g = word_grid(P, M, D, w, rng_f)
            cv2.putText(g, f"w{w} n={base[w]['count']}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
            rows.append(np.pad(g, ((0, 6), (0, 0), (0, 0)), constant_values=200))
        if rows:
            cv2.imwrite(str(out / f"words_{v}.png"), np.vstack(rows))
    print("out ->", out)


if __name__ == "__main__":
    main()
