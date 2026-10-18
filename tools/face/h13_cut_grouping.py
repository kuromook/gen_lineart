#!/usr/bin/env python
"""実験 H13(2026-10-17 事前登録): 近さ τ でつなぎ、学んだ組の判定は「切る」ほうにだけ使う。組の正例の重みなし。学習用だけ。

C0 = τ 0.04 でつなぐだけ、C1 = τ 0.04 + 切る、C2 = τ 0.08 + 切る。
出力: summary.json、check.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_server as G                                               # noqa: E402
import h7_stroke_parts as H                                            # noqa: E402
import h10_scale_pose as H10                                           # noqa: E402
import h11_set_model as M                                              # noqa: E402
import h12_learned_grouping as H12                                     # noqa: E402

H12.RADIUS = 0.08
NONE = H12.NONE
VARIANTS = {"C0": (0.04, False), "C1": (0.04, True), "C2": (0.08, True)}
T2_ALL, T2_PREC = 0.321, 0.204


def train_eval(faces, PAIRS, tr, te, seed, dev, labels=None):
    """H12.train_eval と同じ。違いは組の交差エントロピーに正例の重みを付けないことだけ"""
    rng = np.random.default_rng(20261017 + seed); torch.manual_seed(20261017 + seed)
    X0 = np.vstack([M.feats(faces[i], True) for i in tr]); mu, sd = X0.mean(0), X0.std(0) + 1e-6
    lab = (lambda i: faces[i]["y"]) if labels is None else (lambda i: labels[i][0])
    psame = (lambda i: PAIRS[i][3]) if labels is None else (lambda i: labels[i][1])
    net = H12.Net(X0.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(M.EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // M.BATCH)):
            Xs, ys = [], []
            for i in b:
                f = faces[i]
                if rng.random() < 0.5:
                    f = M.flipped(f)
                f = M.jittered(f, rng)
                Xs.append((M.feats(f, True) - mu) / sd); ys.append(lab(i))
            X, Y, pad = M.batchify(Xs, ys, dev)
            h, logit = net(X, pad)
            loss = torch.nn.functional.cross_entropy(logit.reshape(-1, len(H.CLASSES)), Y.reshape(-1))
            pl, pt = [], []
            for r, i in enumerate(b):
                pi, pj, pd, _ = PAIRS[i]
                if len(pi):
                    pl.append(net.pairs(h[r], torch.tensor(pi, device=dev), torch.tensor(pj, device=dev), torch.tensor(pd, device=dev)))
                    pt.append(torch.tensor(psame(i), dtype=torch.float32, device=dev))
            if pl:
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(torch.cat(pl), torch.cat(pt))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); out = {}
    with torch.no_grad():
        for i in te:
            x = torch.tensor(((M.feats(faces[i], True) - mu) / sd)[None], dtype=torch.float32, device=dev)
            h, logit = net(x, torch.zeros((1, x.shape[1]), dtype=torch.bool, device=dev))
            pi, pj, pd, _ = PAIRS[i]
            ps = torch.sigmoid(net.pairs(h[0], torch.tensor(pi, device=dev), torch.tensor(pj, device=dev), torch.tensor(pd, device=dev))).cpu().numpy() if len(pi) else np.zeros(0)
            out[i] = (torch.softmax(logit[0], 1).cpu().numpy(), ps)
    return out, net, mu, sd


def groups_of(f, prob, ps, PAIR, tau, cut):
    pred = prob.argmax(1); pi, pj, pd, _ = PAIR
    par = list(range(len(pred)))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for a, b, d, s in zip(pi, pj, pd, ps):
        if pred[a] == pred[b] != NONE and d < tau and not (cut and s < 0.5):
            par[find(a)] = find(b)
    comp = {}
    for a in np.flatnonzero(pred != NONE):
        comp.setdefault(find(a), set()).add(int(a))
    return [(int(pred[next(iter(v))]), v) for v in comp.values()]


def run(faces, PAIRS, fold, seed, dev, labels=None):
    res = [None] * len(faces)
    for k in sorted(set(fold.tolist())):
        tr = np.flatnonzero(fold != k); te = np.flatnonzero(fold == k)
        o, *_ = train_eval(faces, PAIRS, tr, te, seed * 10 + k, dev, labels)
        for i in te:
            res[i] = o[i]
    return res


def score(faces, PAIRS, res, tau, cut, idx=None):
    idx = range(len(faces)) if idx is None else idx
    return H12.evaluate([faces[i] for i in idx], [groups_of(faces[i], res[i][0], res[i][1], PAIRS[i], tau, cut) for i in idx], [res[i][0].argmax(1) for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h13_cut_grouping_20261017")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda"); t0 = time.time()
    faces = H.load_faces(H10.MAX_ORDER, H10.N_LINES)
    fold, _ = H.folds_of(faces)
    PAIRS = [H12.cand_pairs(f) for f in faces]

    # 道具確認 2: 正解のクラスと正解の組。あわせて C0 のつなぎ方が H7 の components と同じかも見る
    onehot = [np.eye(len(H.CLASSES))[f["y"]] for f in faces]
    ceil = {name: score(faces, PAIRS, [(onehot[i], P[3].astype(float)) for i, P in enumerate(PAIRS)], tau, cut)["all"] for name, (tau, cut) in VARIANTS.items()}
    same_as_h7 = all(sorted(map(sorted, (v for _, v in groups_of(f, onehot[i], np.ones(len(PAIRS[i][0])), PAIRS[i], 0.04, False)))) ==
                     sorted(map(sorted, (v for _, v in H.components(f, f["y"], 0.04)))) for i, f in enumerate(faces))
    # 道具確認 1: 並び順
    tr = np.flatnonzero(fold != 0); te = np.flatnonzero(fold == 0)[:4]
    o, net, mu, sd = train_eval(faces, PAIRS, tr, te, 99, dev)
    ok1 = True
    with torch.no_grad():
        for i in te:
            perm = np.random.default_rng(i).permutation(len(faces[i]["y"])); inv = np.argsort(perm)
            x = torch.tensor(((M.feats(faces[i], True) - mu) / sd)[perm][None], dtype=torch.float32, device=dev)
            h, logit = net(x, torch.zeros((1, x.shape[1]), dtype=torch.bool, device=dev))
            pi, pj, pd, _ = PAIRS[i]
            ps = torch.sigmoid(net.pairs(h[0], torch.tensor(inv[pi], device=dev), torch.tensor(inv[pj], device=dev), torch.tensor(pd, device=dev))).cpu().numpy()
            ok1 &= bool(np.allclose(torch.softmax(logit[0], 1).cpu().numpy()[inv], o[i][0], atol=1e-4) and np.allclose(ps, o[i][1], atol=1e-4))
    # 道具確認 3: かき混ぜ(クラスと組を同じ置換で)
    rng = np.random.default_rng(20261017); shuf = {}
    for i, f in enumerate(faces):
        p = rng.permutation(len(f["y"])); g = f["gid"][p]; pi, pj, _, _ = PAIRS[i]
        shuf[i] = (f["y"][p], (g[pi] >= 0) & (g[pi] == g[pj]))
    neg_res = run(faces, PAIRS, fold, 7, dev, labels=shuf)
    neg = {name: score(faces, PAIRS, neg_res, tau, cut)["all"] for name, (tau, cut) in VARIANTS.items()}
    chk = dict(n_faces=len(faces), n_pairs=int(sum(len(P[0]) for P in PAIRS)), pos_rate=round(float(np.mean(np.concatenate([P[3] for P in PAIRS]))), 4),
               order_invariant=ok1, c0_same_as_h7_components=same_as_h7, oracle_recovered=ceil, negative_recovered=neg, minutes=round((time.time() - t0) / 60, 1))
    # C2 の上限の線は 0.95 で事前登録したが、τ = 0.08 で超えられない値だった(0.940)。2026-10-17 ユーザー了承で、作りは変えず 0.93 に読み替える
    chk["passed"] = bool(ok1 and same_as_h7 and ceil["C1"] >= ceil["C0"] and ceil["C2"] >= 0.93 and max(neg.values()) <= 0.05)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return
    S = {name: [] for name in VARIANTS}
    for seed in M.SEEDS:
        res = run(faces, PAIRS, fold, seed, dev)
        for name, (tau, cut) in VARIANTS.items():
            e = score(faces, PAIRS, res, tau, cut)
            e["by_pose"] = {y: score(faces, PAIRS, res, tau, cut, [i for i, f in enumerate(faces) if f["pose"]["yaw"] == y])["all"] for y in ("front", "oblique", "profile")}
            S[name].append(e)
            print(name, seed, {k: e[k] for k in ("all", "eye", "others", "others_size12", "pred_group_precision", "n_pred_groups")}, f"{(time.time() - t0) / 60:.1f} 分", flush=True)
    summ = {}
    for name in VARIANTS:
        r = S[name]; keys = [k for k in r[0] if not isinstance(r[0][k], (list, dict))]
        summ[name] = dict(mean={k: round(float(np.mean([x[k] for x in r])), 3) for k in keys}, range_all=round(max(x["all"] for x in r) - min(x["all"] for x in r), 3),
                          stroke_PR_mean={c: [round(float(np.mean([x[c + "_stroke_PR"][j] for x in r])), 3) for j in (0, 1)] for c in ("eye", "brow", "nose", "mouth", "ear")},
                          by_pose_mean={y: round(float(np.mean([x["by_pose"][y] for x in r])), 3) for y in ("front", "oblique", "profile")}, seeds=r)
        summ[name]["delta_all_vs_T2"] = round(summ[name]["mean"]["all"] - T2_ALL, 3)
        summ[name]["delta_precision_vs_T2"] = round(summ[name]["mean"]["pred_group_precision"] - T2_PREC, 3)
    for name in ("C1", "C2"):
        summ[name]["delta_all_vs_C0"] = round(summ[name]["mean"]["all"] - summ["C0"]["mean"]["all"], 3)
    json.dump(summ, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for name in VARIANTS:
        s = summ[name]; print(name, s["mean"], s["range_all"], s["stroke_PR_mean"], s["by_pose_mean"], s["delta_all_vs_T2"], s["delta_precision_vs_T2"], s.get("delta_all_vs_C0"))
    print("->", out)


if __name__ == "__main__":
    main()
