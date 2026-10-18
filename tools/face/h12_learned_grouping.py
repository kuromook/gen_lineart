#!/usr/bin/env python
"""実験 H12(2026-10-16 事前登録): H11 の T2 に「線どうしが同じグループか」の判定を足し、つなぎ方を学ばせる。学習用だけ。

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

RADIUS = 0.25
NONE = H.CLASSES.index("none")


def cand_pairs(f):
    P = f["pts"]
    D = np.linalg.norm(P[:, None, :, None] - P[None, :, None, :], axis=-1).min((2, 3))
    i, j = np.nonzero(np.triu(D < RADIUS * f["s"], 1))
    same = (f["gid"][i] >= 0) & (f["gid"][i] == f["gid"][j])
    return i, j, (D[i, j] / f["s"]).astype(np.float32), same


class Net(M.Net):
    def __init__(self, din):
        super().__init__(din)
        self.pair = torch.nn.Sequential(torch.nn.Linear(2 * M.D + 1, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

    def forward(self, x, pad):
        h = self.enc(self.inp(x), src_key_padding_mask=pad)
        return h, self.out(h)

    def pairs(self, h, i, j, d):
        a, b = h[i], h[j]
        return self.pair(torch.cat([a + b, (a - b).abs(), d[:, None]], 1))[:, 0]


def train_eval(faces, PAIRS, tr, te, seed, dev, labels=None):
    rng = np.random.default_rng(20261016 + seed); torch.manual_seed(20261016 + seed)
    X0 = np.vstack([M.feats(faces[i], True) for i in tr]); mu, sd = X0.mean(0), X0.std(0) + 1e-6
    lab = (lambda i: faces[i]["y"]) if labels is None else (lambda i: labels[i][0])
    psame = (lambda i: PAIRS[i][3]) if labels is None else (lambda i: labels[i][1])
    pos = np.mean(np.concatenate([psame(i) for i in tr])); pw = torch.tensor((1 - pos) / max(pos, 1e-6), device=dev)
    net = Net(X0.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
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
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(torch.cat(pl), torch.cat(pt), pos_weight=pw)
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


def groups_of(f, prob, ps, PAIR, drop):
    pred = prob.argmax(1); pi, pj, _, _ = PAIR
    par = list(range(len(pred)))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for a, b, s in zip(pi, pj, ps):
        if pred[a] == pred[b] != NONE and s > 0.5:
            par[find(a)] = find(b)
    comp = {}
    for a in np.flatnonzero(pred != NONE):
        comp.setdefault(find(a), set()).add(int(a))
    res = []
    for v in comp.values():
        c = int(pred[next(iter(v))])
        if drop and prob[list(v), c].mean() < 0.5:
            continue
        res.append((c, v))
    return res


def evaluate(faces, comps_list, preds):
    """H7 の group_eval と同じ数え方を、与えたグループで行う"""
    rows, n_pred, n_hit = [], 0, 0
    for f, comps in zip(faces, comps_list):
        truth = [(int(f["y"][f["gid"] == g][0]), set(np.flatnonzero(f["gid"] == g).tolist())) for g in sorted(set(f["gid"][f["gid"] >= 0].tolist()))]
        hit = set()
        for c, Gs in truth:
            ok = False
            for j, (pc, Ps) in enumerate(comps):
                if pc == c and len(Gs & Ps) >= H.TH * len(Gs) and len(Gs & Ps) >= H.TH * len(Ps):
                    ok = True; hit.add(j); break
            rows.append((c, len(Gs), ok))
        n_pred += len(comps); n_hit += len(hit)
    y = np.concatenate([f["y"] for f in faces]); p = np.concatenate(preds)
    out = dict(all=H.rate(rows)[0], eye=H.rate(rows, lambda c, n: c == H10.EYE)[0], others=H.rate(rows, lambda c, n: c in H10.OTHERS)[0],
               others_size12=H.rate(rows, lambda c, n: c in H10.OTHERS and n <= 2)[0], pred_group_precision=round(n_hit / max(n_pred, 1), 3), n_pred_groups=n_pred)
    for c in H10.OTHERS + [H10.EYE]:
        out[H.CLASSES[c]] = H.rate(rows, lambda cc, n, c=c: cc == c)[0]
        tp = int(((p == c) & (y == c)).sum())
        out[H.CLASSES[c] + "_stroke_PR"] = [round(tp / max(int((p == c).sum()), 1), 3), round(tp / max(int((y == c).sum()), 1), 3)]
    return out


def run(faces, PAIRS, fold, seed, dev, labels=None):
    res = [None] * len(faces)
    for k in sorted(set(fold.tolist())):
        tr = np.flatnonzero(fold != k); te = np.flatnonzero(fold == k)
        o, *_ = train_eval(faces, PAIRS, tr, te, seed * 10 + k, dev, labels)
        for i in te:
            res[i] = o[i]
    return res


def score(faces, PAIRS, res, drop, idx=None):
    idx = range(len(faces)) if idx is None else idx
    fs = [faces[i] for i in idx]
    return evaluate(fs, [groups_of(faces[i], res[i][0], res[i][1], PAIRS[i], drop) for i in idx], [res[i][0].argmax(1) for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h12_learned_grouping_20261016")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda"); t0 = time.time()
    faces = H.load_faces(H10.MAX_ORDER, H10.N_LINES)
    fold, _ = H.folds_of(faces)
    PAIRS = [cand_pairs(f) for f in faces]

    # 道具確認 2: 正解のクラスと正解の組
    oracle = [groups_of(f, np.eye(len(H.CLASSES))[f["y"]], P[3].astype(float), P, False) for f, P in zip(faces, PAIRS)]
    ceil = evaluate(faces, oracle, [f["y"] for f in faces])["all"]
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
    rng = np.random.default_rng(20261016); shuf = {}
    for i, f in enumerate(faces):
        p = rng.permutation(len(f["y"])); g = f["gid"][p]; pi, pj, _, _ = PAIRS[i]
        shuf[i] = (f["y"][p], (g[pi] >= 0) & (g[pi] == g[pj]))
    neg = score(faces, PAIRS, run(faces, PAIRS, fold, 7, dev, labels=shuf), False)["all"]
    chk = dict(n_faces=len(faces), n_pairs=int(sum(len(P[0]) for P in PAIRS)), pos_rate=round(float(np.mean(np.concatenate([P[3] for P in PAIRS]))), 4),
               order_invariant=ok1, oracle_grouping_recovered=ceil, negative_recovered=neg, minutes=round((time.time() - t0) / 60, 1))
    chk["passed"] = bool(ok1 and ceil >= 0.95 and neg <= 0.05)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return
    S = {"G1": [], "G2": []}
    for seed in M.SEEDS:
        res = run(faces, PAIRS, fold, seed, dev)
        for name, drop in (("G1", False), ("G2", True)):
            e = score(faces, PAIRS, res, drop)
            e["by_pose"] = {y: score(faces, PAIRS, res, drop, [i for i, f in enumerate(faces) if f["pose"]["yaw"] == y])["all"] for y in ("front", "oblique", "profile")}
            S[name].append(e)
            print(name, seed, {k: e[k] for k in ("all", "eye", "others", "others_size12", "pred_group_precision", "n_pred_groups")}, f"{(time.time() - t0) / 60:.1f} 分", flush=True)
    summ = {}
    for name in ("G1", "G2"):
        r = S[name]; keys = [k for k in r[0] if not isinstance(r[0][k], (list, dict))]
        summ[name] = dict(mean={k: round(float(np.mean([x[k] for x in r])), 3) for k in keys}, range_all=round(max(x["all"] for x in r) - min(x["all"] for x in r), 3),
                          stroke_PR_mean={c: [round(float(np.mean([x[c + "_stroke_PR"][j] for x in r])), 3) for j in (0, 1)] for c in ("eye", "brow", "nose", "mouth", "ear")},
                          by_pose_mean={y: round(float(np.mean([x["by_pose"][y] for x in r])), 3) for y in ("front", "oblique", "profile")}, seeds=r)
        summ[name]["delta_all_vs_T2"] = round(summ[name]["mean"]["all"] - 0.321, 3)
        summ[name]["delta_precision_vs_T2"] = round(summ[name]["mean"]["pred_group_precision"] - 0.204, 3)
    json.dump(summ, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for name in ("G1", "G2"):
        s = summ[name]; print(name, s["mean"], s["range_all"], s["stroke_PR_mean"], s["by_pose_mean"], s["delta_all_vs_T2"], s["delta_precision_vs_T2"])
    print("->", out)


if __name__ == "__main__":
    main()
