#!/usr/bin/env python
"""実験 H11(2026-10-16 事前登録): 顔の中の線を全部同時に見る(transformer の符号器)。学習用だけ。

T1 = H7 の B の特徴、T2 = T1 + 線の 16 点の座標。左右反転と枠のずれで水増し。乱数 3 通り。
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

EPOCHS, BATCH, D = 200, 16, 64
SEEDS = (0, 1, 2)


def flipped(f):
    g = dict(f); p = f["pts"].copy(); x0, _, x1, _ = f["box"]; p[..., 0] = x0 + x1 - p[..., 0]; g["pts"] = p
    return g


def jittered(f, rng):
    x0, y0, x1, y1 = f["box"]; w, h = x1 - x0, y1 - y0
    s = 1 + rng.uniform(-0.05, 0.05); dx, dy = rng.uniform(-0.05, 0.05, 2) * (w, h)
    cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
    g = dict(f); g["box"] = (cx - s * w / 2, cy - s * h / 2, cx + s * w / 2, cy + s * h / 2); g["s"] = f["s"] * s
    return g


def feats(f, raw):
    X = H.feats_B(f)
    if raw:
        x0, y0, x1, y1 = f["box"]
        P = (f["pts"] - [(x0 + x1) / 2, (y0 + y1) / 2]) / [x1 - x0, y1 - y0]
        X = np.column_stack([X, P.reshape(len(P), -1)])
    return X


class Net(torch.nn.Module):
    def __init__(self, din):
        super().__init__()
        self.inp = torch.nn.Linear(din, D)
        layer = torch.nn.TransformerEncoderLayer(D, 4, 128, 0.1, batch_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, 3)
        self.out = torch.nn.Linear(D, len(H.CLASSES))

    def forward(self, x, pad):
        return self.out(self.enc(self.inp(x), src_key_padding_mask=pad))


def batchify(Xs, ys, dev):
    n = max(len(x) for x in Xs); d = Xs[0].shape[1]
    X = np.zeros((len(Xs), n, d), np.float32); Y = np.full((len(Xs), n), -100, np.int64); pad = np.ones((len(Xs), n), bool)
    for i, (x, y) in enumerate(zip(Xs, ys)):
        X[i, :len(x)] = x; Y[i, :len(x)] = y; pad[i, :len(x)] = False
    return torch.tensor(X, device=dev), torch.tensor(Y, device=dev), torch.tensor(pad, device=dev)


def train_eval(faces, tr, te, raw, seed, dev, labels=None):
    rng = np.random.default_rng(20261016 + seed); torch.manual_seed(20261016 + seed)
    X0 = np.vstack([feats(faces[i], raw) for i in tr]); mu, sd = X0.mean(0), X0.std(0) + 1e-6
    net = Net(X0.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // BATCH)):
            Xs, ys = [], []
            for i in b:
                f = faces[i]
                if rng.random() < 0.5:
                    f = flipped(f)
                f = jittered(f, rng)
                Xs.append((feats(f, raw) - mu) / sd); ys.append(faces[i]["y"] if labels is None else labels[i])
            X, Y, pad = batchify(Xs, ys, dev)
            opt.zero_grad(); torch.nn.functional.cross_entropy(net(X, pad).reshape(-1, len(H.CLASSES)), Y.reshape(-1)).backward(); opt.step()
    net.eval(); preds = {}
    with torch.no_grad():
        for i in te:
            x = torch.tensor(((feats(faces[i], raw) - mu) / sd)[None], dtype=torch.float32, device=dev)
            preds[i] = net(x, torch.zeros((1, x.shape[1]), dtype=torch.bool, device=dev))[0].argmax(1).cpu().numpy()
    return preds, net, mu, sd


def run(faces, fold, raw, seed, dev, labels=None):
    preds = [None] * len(faces)
    for k in sorted(set(fold.tolist())):
        tr = np.flatnonzero(fold != k); te = np.flatnonzero(fold == k)
        p, *_ = train_eval(faces, tr, te, raw, seed * 10 + k, dev, labels)
        for i in te:
            preds[i] = p[i]
    return preds


def by_pose(faces, preds):
    out = {}
    for y in ("front", "oblique", "profile"):
        idx = [i for i, f in enumerate(faces) if f["pose"]["yaw"] == y]
        out[y] = dict(n_faces=len(idx), all=H.rate(H.group_eval([faces[i] for i in idx], [preds[i] for i in idx], H10.TAU)[0])[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h11_set_model_20261016")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    faces = H.load_faces(H10.MAX_ORDER, H10.N_LINES)
    fold, g2f = H.folds_of(faces)
    t0 = time.time()

    ok1 = all(len({fold[i] for i in range(len(faces)) if faces[i]["group"] == g}) == 1 for g in g2f)
    f = faces[0]; A, B = H.feats_B(f), H.feats_B(flipped(f))
    exp = A.copy(); exp[:, 4] = -A[:, 4]; exp[:, 9] = 1 - A[:, 9]; exp[:, 11] = 1 - A[:, 13]; exp[:, 13] = 1 - A[:, 11]
    ok3 = bool(np.allclose(B, exp, atol=1e-6))
    tr = np.flatnonzero(fold != 0); te = np.flatnonzero(fold == 0)[:5]
    p, net, mu, sd = train_eval(faces, tr, te, False, 99, dev)
    ok2 = True
    with torch.no_grad():
        for i in te:
            perm = np.random.default_rng(i).permutation(len(faces[i]["y"]))
            x = torch.tensor(((feats(faces[i], False) - mu) / sd)[perm][None], dtype=torch.float32, device=dev)
            q = net(x, torch.zeros((1, x.shape[1]), dtype=torch.bool, device=dev))[0].argmax(1).cpu().numpy()
            ok2 &= bool(np.array_equal(q, p[i][perm]))
    rng = np.random.default_rng(20261016)
    shuf = {i: rng.permutation(fc["y"]) for i, fc in enumerate(faces)}
    neg = H.rate(H.group_eval(faces, run(faces, fold, False, 7, dev, labels=shuf), H10.TAU)[0])[0]
    chk = dict(n_faces=len(faces), folds_disjoint=bool(ok1), order_invariant=ok2, flip_features_ok=ok3, negative_recovered=neg, minutes=round((time.time() - t0) / 60, 1))
    chk["passed"] = bool(ok1 and ok2 and ok3 and neg <= 0.05)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return

    S = {}
    for name, raw in (("T1", False), ("T2", True)):
        res = []
        for seed in SEEDS:
            pr = run(faces, fold, raw, seed, dev)
            e = H10.part_eval(faces, pr); e["by_pose"] = by_pose(faces, pr); res.append(e)
            print(name, seed, {k: e[k] for k in ("all", "eye", "others", "others_size12", "pred_group_precision")}, f"{(time.time() - t0) / 60:.1f} 分", flush=True)
        keys = [k for k in res[0] if not isinstance(res[0][k], (list, dict))]
        S[name] = dict(seeds=res, mean={k: round(float(np.mean([r[k] for r in res])), 3) for k in keys},
                       range_all=round(max(r["all"] for r in res) - min(r["all"] for r in res), 3),
                       stroke_PR_mean={c: [round(float(np.mean([r[c + "_stroke_PR"][j] for r in res])), 3) for j in (0, 1)]
                                       for c in ("eye", "brow", "nose", "mouth", "ear")},
                       by_pose_mean={y: round(float(np.mean([r["by_pose"][y]["all"] for r in res])), 3) for y in ("front", "oblique", "profile")})
    S["B_reference_mean_all"] = 0.284
    S["delta_T1"] = round(S["T1"]["mean"]["all"] - 0.284, 3); S["delta_T2"] = round(S["T2"]["mean"]["all"] - 0.284, 3)
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for n in ("T1", "T2"):
        print(n, S[n]["mean"], S[n]["range_all"], S[n]["stroke_PR_mean"], S[n]["by_pose_mean"])
    print("差", S["delta_T1"], S["delta_T2"]); print("->", out)


if __name__ == "__main__":
    main()
