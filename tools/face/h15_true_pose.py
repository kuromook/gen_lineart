#!/usr/bin/env python
"""実験 H15(2026-10-18 事前登録): H11 の T2 に正解の向き(左右 5 択 + 上下 3 択の one-hot)を全部の線に足す。上限の確認。学習用だけ。

P0 = H11 の T2 を同じ乱数の組でもう一度、P1 = P0 + 正解の向き。
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

SWAP = {"left": "right", "right": "left"}


def pose_vec(f):
    p = f["pose"]
    return np.concatenate([np.eye(5)[H10.yaw_index(p)], np.eye(3)[H10.PITCH3.index(p["pitch"])]])


def flipped(f):
    g = M.flipped(f); p = dict(f["pose"])
    if p.get("dir") in SWAP:
        p["dir"] = SWAP[p["dir"]]
    g["pose"] = p
    return g


def feats(f):
    X = M.feats(f, True)
    return np.column_stack([X, np.tile(pose_vec(f), (len(X), 1))])


def train_eval(faces, tr, te, seed, dev, labels=None):
    """M.train_eval と同じ(乱数の使い方も同じ)。特徴に向きを足し、反転で向きの左右も入れ替える"""
    rng = np.random.default_rng(20261016 + seed); torch.manual_seed(20261016 + seed)
    X0 = np.vstack([feats(faces[i]) for i in tr]); mu, sd = X0.mean(0), X0.std(0) + 1e-6
    net = M.Net(X0.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(M.EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // M.BATCH)):
            Xs, ys = [], []
            for i in b:
                f = faces[i]
                if rng.random() < 0.5:
                    f = flipped(f)
                f = M.jittered(f, rng)
                Xs.append((feats(f) - mu) / sd); ys.append(faces[i]["y"] if labels is None else labels[i])
            X, Y, pad = M.batchify(Xs, ys, dev)
            opt.zero_grad(); torch.nn.functional.cross_entropy(net(X, pad).reshape(-1, len(H.CLASSES)), Y.reshape(-1)).backward(); opt.step()
    net.eval(); preds = {}
    with torch.no_grad():
        for i in te:
            x = torch.tensor(((feats(faces[i]) - mu) / sd)[None], dtype=torch.float32, device=dev)
            preds[i] = net(x, torch.zeros((1, x.shape[1]), dtype=torch.bool, device=dev))[0].argmax(1).cpu().numpy()
    return preds, net, mu, sd


def run(faces, fold, seed, dev, labels=None):
    preds = [None] * len(faces)
    for k in sorted(set(fold.tolist())):
        tr = np.flatnonzero(fold != k); te = np.flatnonzero(fold == k)
        p, *_ = train_eval(faces, tr, te, seed * 10 + k, dev, labels)
        for i in te:
            preds[i] = p[i]
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h15_true_pose_20261018")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda"); t0 = time.time()
    faces = H.load_faces(H10.MAX_ORDER, H10.N_LINES)
    assert all(f["pose"] for f in faces)
    fold, _ = H.folds_of(faces)

    # 道具確認 1: 向きの特徴の左右反転
    perm5 = [0, 2, 1, 4, 3]
    ok1 = all(np.array_equal(pose_vec(flipped(f))[:5], pose_vec(f)[:5][perm5]) and np.array_equal(pose_vec(flipped(f))[5:], pose_vec(f)[5:]) for f in faces)
    # 道具確認 2: 並び順
    tr = np.flatnonzero(fold != 0); te = np.flatnonzero(fold == 0)[:5]
    p, net, mu, sd = train_eval(faces, tr, te, 99, dev)
    ok2 = True
    with torch.no_grad():
        for i in te:
            perm = np.random.default_rng(i).permutation(len(faces[i]["y"]))
            x = torch.tensor(((feats(faces[i]) - mu) / sd)[perm][None], dtype=torch.float32, device=dev)
            q = net(x, torch.zeros((1, x.shape[1]), dtype=torch.bool, device=dev))[0].argmax(1).cpu().numpy()
            ok2 &= bool(np.array_equal(q, p[i][perm]))
    # 道具確認 3: かき混ぜ
    rng = np.random.default_rng(20261018)
    shuf = {i: rng.permutation(fc["y"]) for i, fc in enumerate(faces)}
    neg = H.rate(H.group_eval(faces, run(faces, fold, 7, dev, labels=shuf), H10.TAU)[0])[0]
    chk = dict(n_faces=len(faces), flip_pose_ok=bool(ok1), order_invariant=ok2, negative_recovered=neg, minutes=round((time.time() - t0) / 60, 1))
    chk["passed"] = bool(ok1 and ok2 and neg <= 0.05)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return

    S = {}
    for name in ("P0", "P1"):
        res = []
        for seed in M.SEEDS:
            pr = M.run(faces, fold, True, seed, dev) if name == "P0" else run(faces, fold, seed, dev)
            e = H10.part_eval(faces, pr); e["by_pose"] = M.by_pose(faces, pr); res.append(e)
            print(name, seed, {k: e[k] for k in ("all", "eye", "others", "others_size12", "pred_group_precision")},
                  {y: v["all"] for y, v in e["by_pose"].items()}, f"{(time.time() - t0) / 60:.1f} 分", flush=True)
        keys = [k for k in res[0] if not isinstance(res[0][k], (list, dict))]
        S[name] = dict(seeds=res, mean={k: round(float(np.mean([r[k] for r in res])), 3) for k in keys},
                       range_all=round(max(r["all"] for r in res) - min(r["all"] for r in res), 3),
                       stroke_PR_mean={c: [round(float(np.mean([r[c + "_stroke_PR"][j] for r in res])), 3) for j in (0, 1)]
                                       for c in ("eye", "brow", "nose", "mouth", "ear")},
                       by_pose_mean={y: round(float(np.mean([r["by_pose"][y]["all"] for r in res])), 3) for y in ("front", "oblique", "profile")},
                       by_pose_n={y: res[0]["by_pose"][y]["n_faces"] for y in ("front", "oblique", "profile")})
    S["delta_all"] = round(S["P1"]["mean"]["all"] - S["P0"]["mean"]["all"], 3)
    S["delta_by_pose"] = {y: round(S["P1"]["by_pose_mean"][y] - S["P0"]["by_pose_mean"][y], 3) for y in ("front", "oblique", "profile")}
    S["delta_seedwise"] = [round(a["all"] - b["all"], 3) for a, b in zip(S["P1"]["seeds"], S["P0"]["seeds"])]
    S["P0_minus_H11_T2"] = round(S["P0"]["mean"]["all"] - 0.321, 3)
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for n in ("P0", "P1"):
        print(n, S[n]["mean"], S[n]["range_all"], S[n]["stroke_PR_mean"], S[n]["by_pose_mean"])
    print("差", S["delta_all"], S["delta_by_pose"], S["delta_seedwise"], "P0−T2", S["P0_minus_H11_T2"]); print("->", out)


if __name__ == "__main__":
    main()
