#!/usr/bin/env python
"""実験 H19(2026-10-23 事前登録): 線の真ん中の周りの画の切り抜き(2 チャンネル、32 × 32)を、小さな畳み込みで網の入力に足す。学習用だけ。

Q0 = H17 の作り(T2)をもう一度、Q1 = Q0 + 切り抜き。元の 193 顔の中で数える(260 顔全体も参考に)。
出力: summary.json、check.json、patches_check.png
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_server as G                                               # noqa: E402
import h7_stroke_parts as H                                            # noqa: E402
import h10_scale_pose as H10                                           # noqa: E402
import h11_set_model as M                                              # noqa: E402

N_LINES, MAX_ORDER = 518, 400
PATCH, SIDE_K = 32, 0.25


def folds(faces):
    """H17 と同じ: 元の 193 顔の分割を保ち、新しい顔は同じシリーズ群の分割へ"""
    old = H.load_faces(H10.MAX_ORDER, H10.N_LINES)
    fold_old, g2f = H.folds_of(old)
    load = np.bincount(fold_old, minlength=H.NFOLD).tolist()
    for f in faces:
        if f["group"] not in g2f:
            k = int(np.argmin(load)); g2f[f["group"]] = k
        load[g2f[f["group"]]] += 0 if f["order"] <= H10.MAX_ORDER else 1
    return np.array([g2f[f["group"]] for f in faces]), old, fold_old


def patches(f, img=None, flip_img=False):
    """返る: (n, 2, 32, 32) float32。img を渡すとそれを使う(反転の確認用)"""
    if img is None:
        img = G.panel_img(f["panel"])
    g = 1.0 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    W = g.shape[1]
    pts = f["pts"].copy()
    if flip_img:
        pts[..., 0] = W - 1 - pts[..., 0]
    side = SIDE_K * f["s"]; k = PATCH / side
    out = np.zeros((len(pts), 2, PATCH, PATCH), np.float32)
    for i, p in enumerate(pts):
        c = (p[7] + p[8]) / 2
        A = np.array([[k, 0, PATCH / 2 - 0.5 - k * c[0]], [0, k, PATCH / 2 - 0.5 - k * c[1]]], np.float32)
        out[i, 0] = cv2.warpAffine(g, A, (PATCH, PATCH), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR, borderValue=0)
        m = np.zeros((PATCH, PATCH), np.uint8)
        q = ((p - c) * k + PATCH / 2 - 0.5).round().astype(np.int32)
        cv2.polylines(m, [q], False, 1, 2)
        out[i, 1] = m
    return out


class Net(M.Net):
    def __init__(self, din):
        super().__init__(din + 32)
        self.cnn = torch.nn.Sequential(torch.nn.Conv2d(2, 16, 3, 2, 1), torch.nn.ReLU(), torch.nn.Conv2d(16, 32, 3, 2, 1), torch.nn.ReLU(),
                                       torch.nn.Conv2d(32, 32, 3, 2, 1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())

    def forward(self, x, pad, pt):
        B, N = x.shape[:2]
        z = self.cnn(pt.reshape(B * N, 2, PATCH, PATCH)).reshape(B, N, 32)
        return self.out(self.enc(self.inp(torch.cat([x, z], -1)), src_key_padding_mask=pad))


def batch_patches(Ps, n, dev):
    out = np.zeros((len(Ps), n, 2, PATCH, PATCH), np.float32)
    for i, p in enumerate(Ps):
        out[i, :len(p)] = p
    return torch.tensor(out, device=dev)


def train_eval(faces, PT, tr, te, seed, dev):
    """M.train_eval と同じ(乱数の使い方も同じ)。切り抜きを足し、反転では切り抜きも左右反転する"""
    rng = np.random.default_rng(20261016 + seed); torch.manual_seed(20261016 + seed)
    X0 = np.vstack([M.feats(faces[i], True) for i in tr]); mu, sd = X0.mean(0), X0.std(0) + 1e-6
    net = Net(X0.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(M.EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // M.BATCH)):
            Xs, ys, Ps = [], [], []
            for i in b:
                f = faces[i]; p = PT[i]
                if rng.random() < 0.5:
                    f = M.flipped(f); p = p[..., ::-1]
                f = M.jittered(f, rng)
                Xs.append((M.feats(f, True) - mu) / sd); ys.append(faces[i]["y"]); Ps.append(p)
            X, Y, pad = M.batchify(Xs, ys, dev)
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(X, pad, batch_patches(Ps, X.shape[1], dev)).reshape(-1, len(H.CLASSES)), Y.reshape(-1)).backward(); opt.step()
    net.eval(); preds = {}
    with torch.no_grad():
        for i in te:
            x = torch.tensor(((M.feats(faces[i], True) - mu) / sd)[None], dtype=torch.float32, device=dev)
            preds[i] = net(x, torch.zeros((1, x.shape[1]), dtype=torch.bool, device=dev), batch_patches([PT[i]], x.shape[1], dev))[0].argmax(1).cpu().numpy()
    return preds, net, mu, sd


def montage(faces, PT, path):
    rows = []
    for cname in ("brow", "ear", "none"):
        c = H.CLASSES.index(cname); tiles = []
        for i, f in enumerate(faces):
            for j in np.flatnonzero(f["y"] == c)[:1]:
                p = PT[i][j]; t = np.dstack([1 - p[0]] * 3); t[p[1] > 0] = (0, 0, 1)
                tiles.append(cv2.resize((t * 255).astype(np.uint8), (96, 96), interpolation=cv2.INTER_NEAREST))
            if len(tiles) == 12:
                break
        row = np.hstack([cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(200, 200, 200)) for t in tiles])
        rows.append(cv2.copyMakeBorder(row, 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        cv2.putText(rows[-1], cname, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(path), np.vstack(rows))


def summarize(res):
    keys = [k for k in res[0] if not isinstance(res[0][k], (list, dict))]
    return dict(mean={k: round(float(np.mean([r[k] for r in res])), 3) for k in keys}, range_all=round(max(r["all"] for r in res) - min(r["all"] for r in res), 3),
                stroke_PR_mean={c: [round(float(np.mean([r[c + "_stroke_PR"][j] for r in res])), 3) for j in (0, 1)] for c in ("eye", "brow", "nose", "mouth", "ear")},
                seeds=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h19_pixels_20261023")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda"); t0 = time.time()
    faces = H.load_faces(MAX_ORDER, N_LINES)
    fold, old, fold_old = folds(faces)
    is_old = np.array([f["order"] <= H10.MAX_ORDER for f in faces]); oi = np.flatnonzero(is_old)
    assert np.array_equal(fold[is_old], fold_old)
    PT = [patches(f) for f in faces]
    print("切り抜き", f"{(time.time() - t0) / 60:.1f} 分", flush=True)
    # 道具確認 1: 画像を反転してから切る = 切り抜きを反転
    errs = []
    for f in faces[:5]:
        img = G.panel_img(f["panel"])
        a1 = patches(f, img[:, ::-1].copy(), flip_img=True); a0 = PT[faces.index(f)][..., ::-1]
        errs.append(float(np.abs(a1[:, 0] - a0[:, 0]).mean()))
    ok1 = max(errs) <= 0.02
    montage(faces, PT, out / "patches_check.png")
    chk = dict(n_faces=len(faces), n_old=int(is_old.sum()), flip_patch_mean_abs_err=[round(e, 4) for e in errs], flip_ok=ok1)
    if a.check_only:
        json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False); print(chk); return
    # 道具確認 3: 並び順
    tr = np.flatnonzero(fold != 0); te = np.flatnonzero((fold == 0) & is_old)[:5]
    p, net, mu, sd = train_eval(faces, PT, tr, te, 99, dev); ok3 = True
    with torch.no_grad():
        for i in te:
            perm = np.random.default_rng(i).permutation(len(faces[i]["y"]))
            x = torch.tensor(((M.feats(faces[i], True) - mu) / sd)[perm][None], dtype=torch.float32, device=dev)
            q = net(x, torch.zeros((1, x.shape[1]), dtype=torch.bool, device=dev), batch_patches([PT[i][perm]], x.shape[1], dev))[0].argmax(1).cpu().numpy()
            ok3 &= bool(np.array_equal(q, p[i][perm]))
    chk["order_invariant"] = ok3

    def run(seed, PTs, model):
        preds = [None] * len(faces)
        for k in range(H.NFOLD):
            tr = np.flatnonzero(fold != k); te = np.flatnonzero(fold == k)
            pr = M.train_eval(faces, tr, te, True, seed * 10 + k, dev)[0] if model == "Q0" else train_eval(faces, PTs, tr, te, seed * 10 + k, dev)[0]
            for i in te:
                preds[i] = pr[i]
        return preds

    S = {"Q0": [], "Q1": [], "Q0_all260": [], "Q1_all260": []}
    allp = {}
    for seed in M.SEEDS:
        for name in ("Q0", "Q1"):
            pr = run(seed, PT, name); allp[(name, seed)] = pr
            e = H10.part_eval([faces[i] for i in oi], [pr[i] for i in oi]); S[name].append(e)
            S[name + "_all260"].append(H10.part_eval(faces, pr))
            print(name, seed, {k: e[k] for k in ("all", "eye", "others", "brow", "ear", "pred_group_precision")}, f"{(time.time() - t0) / 60:.1f} 分", flush=True)
    # 道具確認 4: 切り抜きを顔の間でかき混ぜる(乱数の組 0 だけ)
    rng = np.random.default_rng(20261023); pool = np.concatenate(PT); sh = []
    for P in PT:
        sh.append(pool[rng.choice(len(pool), len(P), replace=False)])
    neg = H10.part_eval([faces[i] for i in oi], [run(0, sh, "Q1")[i] for i in oi])["all"]
    chk["negative_shuffled_patches_all"] = neg; chk["negative_minus_Q0_seed0"] = round(neg - S["Q0"][0]["all"], 3)
    chk["passed"] = bool(ok1 and ok3 and abs(chk["negative_minus_Q0_seed0"]) <= 0.03)
    chk["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格(結果は summary に残すが、読む前に相談)", flush=True)
    summ = {k: summarize(v) for k, v in S.items()}
    summ["delta_Q1_minus_Q0"] = round(summ["Q1"]["mean"]["all"] - summ["Q0"]["mean"]["all"], 3)
    summ["delta_seedwise"] = [round(a["all"] - b["all"], 3) for a, b in zip(S["Q1"], S["Q0"])]
    summ["delta_all260"] = round(summ["Q1_all260"]["mean"]["all"] - summ["Q0_all260"]["mean"]["all"], 3)
    summ["passed_checks"] = chk["passed"]
    json.dump(summ, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    if chk["passed"]:
        for n in ("Q0", "Q1"):
            print(n, summ[n]["mean"], summ[n]["range_all"], summ[n]["stroke_PR_mean"])
        print("差", summ["delta_Q1_minus_Q0"], summ["delta_seedwise"], "260 顔", summ["delta_all260"])
    print("->", out)


if __name__ == "__main__":
    main()
