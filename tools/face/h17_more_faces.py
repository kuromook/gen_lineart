#!/usr/bin/env python
"""実験 H17(2026-10-18 事前登録): order 0〜400 の学習用 260 顔で H11 の T2 を学び、元の 193 顔の中だけで数える。学習用だけ。

分割は元の 193 顔の分割(シリーズ群 → 分割)を保ち、新しい顔は同じシリーズ群の分割に入れる(新しいシリーズ群は顔の少ない分割へ)。
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

N_LINES, MAX_ORDER = 518, 400
P0_ALL = 0.319


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h17_more_faces_20261023")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda"); t0 = time.time()
    old = H.load_faces(H10.MAX_ORDER, H10.N_LINES)
    fold_old, g2f = H.folds_of(old)
    faces = H.load_faces(MAX_ORDER, N_LINES)
    load = np.bincount(fold_old, minlength=H.NFOLD).tolist(); new_groups = []
    for f in faces:
        if f["group"] not in g2f:
            k = int(np.argmin(load)); g2f[f["group"]] = k; new_groups.append(f["group"])
        load[g2f[f["group"]]] += 0 if f["order"] <= H10.MAX_ORDER else 1
    fold = np.array([g2f[f["group"]] for f in faces])
    is_old = np.array([f["order"] <= H10.MAX_ORDER for f in faces])
    # 道具確認: 元の顔が同じ並び・同じ目印・同じ分割で読めている、学ぶ側と当てる側がシリーズ群で重ならない
    ok1 = bool(is_old.sum() == len(old) and all(np.array_equal(f["y"], o["y"]) and np.array_equal(f["gid"], o["gid"]) and f["order"] == o["order"]
                                                for f, o in zip([f for f in faces if f["order"] <= H10.MAX_ORDER], old)))
    ok2 = bool(np.array_equal(fold[is_old], fold_old))
    ok3 = all(len({fold[i] for i in range(len(faces)) if faces[i]["group"] == g}) == 1 for g in set(f["group"] for f in faces))
    chk = dict(n_faces=len(faces), n_old=int(is_old.sum()), n_new=int((~is_old).sum()), new_series_groups=len(new_groups),
               per_fold_all=np.bincount(fold).tolist(), per_fold_new=np.bincount(fold[~is_old], minlength=H.NFOLD).tolist(),
               old_faces_identical=ok1, old_folds_identical=ok2, folds_disjoint=bool(ok3))
    chk["passed"] = bool(ok1 and ok2 and ok3)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return
    oi = np.flatnonzero(is_old); ofaces = [faces[i] for i in oi]

    def run(seed, frac=None, rep=0):
        preds = [None] * len(faces)
        for k in range(H.NFOLD):
            tr = np.flatnonzero(fold != k); te = np.flatnonzero((fold == k) & is_old)
            if frac is not None:
                tr = np.sort(np.random.default_rng(20261023 + 100 * rep + k).choice(tr, int(round(frac * len(tr))), replace=False))
            p, *_ = M.train_eval(faces, tr, te, True, seed * 10 + k, dev)
            for i in te:
                preds[i] = p[i]
        return [preds[i] for i in oi]

    res = []
    M.EPOCHS = 200
    for seed in M.SEEDS:
        pr = run(seed); e = H10.part_eval(ofaces, pr); e["by_pose"] = M.by_pose(ofaces, pr); res.append(e)
        print("all260", seed, {k: e[k] for k in ("all", "eye", "others", "others_size12", "pred_group_precision")}, f"{(time.time() - t0) / 60:.1f} 分", flush=True)
    keys = [k for k in res[0] if not isinstance(res[0][k], (list, dict))]
    S = dict(full=dict(mean={k: round(float(np.mean([r[k] for r in res])), 3) for k in keys}, range_all=round(max(r["all"] for r in res) - min(r["all"] for r in res), 3),
                       stroke_PR_mean={c: [round(float(np.mean([r[c + "_stroke_PR"][j] for r in res])), 3) for j in (0, 1)] for c in ("eye", "brow", "nose", "mouth", "ear")},
                       by_pose_mean={y: round(float(np.mean([r["by_pose"][y]["all"] for r in res])), 3) for y in ("front", "oblique", "profile")}, seeds=res))
    S["delta_vs_P0"] = round(S["full"]["mean"]["all"] - P0_ALL, 3)
    S["delta_seedwise_vs_P0"] = [round(r["all"] - p, 3) for r, p in zip(res, (0.328, 0.323, 0.306))]
    print("全体", S["full"]["mean"], "P0 との差", S["delta_vs_P0"], S["delta_seedwise_vs_P0"], flush=True)
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    # 参考: 増えた全体での学ぶ側の割合の曲線(H16 と同じ作り。周の数 = 200 / 割合)
    curve = {}
    for frac in (0.5, 0.75):
        M.EPOCHS = int(round(200 / frac)); rr = []
        for rep in (0, 1, 2):
            pr = run(1000 + rep, frac, rep); rr.append(H10.part_eval(ofaces, pr)["all"])
            print("curve", frac, rep, rr[-1], f"{(time.time() - t0) / 60:.1f} 分", flush=True)
        curve[str(frac)] = dict(mean=round(float(np.mean(rr)), 3), reps=rr, epochs=M.EPOCHS)
    curve["1.0"] = dict(mean=S["full"]["mean"]["all"])
    S["curve_reference"] = curve
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print("曲線(参考)", {k: v["mean"] for k, v in curve.items()}); print("->", out)


if __name__ == "__main__":
    main()
