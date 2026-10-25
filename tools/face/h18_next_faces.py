#!/usr/bin/env python
"""実験 H18(2026-10-23 事前登録): order 0〜500 の学習用で完了の顔すべてで H11 の T2 を学び、元の 193 顔の中だけで数える。学習用だけ。

H17 と同じ作り(分割は元の 193 顔の分割を保ち、新しい顔は同じシリーズ群の分割へ。新しいシリーズ群は顔の少ない分割へ)。
判断: H17(0.351)より 0.025 以上高ければ次の 100 顔。参考に H19 の Q1(画素を足した網)を同じ分割で並べる(判断には使わない)。
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
import h19_pixels as P                                                 # noqa: E402

N_LINES, MAX_ORDER = 619, 500
H17_ALL, H17_SEEDS = 0.351, (0.349, 0.351, 0.352)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h18_next_faces_20261025")
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
    # 道具確認(H17 と同じ): 元の顔が同じ並び・同じ目印・同じ分割で読めている、学ぶ側と当てる側がシリーズ群で重ならない
    ok1 = bool(is_old.sum() == len(old) and all(np.array_equal(f["y"], o["y"]) and np.array_equal(f["gid"], o["gid"]) and f["order"] == o["order"]
                                                for f, o in zip([f for f in faces if f["order"] <= H10.MAX_ORDER], old)))
    ok2 = bool(np.array_equal(fold[is_old], fold_old))
    ok3 = all(len({fold[i] for i in range(len(faces)) if faces[i]["group"] == g}) == 1 for g in set(f["group"] for f in faces))
    chk = dict(n_faces=len(faces), n_old=int(is_old.sum()), n_new=int((~is_old).sum()), n_order_401_500=int(sum(f["order"] > 400 for f in faces)),
               new_series_groups=len(new_groups), per_fold_all=np.bincount(fold).tolist(), per_fold_new=np.bincount(fold[~is_old], minlength=H.NFOLD).tolist(),
               old_faces_identical=ok1, old_folds_identical=ok2, folds_disjoint=bool(ok3))
    chk["passed"] = bool(ok1 and ok2 and ok3)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return
    oi = np.flatnonzero(is_old); ofaces = [faces[i] for i in oi]

    def run(seed, model, PT=None):
        preds = [None] * len(faces)
        for k in range(H.NFOLD):
            tr = np.flatnonzero(fold != k); te = np.flatnonzero((fold == k) & is_old)
            p = M.train_eval(faces, tr, te, True, seed * 10 + k, dev)[0] if model == "T2" else P.train_eval(faces, PT, tr, te, seed * 10 + k, dev)[0]
            for i in te:
                preds[i] = p[i]
            print(" ", model, seed, "分割", k, f"{(time.time() - t0) / 60:.1f} 分", flush=True)
        return [preds[i] for i in oi]

    def summ(res):
        keys = [k for k in res[0] if not isinstance(res[0][k], (list, dict))]
        return dict(mean={k: round(float(np.mean([r[k] for r in res])), 3) for k in keys}, range_all=round(max(r["all"] for r in res) - min(r["all"] for r in res), 3),
                    stroke_PR_mean={c: [round(float(np.mean([r[c + "_stroke_PR"][j] for r in res])), 3) for j in (0, 1)] for c in ("eye", "brow", "nose", "mouth", "ear")},
                    by_pose_mean={y: round(float(np.mean([r["by_pose"][y]["all"] for r in res])), 3) for y in ("front", "oblique", "profile")}, seeds=res)

    M.EPOCHS = 200; res = []
    for seed in M.SEEDS:
        pr = run(seed, "T2"); e = H10.part_eval(ofaces, pr); e["by_pose"] = M.by_pose(ofaces, pr); res.append(e)
        print("T2", seed, {k: e[k] for k in ("all", "eye", "others", "brow", "ear", "pred_group_precision")}, f"{(time.time() - t0) / 60:.1f} 分", flush=True)
    S = dict(T2=summ(res))
    S["delta_vs_H17"] = round(S["T2"]["mean"]["all"] - H17_ALL, 3)
    S["delta_seedwise_vs_H17"] = [round(r["all"] - p, 3) for r, p in zip(res, H17_SEEDS)]
    S["decision_more_faces"] = bool(S["delta_vs_H17"] >= 0.025)
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print("T2 全体", S["T2"]["mean"], "H17 との差", S["delta_vs_H17"], S["delta_seedwise_vs_H17"], "次の 100 顔:", S["decision_more_faces"], flush=True)
    # 参考: H19 の Q1(画素を足した網)。判断には使わない
    PT = [P.patches(f) for f in faces]; res = []
    for seed in M.SEEDS:
        pr = run(seed, "Q1", PT); e = H10.part_eval(ofaces, pr); e["by_pose"] = M.by_pose(ofaces, pr); res.append(e)
        print("Q1", seed, {k: e[k] for k in ("all", "eye", "others", "brow", "ear", "pred_group_precision")}, f"{(time.time() - t0) / 60:.1f} 分", flush=True)
    S["Q1_reference"] = summ(res)
    S["Q1_minus_T2"] = round(S["Q1_reference"]["mean"]["all"] - S["T2"]["mean"]["all"], 3)
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print("Q1(参考)", S["Q1_reference"]["mean"], "Q1 − T2", S["Q1_minus_T2"]); print("->", out, flush=True)


if __name__ == "__main__":
    main()
