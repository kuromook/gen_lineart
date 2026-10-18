#!/usr/bin/env python
"""実験 H16(2026-10-18 事前登録): H11 の T2 を、学ぶ側の顔を 25% / 50% / 75% に減らして学ぶ(各 3 回、周の数は 200 / 割合)。学習用だけ。

100% は H15 の P0 を使う。出力: summary.json、check.json
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

FRACS, REPS = (0.25, 0.5, 0.75), (0, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h16_set_scale_20261018")
    a = ap.parse_args()
    out = G.ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda"); t0 = time.time()
    faces = H.load_faces(H10.MAX_ORDER, H10.N_LINES)
    fold, _ = H.folds_of(faces)
    rng = np.random.default_rng(20261018); sub = {}
    for frac in FRACS:
        for rep in REPS:
            for k in sorted(set(fold.tolist())):
                tr = np.flatnonzero(fold != k)
                sub[(frac, rep, k)] = np.sort(rng.choice(tr, int(round(frac * len(tr))), replace=False))
    ok1 = all(not np.any(fold[v] == k) for (fr, rp, k), v in sub.items())
    ok2 = all(len(v) == int(round(fr * int((fold != k).sum()))) for (fr, rp, k), v in sub.items())
    chk = dict(n_faces=len(faces), no_overlap=bool(ok1), sizes_ok=bool(ok2),
               n_train_per_fold={str(fr): [len(sub[(fr, 0, k)]) for k in sorted(set(fold.tolist()))] for fr in FRACS})
    chk["passed"] = bool(ok1 and ok2)
    json.dump(chk, open(out / "check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("道具確認:", "合格" if chk["passed"] else "不合格", flush=True)
    if not chk["passed"]:
        return
    curve = {}
    for frac in FRACS:
        M.EPOCHS = int(round(200 / frac)); res = []
        for rep in REPS:
            preds = [None] * len(faces)
            for k in sorted(set(fold.tolist())):
                te = np.flatnonzero(fold == k)
                p, *_ = M.train_eval(faces, sub[(frac, rep, k)], te, True, 1000 + rep * 10 + k, dev)
                for i in te:
                    preds[i] = p[i]
            e = H10.part_eval(faces, preds); res.append(e)
            print(frac, rep, {k: e[k] for k in ("all", "eye", "others", "others_size12", "pred_group_precision")}, f"{(time.time() - t0) / 60:.1f} 分", flush=True)
        keys = [k for k in res[0] if not isinstance(res[0][k], (list, dict))]
        curve[str(frac)] = dict(mean={k: round(float(np.mean([r[k] for r in res])), 3) for k in keys},
                                range_all=round(max(r["all"] for r in res) - min(r["all"] for r in res), 3), epochs=M.EPOCHS, reps=res)
    P0 = json.load(open(G.ROOT / "results/h15_true_pose_20261018/summary.json"))["P0"]
    curve["1.0"] = dict(mean=P0["mean"], range_all=P0["range_all"], epochs=200, source="H15 P0")
    S = dict(curve=curve, delta_100_minus_50=round(curve["1.0"]["mean"]["all"] - curve["0.5"]["mean"]["all"], 3),
             delta_100_minus_75=round(curve["1.0"]["mean"]["all"] - curve["0.75"]["mean"]["all"], 3),
             delta_others_100_minus_50=round(curve["1.0"]["mean"]["others"] - curve["0.5"]["mean"]["others"], 3),
             delta_eye_100_minus_50=round(curve["1.0"]["mean"]["eye"] - curve["0.5"]["mean"]["eye"], 3))
    json.dump(S, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    for fr in ("0.25", "0.5", "0.75", "1.0"):
        print(fr, curve[fr]["mean"], curve[fr]["range_all"])
    print({k: v for k, v in S.items() if k != "curve"}); print("->", out)


if __name__ == "__main__":
    main()
