#!/usr/bin/env python
"""H19 の道具確認 4(切り抜きを顔の間でかき混ぜる、乱数の組 0)だけを、本体と同じ種で走らせ直す。
本体(h19_pixels.py)がこの段で 6 時間以上進まなかったため(2026-10-23)。分割ごとに経過を出す。
比べる Q0 の組 0 は本体の記録(results/h19_run.log の「Q0 0」の行、0.348)。
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_server as G                                               # noqa: E402
import h7_stroke_parts as H                                            # noqa: E402
import h10_scale_pose as H10                                           # noqa: E402
import h19_pixels as P                                                 # noqa: E402

Q0_SEED0 = 0.348


def main():
    out = G.ROOT / "results/h19_pixels_20261023"; dev = torch.device("cuda"); t0 = time.time()
    faces = H.load_faces(P.MAX_ORDER, P.N_LINES)
    fold, _, fold_old = P.folds(faces)
    is_old = np.array([f["order"] <= H10.MAX_ORDER for f in faces]); oi = np.flatnonzero(is_old)
    PT = [P.patches(f) for f in faces]
    rng = np.random.default_rng(20261023); pool = np.concatenate(PT); sh = []
    for p in PT:
        sh.append(pool[rng.choice(len(pool), len(p), replace=False)])
    preds = [None] * len(faces)
    for k in range(H.NFOLD):
        tr = np.flatnonzero(fold != k); te = np.flatnonzero(fold == k)
        pr = P.train_eval(faces, sh, tr, te, k, dev)[0]
        for i in te:
            preds[i] = pr[i]
        print("分割", k, f"{(time.time() - t0) / 60:.1f} 分", flush=True)
    e = H10.part_eval([faces[i] for i in oi], [preds[i] for i in oi])
    chk = dict(negative_shuffled_patches_all=e["all"], Q0_seed0_from_log=Q0_SEED0, negative_minus_Q0_seed0=round(e["all"] - Q0_SEED0, 3),
               negative_parts={k: e[k] for k in ("eye", "brow", "ear", "others")}, minutes=round((time.time() - t0) / 60, 1))
    chk["negative_ok"] = abs(chk["negative_minus_Q0_seed0"]) <= 0.03
    json.dump(chk, open(out / "check_negative.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(chk, ensure_ascii=False)); print("->", out / "check_negative.json", flush=True)


if __name__ == "__main__":
    main()
