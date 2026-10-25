#!/usr/bin/env python
"""H19 の道具確認 3(線の並び順を入れ替えても Q1 の予測が変わらない)だけを、本体と同じ作り・種(99)で走らせ直す。
本体(h19_pixels.py)がその後の段で止まり、check.json に書かれなかったため(2026-10-23)。
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_server as G                                               # noqa: E402
import h7_stroke_parts as H                                            # noqa: E402
import h10_scale_pose as H10                                           # noqa: E402
import h11_set_model as M                                              # noqa: E402
import h19_pixels as P                                                 # noqa: E402


def main():
    out = G.ROOT / "results/h19_pixels_20261023"; dev = torch.device("cuda")
    faces = H.load_faces(P.MAX_ORDER, P.N_LINES)
    fold, _, _ = P.folds(faces)
    is_old = np.array([f["order"] <= H10.MAX_ORDER for f in faces])
    PT = [P.patches(f) for f in faces]
    tr = np.flatnonzero(fold != 0); te = np.flatnonzero((fold == 0) & is_old)[:5]
    p, net, mu, sd = P.train_eval(faces, PT, tr, te, 99, dev); ok = True; diff = []
    with torch.no_grad():
        for i in te:
            perm = np.random.default_rng(i).permutation(len(faces[i]["y"]))
            x = torch.tensor(((M.feats(faces[i], True) - mu) / sd)[perm][None], dtype=torch.float32, device=dev)
            q = net(x, torch.zeros((1, x.shape[1]), dtype=torch.bool, device=dev), P.batch_patches([PT[i][perm]], x.shape[1], dev))[0].argmax(1).cpu().numpy()
            ok &= bool(np.array_equal(q, p[i][perm])); diff.append(int((q != p[i][perm]).sum()))
    chk = dict(order_invariant=ok, n_lines_changed=diff)
    json.dump(chk, open(out / "check_order.json", "w"), indent=1, ensure_ascii=False); print(chk, flush=True)


if __name__ == "__main__":
    main()
