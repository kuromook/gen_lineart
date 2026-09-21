#!/usr/bin/env python
"""道具健全性チェック (事前登録の「道具健全性チェック」項に対応、訓練前に実行):
1. train 頻度の unigram / pos 周辺分布が Track F 実測 (7.348 / 7.563) と一致するか
2. 未学習 SetAR の test bits が word≈log2(501)≈8.97 / pos≈8.0 / scale≈5.17 付近か
3. 語原型の復号結果が破片でないか (protos.png を目視)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import N_WORDS, CORPUS, load_all, baselines, load_protos, nat_place
from train_set import SetAR, evaluate

H_CANVAS = 220


def main():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    d = load_all(0)
    panels = d["panels"]
    split = np.load(CORPUS)["split"]
    H_uni, H_pos, H_uni_f = baselines(panels, np.flatnonzero(split))
    print(f"[1] unigram(EOS なし) {H_uni:.3f}  unigram(Track F 式・EOS 混入) {H_uni_f:.3f} "
          f"(Track F 実測 7.348)   pos 周辺 {H_pos:.3f} (Track F 7.563)")
    assert abs(H_uni_f - 7.348) < 0.01 and abs(H_pos - 7.563) < 0.01, \
        "Track F 式 baseline が Track F 実測と一致しない"

    model = SetAR(n_types=d["n_types"], n_tagf=d["n_tagf"]).to(dev)
    ev = evaluate(model, panels, d["ite"][:64], d["lab"], d["tg"], dev)
    print(f"[2] 未学習 SetAR (test 64): word {ev['word_bits']:.3f} (≈8.97) "
          f"pos {ev['pos_bits']:.3f} (≈8.0) scale {ev['scale_bits']:.3f} (≈3.58)")
    # ランダム head の logit 分散で期待 CE はやや uniform より悪化する。大きく外れないことのみ確認
    assert 8.6 < ev["word_bits"] < 9.6 and 7.6 < ev["pos_bits"] < 8.9 and 3.4 < ev["scale_bits"] < 4.5

    proto, keep = load_protos(dev)
    wc = np.zeros(N_WORDS)
    for i in np.flatnonzero(split):
        w, _py, _px, _sc = panels[i]
        wc += np.bincount(w[w < N_WORDS], minlength=N_WORDS)
    ids = list(np.argsort(-wc)[:6]) + [37, 251, 409]     # 頻出6 + 乱数3
    cells = []
    for wi in ids:
        H = Wd = 56
        pts, km = nat_place(proto, keep, wi, 7, 7, 5, H * 4, Wd * 4)
        zf = H_CANVAS / (H * 4)
        c = np.full((H_CANVAS, H_CANVAS, 3), 255, np.uint8)
        for s in range(pts.shape[0]):
            if not km[s]:
                continue
            xy = np.clip(np.round(pts[s] * zf)[:, ::-1].astype(np.int32), 0, H_CANVAS - 1)
            cv2.polylines(c, [xy], False, (40, 40, 40), 1)
        cv2.putText(c, f"w{wi} n{int(km.sum())}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 200), 1, cv2.LINE_AA)
        cells.append(c)
    out = Path("results/instruments"); out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "protos.png"), np.hstack(cells))
    print(f"[3] 語原型モンタージュ -> {out/'protos.png'} (9語: 頻出上位6+乱数3。目視で破片でないこと)")


if __name__ == "__main__":
    main()
