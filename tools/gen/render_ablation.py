#!/usr/bin/env python
"""配置/選択の分解モンタージュ (2026-09-23 事前登録の実験2)。
列: A 真の語+真の配置 | B 真の語+生成配置 | C 生成語+真の配置 | D 生成語+生成配置。
描画はすべて「原型+代表残差」(results/protos_detail_20260923/mean_residual.npy)。
B は PosAR に真の語列を与えて配置をサンプリング、C は Stage1 の語を真の位置に置く。
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trackf"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import N_WORDS, CORPUS, load_all, nat_place       # noqa: E402
from train_codebook2 import Codebook2, topn_mask             # noqa: E402
from train_set import SetAR                                  # noqa: E402
from train_2stage import WordAR, PosAR                       # noqa: E402
from render_gen import sample_set, sample_words, sample_pos  # noqa: E402

H_CANVAS = 220
RES = Path("results/protos_detail_20260923/mean_residual.npy")


def load_rich_protos(dev):
    cb = torch.load(__import__("common").CODEBOOK, map_location=dev, weights_only=False)
    levels = tuple(cb["levels"])
    cm = Codebook2(levels=levels, coord_bins=cb["args"].get("coord_bins", 0)).to(dev)
    cm.load_state_dict(cb["model"]); cm.eval(); cm.rfsq.stages = cb["args"].get("stages", 3)
    basis = np.cumprod((1,) + levels[:-1])
    half_w = np.floor(np.array(levels) / 2)
    qq = np.zeros((N_WORDS, len(levels)), np.float32)
    for w in range(N_WORDS):
        d = (w // basis) % np.array(levels)
        qq[w] = (d - half_w) / half_w
    mean_res = torch.from_numpy(np.load(RES)).to(dev)
    with torch.no_grad():
        ex, pp, _pw, cl, _ = cm.decode(torch.from_numpy(qq).to(dev) + mean_res)
        keep = topn_mask(ex.float(), cl.float())
    return pp.float().cpu().numpy(), keep.cpu().numpy()


def render(fields, proto, keep, H, Wd, label=""):
    zf = H_CANVAS / H
    canvas = np.full((H_CANVAS + (24 if label else 0), max(int(Wd * zf), 2), 3), 255, np.uint8)
    for wi, py, px, sc in zip(*fields):
        pts, km = nat_place(proto, keep, wi, py, px, sc, H, Wd)
        for s in range(pts.shape[0]):
            if not km[s]:
                continue
            xy = np.clip(np.round(pts[s] * zf)[:, ::-1].astype(np.int32),
                         0, [canvas.shape[1] - 1, H_CANVAS - 1])
            cv2.polylines(canvas, [xy], False, (40, 40, 40), 1)
    if label:
        cv2.putText(canvas, label, (4, H_CANVAS + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 200), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_ckpt", default="results/gen_smoke_20260921/set/gen_best.pt")
    ap.add_argument("--s1_ckpt", default="results/gen_smoke_20260921/twostage/stage1/gen_best.pt")
    ap.add_argument("--s2_ckpt", default="results/gen_smoke_20260921/twostage/stage2/gen_best.pt")
    ap.add_argument("--out", default="results/ablation_20260923")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    dev = torch.device("cuda")
    rng = np.random.default_rng(777)
    d = load_all(0)
    panels, lab, tg, ite = d["panels"], d["lab"], d["tg"], d["ite"]
    z = np.load(CORPUS)
    dims = z["dims"]
    proto, keep = load_rich_protos(dev)

    ck = torch.load(a.set_ckpt, map_location=dev, weights_only=False)
    ms = SetAR().to(dev); ms.load_state_dict(ck["model"]); ms.eval()
    ck1 = torch.load(a.s1_ckpt, map_location=dev, weights_only=False)
    m1 = WordAR().to(dev); m1.load_state_dict(ck1["model"]); m1.eval()
    ck2 = torch.load(a.s2_ckpt, map_location=dev, weights_only=False)
    m2 = PosAR().to(dev); m2.load_state_dict(ck2["model"]); m2.eval()

    seen, chosen = set(), []
    for i in ite:
        t = int(lab[i])
        if t in seen:
            continue
        seen.add(t); chosen.append(i)
        if len(chosen) >= a.n:
            break

    rows, samples = [], []
    for i in chosen:
        w, py, px, sc = panels[i]
        wt, pyt, pxt, sct = w.tolist(), py.tolist(), px.tolist(), sc.tolist()
        ctype = torch.tensor([lab[i]], device=dev)
        ctag = torch.tensor(tg[i:i + 1], device=dev).float()
        pid = int(z["panels"][i]); H, Wd = int(dims[pid][0]), int(dims[pid][1])
        # B: 真の語列 + 生成配置 (PosAR)
        pys_b, pxs_b, scs_b = sample_pos(m2, wt, ctype, ctag, dev, rng)
        # C: 生成語 + 真の位置
        ws_c = sample_words(m1, ctype, ctag, dev, rng)
        nc = min(len(ws_c), len(wt))
        fs_c = (ws_c[:nc], pyt[:nc], pxt[:nc], sct[:nc])
        # D: 生成語 + 生成配置 (セットモデル)
        fs_d = sample_set(ms, ctype, ctag, dev, rng)
        samples.append((int(i), dict(type=int(lab[i]),
                                     A_truth=[wt, pyt, pxt, sct],
                                     B=[wt, pys_b, pxs_b, scs_b],
                                     C=[list(map(list, fs_c))],
                                     D=[list(map(list, fs_d))])))
        label = f"#{i} ty{int(lab[i])}"
        rows.append([render((wt, pyt, pxt, sct), proto, keep, H, Wd, label + " A:truth"),
                     render((wt, pys_b, pxs_b, scs_b), proto, keep, H, Wd, label + " B:trueW+genP"),
                     render(fs_c, proto, keep, H, Wd, label + " C:genW+trueP"),
                     render(fs_d, proto, keep, H, Wd, label + " D:gen")])
    # セル幅をグローバルに統一(切り出し・目視のため)
    cw = max(c.shape[1] for row in rows for c in row)
    rows = [np.hstack([np.pad(c, ((0, 0), (0, cw - c.shape[1]), (0, 0)), constant_values=255)
                       for c in row]) for row in rows]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "montage.png"), np.vstack(rows))
    json.dump(dict(samples), open(out / "ablation_samples.json", "w"))
    print("montage ->", out / "montage.png", f"(cell width {cw}; A truth | B trueW+genP | C genW+trueP | D gen)")


if __name__ == "__main__":
    main()
