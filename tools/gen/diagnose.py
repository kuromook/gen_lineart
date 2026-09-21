#!/usr/bin/env python
"""結果検証ダイアグノスティクス(「道具を測っていないか」の確認用):
1. リーク検査: ターゲット列をコマ内でランダム並べ替え → bits が大きく悪化するはず。
   悪化しない = 入力にターゲットが漏れている
2. ランク別 bits: 正準順(粗大先行)での予測位置 k ごとの CE。語・scale・pos を分割して出す。
   Track F clozeは「1コマから1まとまりをランダムに抜く」(ランク無作為)。
   一方 AR 生成は全ランクを平均に含める → 平均の差の分解用
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EOS_W, load_all
from train_set import SetAR, batch, masked_ce
from train_2stage import PosAR, batch_pos

L2 = np.log(2)


def eval_set(model, panels, idx, lab, tg, dev, scramble=False, bs=64, rng=None):
    model.eval()
    sw = sy = sx = ss = 0.0
    nf = 0.0
    rk_w = np.zeros(80); rk_n = np.zeros(80)
    rk_p = np.zeros(80); rk_s = np.zeros(80)
    for i in range(0, len(idx), bs):
        bt = batch(panels, idx[i:i + bs], lab, tg, dev)
        if scramble:
            TW, TY, TX, TS, WV2, FV2 = bt[5].clone(), bt[6].clone(), bt[7].clone(), bt[8].clone(), bt[9].clone(), bt[10].clone()
            TW[:], TY[:], TX[:], TS[:] = 0, 0, 0, 0
            WV2[:], FV2[:] = False, False
            for b in range(len(TW)):
                n = int(bt[10][b].sum())
                perm = rng.permutation(n)
                TW[b, :n] = bt[5][b, perm]
                TY[b, :n] = bt[6][b, perm]; TX[b, :n] = bt[7][b, perm]; TS[b, :n] = bt[8][b, perm]
                WV2[b, :n + 1] = True; FV2[b, :n] = True
            bt = bt[:5] + (TW, TY, TX, TS, WV2, FV2) + bt[11:]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(*bt[:5], bt[11], bt[12])
            lw, ly, lx, ls = model.heads(h)
        ce_w = torch.nn.functional.cross_entropy(lw.transpose(1, 2).float(), bt[5], reduction="none")
        ce_y = torch.nn.functional.cross_entropy(ly.transpose(1, 2).float(), bt[6], reduction="none")
        ce_x = torch.nn.functional.cross_entropy(lx.transpose(1, 2).float(), bt[7], reduction="none")
        ce_s = torch.nn.functional.cross_entropy(ls.transpose(1, 2).float(), bt[8], reduction="none")
        fv = bt[10]
        sw += float((ce_w * fv).sum()); sy += float((ce_y * fv).sum())
        sx += float((ce_x * fv).sum()); ss += float((ce_s * fv).sum())
        nf += float(fv.sum())
        for b in range(len(fv)):
            n = int(fv[b].sum())
            for k in range(min(n, 80)):
                rk_w[k] += float(ce_w[b, k]); rk_p[k] += float(ce_y[b, k] + ce_x[b, k])
                rk_s[k] += float(ce_s[b, k]); rk_n[k] += 1
    out = dict(word_bits=sw / nf / L2, pos_bits=(sy + sx) / nf / L2, scale_bits=ss / nf / L2)
    if not scramble:
        out["per_rank"] = [[int(k), int(rk_n[k]), rk_w[k] / max(rk_n[k], 1) / L2,
                            rk_p[k] / max(rk_n[k], 1) / L2, rk_s[k] / max(rk_n[k], 1) / L2]
                           for k in range(60) if rk_n[k] > 0]
    return out


@torch.no_grad()
def eval_pos_rank(model, panels, idx, lab, tg, dev, bs=64):
    model.eval()
    rk_p = np.zeros(80); rk_s = np.zeros(80); rk_n = np.zeros(80)
    sy = sx = nf = 0.0
    for i in range(0, len(idx), bs):
        bt = batch_pos(panels, idx[i:i + bs], lab, tg, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ly, lx, ls = model.heads(model.encode(*bt[:5], bt[9], bt[10]))
        ce_y = torch.nn.functional.cross_entropy(ly.transpose(1, 2).float(), bt[5], reduction="none")
        ce_x = torch.nn.functional.cross_entropy(lx.transpose(1, 2).float(), bt[6], reduction="none")
        ce_s = torch.nn.functional.cross_entropy(ls.transpose(1, 2).float(), bt[7], reduction="none")
        fv = bt[8]
        sy += float((ce_y * fv).sum()); sx += float((ce_x * fv).sum()); nf += float(fv.sum())
        for b in range(len(fv)):
            n = int(fv[b].sum())
            for k in range(min(n, 80)):
                rk_p[k] += float(ce_y[b, k] + ce_x[b, k]); rk_s[k] += float(ce_s[b, k]); rk_n[k] += 1
    return dict(pos_bits=(sy + sx) / nf / np.log(2),
                per_rank=[[int(k), int(rk_n[k]), rk_p[k] / max(rk_n[k], 1) / np.log(2),
                           rk_s[k] / max(rk_n[k], 1) / np.log(2)] for k in range(60) if rk_n[k] > 0])


def main():
    dev = torch.device("cuda")
    d = load_all(0)
    panels, lab, tg, ite = d["panels"], d["lab"], d["tg"], d["ite"]
    rng = np.random.default_rng(123)

    ck = torch.load("results/gen_smoke_20260921/set/gen_best.pt", map_location=dev, weights_only=False)
    ms = SetAR().to(dev); ms.load_state_dict(ck["model"]); ms.eval()
    true = eval_set(ms, panels, ite, lab, tg, dev)
    scram = eval_set(ms, panels, ite, lab, tg, dev, scramble=True, rng=rng)
    print("[set] 真の並び:", {k: round(v, 3) for k, v in true.items() if k != "per_rank"})
    print("[set] 並び替え:", {k: round(v, 3) for k, v in scram.items()})
    print("[set] ランク別 (rank, n, word_bits, pos_bits, scale_bits):")
    for r in true["per_rank"][:16]:
        print("   ", r[0], r[1], round(r[2], 3), round(r[3], 3), round(r[4], 3))

    ck2 = torch.load("results/gen_smoke_20260921/twostage/stage2/gen_best.pt", map_location=dev, weights_only=False)
    m2 = PosAR().to(dev); m2.load_state_dict(ck2["model"]); m2.eval()
    p2 = eval_pos_rank(m2, panels, ite, lab, tg, dev)
    print("[2stage-pos] 全体:", round(p2["pos_bits"], 3))
    print("[2stage-pos] ランク別 (rank, n, pos_bits, scale_bits):")
    for r in p2["per_rank"][:16]:
        print("   ", r[0], r[1], round(r[2], 3), round(r[3], 3))
    json.dump(dict(set_true={k: v for k, v in true.items()}, set_scrambled=scram,
                   twostage_pos=p2["pos_bits"]),
              open("results/gen_smoke_20260921/diagnostics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
