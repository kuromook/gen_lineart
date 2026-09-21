#!/usr/bin/env python
"""セットモデル(一括): 条件(型12+タグ40)からまとまり列(語+位置+scale)をAR生成。

入力トークン列: [COND, c_0, ..., c_{n-1}] (c_i = まとまり i の 4フィールド埋め込み+step)。
位置 k の出力から次のまとまり c_k を4 head で予測 (k=n のとき EOS)。
事前登録: doc/work_log.md 2026-09-21「生成モデル比較(セット一括 vs 2段)の事前登録」。
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import N_WORDS, EOS_W, MAXS, N_BINS, S_BINS, CORPUS, load_all, baselines


def causal(S, dev):
    return torch.triu(torch.full((S, S), float("-inf"), device=dev), diagonal=1)


class SetAR(nn.Module):
    def __init__(self, d=256, layers=4, heads=8, dropout=0.1, n_types=12, n_tagf=40):
        super().__init__()
        self.wemb = nn.Embedding(N_WORDS + 1, d)        # +1 = EOS_W
        self.pY = nn.Embedding(N_BINS, d)
        self.pX = nn.Embedding(N_BINS, d)
        self.pS = nn.Embedding(S_BINS, d)
        self.ft = nn.Embedding(4, d)                    # field-type バイアス
        self.step = nn.Embedding(MAXS + 2, d)
        self.cond = nn.Parameter(torch.zeros(1, 1, d))
        self.temb = nn.Embedding(n_types, d)
        self.tagin = nn.Linear(n_tagf, d)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout,
                                       batch_first=True, norm_first=True), layers)
        self.pre = nn.LayerNorm(d)
        self.hw = nn.Linear(d, N_WORDS + 1)
        self.hy = nn.Linear(d, N_BINS)
        self.hx = nn.Linear(d, N_BINS)
        self.hs = nn.Linear(d, S_BINS)

    def encode(self, w, py, px, sc, mask, ctype, ctag):
        """w/py/px/sc/mask: (B,S)。COND+c_0..c_{S-1} をエンコードし (B,S+1,d) を返す。"""
        B, S = w.shape
        c = self.cond + self.temb(ctype).unsqueeze(1) + self.tagin(ctag).unsqueeze(1)
        e = (self.wemb(w.clamp(min=0)) + self.pY(py.clamp(min=0)) + self.pX(px.clamp(min=0))
             + self.pS(sc.clamp(min=0)) + self.ft.weight.sum(0)
             + self.step.weight[1:S + 1].unsqueeze(0))
        x = torch.cat([c, e], 1)                        # (B, S+1, d)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mask], 1)
        return self.tr(x, mask=causal(S + 1, x.device), src_key_padding_mask=pad)

    def heads(self, h):
        h = self.pre(h)
        return self.hw(h), self.hy(h), self.hx(h), self.hs(h)


def batch(panels, idx, lab, tg, dev):
    sel = [panels[i] for i in idx]
    S = max(len(x[0]) for x in sel)
    W = np.zeros((len(sel), S), np.int64); PY = np.zeros_like(W)
    PX = np.zeros_like(W); SC = np.zeros_like(W); MK = np.zeros((len(sel), S), bool)
    TW = np.full((len(sel), S + 1), EOS_W, np.int64)    # 位置 k の予測対象語
    TY = np.zeros((len(sel), S + 1), np.int64); TX = np.zeros_like(TY); TS = np.zeros_like(TY)
    WV = np.zeros((len(sel), S + 1), bool)              # 語+EOS 予測の有効位置
    FV = np.zeros((len(sel), S + 1), bool)              # 位置フィールド予測の有効位置
    for b, (w, py, px, sc) in enumerate(sel):
        n = len(w)
        W[b, :n] = w; PY[b, :n] = py; PX[b, :n] = px; SC[b, :n] = sc; MK[b, :n] = True
        TW[b, :n] = w; TY[b, :n] = py; TX[b, :n] = px; TS[b, :n] = sc
        WV[b, :n + 1] = True                            # 語予測は n+1 位置 (最後は EOS)
        FV[b, :n] = True                                # 位置予測は n 位置
    t = lambda a: torch.from_numpy(a).to(dev)
    return (t(W), t(PY), t(PX), t(SC), t(MK), t(TW), t(TY), t(TX), t(TS),
            t(WV), t(FV), t(lab[idx]), t(tg[idx]).float())


def masked_ce(logits, target, valid, mean=False):
    if logits.dim() == 3:                        # (B,S,C) -> (B,C,S)
        logits = logits.transpose(1, 2)
    l = nn.functional.cross_entropy(logits.float(), target, reduction="none")
    return ((l * valid).mean() if mean else (l * valid).sum()), valid.sum()


@torch.no_grad()
def evaluate(model, panels, idx, lab, tg, dev, bs=64):
    model.eval()
    sw = sew = sy = sx = ss = 0.0
    nf = npanels = 0.0
    for i in range(0, len(idx), bs):
        bt = batch(panels, idx[i:i + bs], lab, tg, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(*bt[:5], bt[11], bt[12])
            lw, ly, lx, ls = model.heads(h)
        a, _ = masked_ce(lw, bt[5], bt[10]); sw += float(a)          # 語: まとまり位置のみ
        a, _ = masked_ce(lw, bt[5], bt[9] & ~bt[10]); sew += float(a)  # EOS: 参考記録
        a, _ = masked_ce(ly, bt[6], bt[10]); sy += float(a)
        a, _ = masked_ce(lx, bt[7], bt[10]); sx += float(a)
        a, _ = masked_ce(ls, bt[8], bt[10]); ss += float(a)
        nf += float(bt[10].sum()); npanels += float(len(bt[5]))
    ln2 = math.log(2)
    return dict(word_bits=sw / nf / ln2, eos_bits=sew / npanels / ln2,
                pos_bits=(sy + sx) / nf / ln2, scale_bits=ss / nf / ln2, n_clusters=nf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/gen_smoke_20260921/set")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--limit", type=int, default=1024, help="学習コマ数 (0=train 全量)")
    p.add_argument("--seed", type=int, default=20260921)
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    d = load_all(a.limit)
    panels, lab, tg, itr, ite = d["panels"], d["lab"], d["tg"], d["itr"], d["ite"]
    H_uni, H_pos, _H_uni_f = baselines(panels, np.flatnonzero(np.load(CORPUS)["split"]))
    print(f"train {len(itr)}  test {len(ite)}  | unigram {H_uni:.3f} 周辺 {H_pos:.3f}", flush=True)

    model = SetAR(n_types=d["n_types"], n_tagf=d["n_tagf"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    steps = max(1, a.epochs * (len(itr) // a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hist, best = [], {"total": 1e9}
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); perm = np.random.permutation(len(itr)); tot = 0.0; nb = 0
        model.train()
        for i in range(0, len(itr) - a.batch + 1, a.batch):
            bt = batch(panels, itr[perm[i:i + a.batch]], lab, tg, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.encode(*bt[:5], bt[11], bt[12])
                lw, ly, lx, ls = model.heads(h)
            l = (masked_ce(lw, bt[5], bt[9], mean=True)[0]
                 + masked_ce(ly, bt[6], bt[10], mean=True)[0]
                 + masked_ce(lx, bt[7], bt[10], mean=True)[0]
                 + masked_ce(ls, bt[8], bt[10], mean=True)[0])
            l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += float(l); nb += 1
        ev = evaluate(model, panels, ite, lab, tg, dev)
        total = ev["word_bits"] + ev["pos_bits"]
        row = dict(epoch=ep, train=tot / max(nb, 1), **ev, sec=time.time() - t0)
        hist.append(row)
        print(f"ep {ep:>3} 学習 {row['train']:.3f} | test word {ev['word_bits']:.3f} "
              f"(unigram {H_uni:.3f}) | pos {ev['pos_bits']:.3f} (周辺 {H_pos:.3f}) | "
              f"scale {ev['scale_bits']:.3f} | {row['sec']:.0f}s", flush=True)
        json.dump(hist, open(out / "history.json", "w"), indent=1)
        torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep,
                    "kind": "set"}, out / "gen.pt")
        if total < best["total"]:
            best = dict(total=total, **ev, epoch=ep)
            torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep,
                        "kind": "set"}, out / "gen_best.pt")
    res = dict(best, unigram_bits=H_uni, pos_marginal_bits=H_pos,
               pass_word=bool(best["word_bits"] <= 7.312),
               pass_pos=bool(best["pos_bits"] < H_pos and best["pos_bits"] <= 6.39))
    json.dump(res, open(out / "eval.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
