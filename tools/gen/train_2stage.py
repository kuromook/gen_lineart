#!/usr/bin/env python
"""2段モデル(選択→配置):
  第1段 (WordAR): 条件(型+タグ)+先行「単語」から語列を AR 生成 (位置はまだ出さない)。
  第2段 (PosAR):  条件+「全単語の計画ベクトル」+先行位置から (py,px,sc) を AR 生成。

事前登録: doc/work_log.md 2026-09-21「生成モデル比較(セット一括 vs 2段)の事前登録」。
語の内容は第1段のサンプルのまま(配置段で語は変えない)。
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


class WordAR(nn.Module):
    """第1段: 条件+先行単語 -> 次の語 (0..499 or EOS_W)"""

    def __init__(self, d=256, layers=4, heads=8, dropout=0.1, n_types=12, n_tagf=40):
        super().__init__()
        self.wemb = nn.Embedding(N_WORDS + 1, d)
        self.step = nn.Embedding(MAXS + 2, d)
        self.cond = nn.Parameter(torch.zeros(1, 1, d))
        self.temb = nn.Embedding(n_types, d)
        self.tagin = nn.Linear(n_tagf, d)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout,
                                       batch_first=True, norm_first=True), layers)
        self.pre = nn.LayerNorm(d)
        self.hw = nn.Linear(d, N_WORDS + 1)

    def encode(self, w, mask, ctype, ctag):
        """w/mask: (B,S)。[COND, w_0..w_{S-1}] -> (B,S+1,d)"""
        B, S = w.shape
        c = self.cond + self.temb(ctype).unsqueeze(1) + self.tagin(ctag).unsqueeze(1)
        e = self.wemb(w.clamp(min=0)) + self.step.weight[1:S + 1].unsqueeze(0)
        x = torch.cat([c, e], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mask], 1)
        return self.tr(x, mask=causal(S + 1, x.device), src_key_padding_mask=pad)

    def head(self, h):
        return self.hw(self.pre(h))


class PosAR(nn.Module):
    """第2段: 条件+全単語の計画+先行位置 -> 次の (py,px,sc)"""

    def __init__(self, d=256, layers=4, heads=8, dropout=0.1, n_types=12, n_tagf=40):
        super().__init__()
        self.wemb = nn.Embedding(N_WORDS, d)
        self.pY = nn.Embedding(N_BINS, d)
        self.pX = nn.Embedding(N_BINS, d)
        self.pS = nn.Embedding(S_BINS, d)
        self.ft = nn.Embedding(4, d)
        self.step = nn.Embedding(MAXS + 2, d)
        self.cond = nn.Parameter(torch.zeros(1, 1, d))
        self.temb = nn.Embedding(n_types, d)
        self.tagin = nn.Linear(n_tagf, d)
        self.plan = nn.Linear(d, d)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout,
                                       batch_first=True, norm_first=True), layers)
        self.pre = nn.LayerNorm(d)
        self.hy = nn.Linear(d, N_BINS)
        self.hx = nn.Linear(d, N_BINS)
        self.hs = nn.Linear(d, S_BINS)

    def encode(self, w, py, px, sc, mask, ctype, ctag):
        """w/py/px/sc/mask: (B,S) 全まとまり。入力は [COND+plan, f_0..f_{S-2}]
        (最後のまとまりは入力から落とす=AR シフト。出力位置 k がまとまり k を予測)。"""
        B, S = w.shape
        pl = (self.wemb(w.clamp(min=0)) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        c = (self.cond + self.temb(ctype).unsqueeze(1) + self.tagin(ctag).unsqueeze(1)
             + self.plan(pl).unsqueeze(1))
        f = (self.wemb(w.clamp(min=0)) + self.pY(py.clamp(min=0)) + self.pX(px.clamp(min=0))
             + self.pS(sc.clamp(min=0)) + self.ft.weight.sum(0)
             + self.step.weight[1:S + 1].unsqueeze(0))
        x = torch.cat([c, f[:, :max(S - 1, 0)]], 1)      # (B, S, d)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device),
                         ~mask[:, :max(S - 1, 0)]], 1)
        return self.tr(x, mask=causal(S, x.device), src_key_padding_mask=pad)

    def heads(self, h):
        h = self.pre(h)
        return self.hy(h), self.hx(h), self.hs(h)


class PosARW(PosAR):
    """第2段(語条件化、2026-09-23 事前登録「配置の語条件化」):

    現行 PosAR は出力スロット k+1 が「置く語 w_{k+1}」を plan(全語平均→1層)
    経由しか見ない。こちらはスロット k+1 の入力に wemb(w_{k+1}) を直接加える
    (g_k = f_k + e_{k+1}。位置は先行のみ=シフト構造維持でターゲット漏れなし)。
    """

    def encode(self, w, py, px, sc, mask, ctype, ctag):
        B, S = w.shape
        pl = (self.wemb(w.clamp(min=0)) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        c = (self.cond + self.temb(ctype).unsqueeze(1) + self.tagin(ctag).unsqueeze(1)
             + self.plan(pl).unsqueeze(1))
        e = self.wemb(w.clamp(min=0))
        f = (e + self.pY(py.clamp(min=0)) + self.pX(px.clamp(min=0))
             + self.pS(sc.clamp(min=0)) + self.ft.weight.sum(0)
             + self.step.weight[1:S + 1].unsqueeze(0))
        g = f[:, :max(S - 1, 0)] + e[:, 1:S]         # 先行まとまり k + 自分の語 w_{k+1}
        x = torch.cat([c, g], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device),
                         ~mask[:, :max(S - 1, 0)]], 1)
        return self.tr(x, mask=causal(S, x.device), src_key_padding_mask=pad)


def masked_ce(logits, target, valid, mean=False):
    if logits.dim() == 3:                        # (B,S,C) -> (B,C,S)
        logits = logits.transpose(1, 2)
    l = nn.functional.cross_entropy(logits.float(), target, reduction="none")
    return ((l * valid).mean() if mean else (l * valid).sum()), valid.sum()


def batch_words(panels, idx, lab, tg, dev):
    """第1段用: [COND, w_0..w_{S-1}] 入力、位置 k で w_k (最後は EOS) を予測"""
    sel = [panels[i] for i in idx]
    S = max(len(x[0]) for x in sel)
    W = np.zeros((len(sel), S), np.int64); MK = np.zeros((len(sel), S), bool)
    TW = np.full((len(sel), S + 1), EOS_W, np.int64)
    WV = np.zeros((len(sel), S + 1), bool)
    for b, (w, _py, _px, _sc) in enumerate(sel):
        n = len(w)
        W[b, :n] = w; MK[b, :n] = True
        TW[b, :n] = w; WV[b, :n + 1] = True
    t = lambda a: torch.from_numpy(a).to(dev)
    return (t(W), t(MK), t(TW), t(WV), t(lab[idx]), t(tg[idx]).float())


def batch_pos(panels, idx, lab, tg, dev):
    """第2段用: 入力 [COND+plan, f_0..f_{S-2}]。出力位置 k でまとまり k の (py,px,sc)"""
    sel = [panels[i] for i in idx]
    S = max(len(x[0]) for x in sel)
    B = len(sel)
    W = np.zeros((B, S), np.int64); PY = np.zeros_like(W)
    PX = np.zeros_like(W); SC = np.zeros_like(W); MK = np.zeros((B, S), bool)
    TY = np.zeros((B, S), np.int64); TX = np.zeros_like(TY); TS = np.zeros_like(TY)
    FV = np.zeros((B, S), bool)
    for b, (w, py, px, sc) in enumerate(sel):
        n = len(w)
        W[b, :n] = w; PY[b, :n] = py; PX[b, :n] = px; SC[b, :n] = sc; MK[b, :n] = True
        TY[b, :n] = py; TX[b, :n] = px; TS[b, :n] = sc
        FV[b, :n] = True                          # 出力位置 0..n-1
    t = lambda a: torch.from_numpy(a).to(dev)
    return (t(W), t(PY), t(PX), t(SC), t(MK), t(TY), t(TX), t(TS), t(FV),
            t(lab[idx]), t(tg[idx]).float())


@torch.no_grad()
def eval_words(model, panels, idx, lab, tg, dev, bs=64):
    model.eval()
    sw = sew = nf = npanels = 0.0
    for i in range(0, len(idx), bs):
        bt = batch_words(panels, idx[i:i + bs], lab, tg, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lw = model.head(model.encode(bt[0], bt[1], bt[4], bt[5]))
        a, _ = masked_ce(lw, bt[2], bt[3] & (bt[2] != EOS_W)); sw += float(a)
        a, _ = masked_ce(lw, bt[2], bt[3] & (bt[2] == EOS_W)); sew += float(a)
        nf += float((bt[2] != EOS_W).sum()); npanels += float(len(bt[2]))
    ln2 = math.log(2)
    return dict(word_bits=sw / nf / ln2, eos_bits=sew / npanels / ln2)


@torch.no_grad()
def eval_pos(model, panels, idx, lab, tg, dev, bs=64):
    model.eval()
    sy = sx = ss = nf = 0.0
    for i in range(0, len(idx), bs):
        bt = batch_pos(panels, idx[i:i + bs], lab, tg, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ly, lx, ls = model.heads(model.encode(*bt[:5], bt[9], bt[10]))
        a, _ = masked_ce(ly, bt[5], bt[8]); sy += float(a)
        a, _ = masked_ce(lx, bt[6], bt[8]); sx += float(a)
        a, _ = masked_ce(ls, bt[7], bt[8]); ss += float(a)
        nf += float(bt[8].sum())
    ln2 = math.log(2)
    return dict(pos_bits=(sy + sx) / nf / ln2, scale_bits=ss / nf / ln2)


def train_stage(kind, model, panels, itr, lab, tg, dev, a, out, batchfn, evfn, lossfn):
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    steps = max(1, a.epochs * (len(itr) // a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    sdir = out / kind; sdir.mkdir(parents=True, exist_ok=True)
    hist, best = [], {"total": 1e9}
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); perm = np.random.permutation(len(itr)); tot = 0.0; nb = 0
        model.train()
        for i in range(0, len(itr) - a.batch + 1, a.batch):
            bt = batchfn(panels, itr[perm[i:i + a.batch]], lab, tg, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l = lossfn(model, bt)
            l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += float(l); nb += 1
        ev = evfn(model)
        hist.append(dict(epoch=ep, train=tot / max(nb, 1), **ev, sec=time.time() - t0))
        print(f"[{kind}] ep {ep:>3} 学習 {hist[-1]['train']:.3f} | {ev} | {hist[-1]['sec']:.0f}s",
              flush=True)
        json.dump(hist, open(sdir / "history.json", "w"), indent=1)
        torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep,
                    "kind": kind}, sdir / "gen.pt")
        if ev["total"] < best["total"]:
            best = dict(ev, epoch=ep)
            torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep,
                        "kind": kind}, sdir / "gen_best.pt")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/gen_smoke_20260921/twostage")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--limit", type=int, default=1024, help="学習コマ数 (0=train 全量)")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--posw", action="store_true",
                   help="stage1 を飛ばし、語条件化 PosAR-W を stage2w として学習")
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    d = load_all(a.limit)
    panels, lab, tg, itr, ite = d["panels"], d["lab"], d["tg"], d["itr"], d["ite"]
    H_uni, H_pos, _H_uni_f = baselines(panels, np.flatnonzero(np.load(CORPUS)["split"]))
    print(f"train {len(itr)}  test {len(ite)}  | unigram {H_uni:.3f} 周辺 {H_pos:.3f}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---- 第1段: 語列 AR ----
    if not a.posw:
        m1 = WordAR(n_types=d["n_types"], n_tagf=d["n_tagf"]).to(dev)

        def loss1(m, bt):
            lw = m.head(m.encode(bt[0], bt[1], bt[4], bt[5]))
            return masked_ce(lw, bt[2], bt[3], mean=True)[0]

        def ev1(m, idx_eval):
            r = eval_words(m, panels, idx_eval, lab, tg, dev)
            r["total"] = r["word_bits"]
            return r

        best1 = train_stage("stage1", m1, panels, itr, lab, tg, dev, a, out,
                            batch_words, lambda m: ev1(m, ite), loss1)

    # ---- 第2段: 位置 AR (PosAR or PosAR-W) ----
    cls = PosARW if a.posw else PosAR
    kind = "stage2w" if a.posw else "stage2"
    m2 = cls(n_types=d["n_types"], n_tagf=d["n_tagf"]).to(dev)

    def loss2(m, bt):
        ly, lx, ls = m.heads(m.encode(*bt[:5], bt[9], bt[10]))
        return (masked_ce(ly, bt[5], bt[8], mean=True)[0]
                + masked_ce(lx, bt[6], bt[8], mean=True)[0]
                + masked_ce(ls, bt[7], bt[8], mean=True)[0])

    def ev2(m, idx_eval):
        r = eval_pos(m, panels, idx_eval, lab, tg, dev)
        r["total"] = r["pos_bits"]
        return r

    best2 = train_stage(kind, m2, panels, itr, lab, tg, dev, a, out,
                        batch_pos, lambda m: ev2(m, ite), loss2)

    res = dict(unigram_bits=H_uni, pos_marginal_bits=H_pos,
               pos_bits=best2["pos_bits"], scale_bits=best2["scale_bits"],
               pass_pos=bool(best2["pos_bits"] < H_pos and best2["pos_bits"] <= 6.39))
    if a.posw:
        res["stage2w"] = best2
        res["ref_stage2_pos_bits"] = 6.658          # 2026-09-21 スモーク実測(比較用)
    else:
        res.update(stage1=best1, stage2=best2, word_bits=best1["word_bits"],
                   pass_word=bool(best1["word_bits"] <= 7.312))
    json.dump(res, open(out / "eval.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
