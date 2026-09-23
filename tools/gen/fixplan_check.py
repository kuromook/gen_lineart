#!/usr/bin/env python
"""道具修正: sample_pos の plan 不整合(2026-09-23 事前登録)。

1. 等価性: 修正版サンプラーの入力形式でのスロット i の logits が teacher forcing(全長)と
   一致し、旧版(先行語のみの plan)では一致しないこと(fp32)
2. 診断: test 全量で「旧版の入力形式」の pos bits を計算し、正規の teacher forcing と並べる
3. 描画: 同じ12コマ・同じ実インスタンス・同じ torch seed で [A 真値 | B旧 | B修正]
"""
import argparse, json, math, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trackf"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS, load_all                                     # noqa: E402
from train_2stage import PosAR, eval_pos                               # noqa: E402
from render_gen import sample_pos, sample_pos_prefix                   # noqa: E402
from render_real import build_clusters, verify_tokens, draw_clusters, bin_pos, H_CANVAS  # noqa: E402


def tens(panel, dev):
    w, py, px, sc = (torch.tensor(a.astype(np.int64), device=dev)[None] for a in panel)
    return w, py, px, sc


@torch.no_grad()
def equivalence(m2, panels, idx, lab, tg, dev):
    """fp32。返る: 各コマの max|Δ|(修正版入力 vs TF, 旧版入力 vs TF)"""
    out = []
    for i in idx:
        w, py, px, sc = tens(panels[i], dev)
        S = w.shape[1]
        ct = torch.tensor([lab[i]], device=dev); cg = torch.tensor(tg[i:i + 1], device=dev).float()
        mk = torch.ones(1, S, dtype=torch.bool, device=dev)
        ref = torch.cat(m2.heads(m2.encode(w, py, px, sc, mk, ct, cg)), -1)[0]   # (S, 16+16+12)
        d_new = d_old = 0.0
        for k in range(S):
            z = torch.zeros_like(py)
            pyk, pxk, sck = z.clone(), z.clone(), z.clone()
            pyk[0, :k] = py[0, :k]; pxk[0, :k] = px[0, :k]; sck[0, :k] = sc[0, :k]
            new = torch.cat(m2.heads(m2.encode(w, pyk, pxk, sck, mk, ct, cg)), -1)[0, k]
            mo = torch.zeros(1, k + 1, dtype=torch.bool, device=dev); mo[0, :k] = True
            old = torch.cat(m2.heads(m2.encode(w[:, :k + 1], pyk[:, :k + 1], pxk[:, :k + 1],
                                               sck[:, :k + 1], mo, ct, cg)), -1)[0, k]
            d_new = max(d_new, float((new - ref[k]).abs().max()))
            d_old = max(d_old, float((old - ref[k]).abs().max()))
        out.append(dict(row=int(i), n=S, maxdiff_new=d_new, maxdiff_old=d_old))
    return out


@torch.no_grad()
def prefix_bits(m2, panels, idx, lab, tg, dev):
    """旧版の入力形式(スロット i の plan = 先行 i 語の平均)で teacher forcing した bits。
    1コマ n 行のバッチ: 行 i は mask 先行 i 個のみ True、スロット i を読む。rank 別も返す"""
    sy = sx = ss = nf = 0.0
    r0 = [0.0, 0.0]
    for i in idx:
        w, py, px, sc = tens(panels[i], dev)
        n = w.shape[1]
        ct = torch.full((n,), int(lab[i]), device=dev)
        cg = torch.tensor(tg[i:i + 1], device=dev).float().expand(n, -1)
        mk = torch.arange(n, device=dev)[None] < torch.arange(n, device=dev)[:, None]   # 行 i: 先行 i 個
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ly, lx, ls = m2.heads(m2.encode(w.expand(n, -1), py.expand(n, -1), px.expand(n, -1),
                                            sc.expand(n, -1), mk, ct, cg))
        d = torch.arange(n, device=dev)
        ly, lx, ls = ly[d, d].float(), lx[d, d].float(), ls[d, d].float()
        cy = F.cross_entropy(ly, py[0], reduction="none"); cx = F.cross_entropy(lx, px[0], reduction="none")
        sy += float(cy.sum()); sx += float(cx.sum())
        ss += float(F.cross_entropy(ls, sc[0], reduction="sum")); nf += n
        r0[0] += float(cy[0] + cx[0]); r0[1] += 1
    ln2 = math.log(2)
    return dict(pos_bits=(sy + sx) / nf / ln2, scale_bits=ss / nf / ln2,
                rank0_pos_bits=r0[0] / r0[1] / ln2)


@torch.no_grad()
def tf_rank0(m2, panels, idx, lab, tg, dev):
    s = 0.0
    for i in idx:
        w, py, px, sc = tens(panels[i], dev)
        ct = torch.tensor([lab[i]], device=dev); cg = torch.tensor(tg[i:i + 1], device=dev).float()
        mk = torch.ones_like(w, dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ly, lx, _ = m2.heads(m2.encode(w, py, px, sc, mk, ct, cg))
        s += float(F.cross_entropy(ly[0, :1].float(), py[0, :1]) + F.cross_entropy(lx[0, :1].float(), px[0, :1]))
    return s / len(idx) / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2_ckpt", default="results/gen_smoke_20260921/twostage/stage2/gen_best.pt")
    ap.add_argument("--out", default="results/fixplan_20260923")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--n_equiv", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260923)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    d = load_all(0)
    panels, lab, tg, ite = d["panels"], d["lab"], d["tg"], d["ite"]
    ck2 = torch.load(a.s2_ckpt, map_location=dev, weights_only=False)
    m2 = PosAR().to(dev); m2.load_state_dict(ck2["model"]); m2.eval()

    # ---- 1. 等価性 ----
    eq = equivalence(m2, panels, ite[:a.n_equiv], lab, tg, dev)
    for r in eq:
        print(f"equiv row {r['row']} n={r['n']}: new {r['maxdiff_new']:.2e}  old {r['maxdiff_old']:.2e}", flush=True)
    ok_new = all(r["maxdiff_new"] < 1e-4 for r in eq)
    bug_old = all(r["maxdiff_old"] > 1e-3 for r in eq)
    print(f"修正版 = TF: {ok_new} / 旧版 != TF: {bug_old}", flush=True)

    # ---- 2. 診断 bits ----
    tf = eval_pos(m2, panels, ite, lab, tg, dev)
    tf["rank0_pos_bits"] = tf_rank0(m2, panels, ite, lab, tg, dev)
    pb = prefix_bits(m2, panels, ite, lab, tg, dev)
    print(f"pos bits  TF(正規) {tf['pos_bits']:.3f} (rank0 {tf['rank0_pos_bits']:.3f})  "
          f"旧版入力 {pb['pos_bits']:.3f} (rank0 {pb['rank0_pos_bits']:.3f})", flush=True)
    json.dump(dict(equiv=eq, pass_new=ok_new, bug_old=bug_old, tf=tf, prefix=pb),
              open(out / "equiv.json", "w"), indent=1)
    assert ok_new, "修正版サンプラーの入力が teacher forcing と一致しない — 描画に進まない"

    # ---- 3. 描画 A | B旧 | B修正 ----
    P, W, M, words, scale, cy, cx, panel_order, pools = build_clusters(dev)
    z = np.load(CORPUS); dims = z["dims"]
    assert verify_tokens(panel_order, words, scale, cy, cx, dims) < 20
    rng = np.random.default_rng(777)
    seen, chosen = set(), []
    for i in ite:                                     # 12コマ選定(従来ロジック)
        t = int(lab[i])
        if t in seen:
            continue
        seen.add(t); chosen.append(i)
        if len(chosen) >= a.n:
            break

    def cell(pid, idxs_pos, label):
        H, Wd = int(dims[pid][0]), int(dims[pid][1])
        zf = H_CANVAS / H
        canvas = np.full((H_CANVAS + 24, max(int(Wd * zf), 2), 3), 255, np.uint8)
        if idxs_pos:
            idxs, pos = zip(*idxs_pos)
            draw_clusters(canvas, P, W, M, list(idxs), list(pos), zf)
        cv2.putText(canvas, label, (4, H_CANVAS + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 200), 1, cv2.LINE_AA)
        return canvas

    rows, samples = [], {}
    for r, i in enumerate(chosen):
        pid = int(z["panels"][i])
        wt = panels[i][0].tolist()
        ctype = torch.tensor([lab[i]], device=dev)
        ctag = torch.tensor(tg[i:i + 1], device=dev).float()
        H, Wd = int(dims[pid][0]), int(dims[pid][1])
        cell_a = cell(pid, [(ci, (cy[ci], cx[ci], scale[ci])) for ci in panel_order[pid]],
                      f"#{i} ty{int(lab[i])} A:truth")
        ib = [rng.choice(pools[wi]) if pools[wi] else -1 for wi in wt]
        res = {}
        for tag, fn in (("old", sample_pos_prefix), ("fix", sample_pos)):
            torch.manual_seed(a.seed + r)
            res[tag] = fn(m2, wt, ctype, ctag, dev, rng)
        cells = [cell_a]
        for tag, lbl in (("old", "B:old(prefix plan)"), ("fix", "B:fixed")):
            pys, pxs, scs = res[tag]
            cells.append(cell(pid, [(ci, bin_pos(y, x, s, H, Wd)) for ci, y, x, s
                                    in zip(ib, pys, pxs, scs) if ci >= 0], f"#{i} {lbl}"))
        samples[int(i)] = dict(type=int(lab[i]), words=wt, old=res["old"], fix=res["fix"])
        rows.append(cells)
    cw = max(c.shape[1] for row in rows for c in row)
    padc = lambda c: np.pad(c, ((0, 0), (0, cw - c.shape[1]), (0, 0)), constant_values=255)
    sep = lambda c: np.pad(padc(c), ((0, 0), (0, 6), (0, 0)), constant_values=200)
    cv2.imwrite(str(out / "montage.png"), np.vstack([np.hstack([sep(c) for c in row]) for row in rows]))
    for row, i in zip(rows, chosen):
        cv2.imwrite(str(out / f"cells_{i}_ty{int(lab[i])}.png"), np.hstack([sep(c) for c in row]))
    json.dump(samples, open(out / "samples.json", "w"))
    print("montage ->", out / "montage.png", " (A truth | B old | B fixed)")


if __name__ == "__main__":
    main()
