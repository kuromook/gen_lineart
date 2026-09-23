#!/usr/bin/env python
"""生成モンタージュ: テストコマの条件(型+タグ)を与えて両モデルに1サンプルずつ生成し、
[真値(原型@真ビン) | セットモデル | 2段モデル] のモンタージュを描く。
描画は語原型のビン中心配置 (Track F render_cloze と同式)。Canvas は条件コマのアスペクト。
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (N_WORDS, EOS_W, MAXS, CORPUS, load_all, load_protos, nat_place)
from train_set import SetAR
from train_2stage import WordAR, PosAR

H_CANVAS = 220


def load_ckpt(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    return ck


def sample_set(model, ctype, ctag, dev, rng, maxs=MAXS):
    """SetAR で1コマ分サンプリング (温度1.0)。返る: (w, py, px, sc) list"""
    ws, pys, pxs, scs = [], [], [], []
    for _ in range(maxs):
        n = len(ws)
        w = torch.zeros(1, n, dtype=torch.long, device=dev)
        py = torch.zeros(1, n, dtype=torch.long, device=dev)
        px = torch.zeros(1, n, dtype=torch.long, device=dev)
        sc = torch.zeros(1, n, dtype=torch.long, device=dev)
        mk = torch.ones(1, n, dtype=torch.bool, device=dev)
        if n:
            w[0] = torch.tensor(ws, device=dev); py[0] = torch.tensor(pys, device=dev)
            px[0] = torch.tensor(pxs, device=dev); sc[0] = torch.tensor(scs, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(w, py, px, sc, mk, ctype, ctag)
            lw, ly, lx, ls = model.heads(h)
        lw, ly, lx, ls = lw[:, -1].float(), ly[:, -1].float(), lx[:, -1].float(), ls[:, -1].float()
        wi = int(torch.multinomial(lw.softmax(-1), 1))
        if wi == EOS_W:
            break
        ws.append(wi)
        pys.append(int(torch.multinomial(ly.softmax(-1), 1)))
        pxs.append(int(torch.multinomial(lx.softmax(-1), 1)))
        scs.append(int(torch.multinomial(ls.softmax(-1), 1)))
    return ws, pys, pxs, scs


def sample_words(m1, ctype, ctag, dev, rng, maxs=MAXS):
    ws = []
    for _ in range(maxs):
        n = len(ws)
        w = torch.zeros(1, n, dtype=torch.long, device=dev)
        mk = torch.ones(1, n, dtype=torch.bool, device=dev)
        if n:
            w[0] = torch.tensor(ws, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lw = m1.head(m1.encode(w, mk, ctype, ctag))
        wi = int(torch.multinomial(lw[:, -1].float().softmax(-1), 1))
        if wi == EOS_W:
            break
        ws.append(wi)
    return ws


def sample_pos(m2, words, ctype, ctag, dev, rng):
    """PosAR / PosAR-W: 語列固定で (py,px,sc) を AR サンプリング(2026-09-23 修正版)。
    学習時 (batch_pos) と同じく語列は全長・mask は全 True で渡す → plan = 全語の平均。
    位置は先行 i 個だけ埋め、スロット i を読む。因果マスクによりスロット i が見るのは
    [c+plan, f_0..f_{i-1}] のみで、0 埋めの未来位置は漏れない。"""
    S = len(words)
    w = torch.tensor([words], dtype=torch.long, device=dev)
    py = torch.zeros(1, S, dtype=torch.long, device=dev)
    px = torch.zeros(1, S, dtype=torch.long, device=dev)
    sc = torch.zeros(1, S, dtype=torch.long, device=dev)
    mk = torch.ones(1, S, dtype=torch.bool, device=dev)
    for i in range(S):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ly, lx, ls = m2.heads(m2.encode(w, py, px, sc, mk, ctype, ctag))
        py[0, i] = torch.multinomial(ly[:, i].float().softmax(-1), 1)[0, 0]
        px[0, i] = torch.multinomial(lx[:, i].float().softmax(-1), 1)[0, 0]
        sc[0, i] = torch.multinomial(ls[:, i].float().softmax(-1), 1)[0, 0]
    return py[0].tolist(), px[0].tolist(), sc[0].tolist()


def sample_pos_prefix(m2, words, ctype, ctag, dev, rng):
    """旧 sample_pos(2026-09-23 まで。比較用に中身不変で残す)。
    **不具合**: 語を i+1 個・mask を先行 i 個だけにしているため plan = 先行語のみの平均
    (rank0 ではゼロ)となり、学習時の plan(全語平均)と食い違う。
    まとまり i の配置には f_0..f_{i-1} が入力に入る必要があるので、
    テンソル長は i+1 (最後は encode 内で AR シフトにより落ちる)"""
    pys, pxs, scs = [], [], []
    for i in range(len(words)):
        S = i + 1                             # 語は i+1 個分、位置は i 個分
        w = torch.zeros(1, S, dtype=torch.long, device=dev)
        py = torch.zeros(1, S, dtype=torch.long, device=dev)
        px = torch.zeros(1, S, dtype=torch.long, device=dev)
        sc = torch.zeros(1, S, dtype=torch.long, device=dev)
        mk = torch.zeros(1, S, dtype=torch.bool, device=dev)
        w[0] = torch.tensor(words[:S], device=dev)
        if i:
            py[0, :i] = torch.tensor(pys, device=dev)
            px[0, :i] = torch.tensor(pxs, device=dev)
            sc[0, :i] = torch.tensor(scs, device=dev)
            mk[0, :i] = True
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ly, lx, ls = m2.heads(m2.encode(w, py, px, sc, mk, ctype, ctag))
        k = i                                     # 出力位置 i がまとまり i を予測
        pys.append(int(torch.multinomial(ly[:, k].float().softmax(-1), 1)))
        pxs.append(int(torch.multinomial(lx[:, k].float().softmax(-1), 1)))
        scs.append(int(torch.multinomial(ls[:, k].float().softmax(-1), 1)))
    return pys, pxs, scs


def render(fields, proto, keep, H, Wd, label=""):
    """(w,py,px,sc) のリストを白 canvas に描く"""
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
    ap.add_argument("--out", default="results/gen_smoke_20260921")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    dev = torch.device("cuda")
    rng = np.random.default_rng(777)
    d = load_all(0)
    panels, lab, tg, ite = d["panels"], d["lab"], d["tg"], d["ite"]
    z = np.load(CORPUS)
    dims = z["dims"]

    proto, keep = load_protos(dev)

    cks = load_ckpt(a.set_ckpt, dev); ms = SetAR().to(dev)
    ms.load_state_dict(cks["model"]); ms.eval()
    ck1 = load_ckpt(a.s1_ckpt, dev); m1 = WordAR().to(dev)
    m1.load_state_dict(ck1["model"]); m1.eval()
    ck2 = load_ckpt(a.s2_ckpt, dev); m2 = PosAR().to(dev)
    m2.load_state_dict(ck2["model"]); m2.eval()

    # 型がなるべく被らないようにテストコマを選ぶ
    seen, chosen = set(), []
    for i in ite:
        t = int(lab[i])
        if t in seen:
            continue
        seen.add(t); chosen.append(i)
        if len(chosen) >= a.n:
            break

    rows, samples = [], {}
    for i in chosen:
        w, py, px, sc = panels[i]
        ctype = torch.tensor([lab[i]], device=dev)
        ctag = torch.tensor(tg[i:i + 1], device=dev).float()
        pid = int(z["panels"][i]); H, Wd = (int(dims[pid][0]), int(dims[pid][1]))
        fs = sample_set(ms, ctype, ctag, dev, rng)
        ws2 = sample_words(m1, ctype, ctag, dev, rng)
        pys2, pxs2, scs2 = sample_pos(m2, ws2, ctype, ctag, dev, rng)
        fs2 = (ws2, pys2, pxs2, scs2)
        samples[int(i)] = dict(type=int(lab[i]), truth=[w.tolist(), py.tolist(), px.tolist(), sc.tolist()],
                               set=[list(map(list, fs))], twostage=[list(map(list, fs2))])
        label = f"panel#{i} type{int(lab[i])}"
        cells = [render((w.tolist(), py.tolist(), px.tolist(), sc.tolist()), proto, keep, H, Wd,
                        label + " TRUTH"),
                 render(fs, proto, keep, H, Wd, label + " SET"),
                 render(fs2, proto, keep, H, Wd, label + " 2STAGE")]
        hmax = max(c.shape[1] for c in cells)
        cells = [np.pad(c, ((0, 0), (0, hmax - c.shape[1]), (0, 0)), constant_values=255)
                 for c in cells]
        rows.append(np.hstack(cells))
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    out = Path(a.out)
    cv2.imwrite(str(out / "montage.png"), np.vstack(rows))
    json.dump(samples, open(out / "gen_samples.json", "w"))
    print("montage ->", out / "montage.png", " (columns: TRUTH | SET | 2STAGE)")


if __name__ == "__main__":
    main()
