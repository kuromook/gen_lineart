#!/usr/bin/env python
"""配置: 相対位置の効きを「関係の種類」ごとに測る(2026-09-25 事前登録・第3版)。

語と大きさは既知、位置だけを予測。コマ内 16×16 格子で真のセルを言い当てる bits。
シリーズ 2 群の一方で数え、他方で評価(A→B / B→A を別報告)。
  M0 一様 / M1 周辺 / M2 絶対(語別) / M3a 純相対 / M3b 相対+絶対
  相対のずれ = 方向 16 × 対数距離 21 環(距離の単位は基準語 a の大きさ 28/scale_a)。
  全ペア共通のずれ分布は基準語の相対的な大きさ 4 段階ごと
  関係の種類(学習側): (1) d_med ≤ 4 かつ IQR 下位 1/3 / (2) d_med ≤ 16 / (3) それ以外
  順序: O1 大きい順 / O2 関係の種類を優先する連鎖 / O3 無作為 ×10(アンカーは M2、以後 M3b)
--check: T1 / T1b(種類(1)の M3a と全体の M3b ≤ M0)/ T2(種類(1)の基準語の位置入替で悪化)/ T3
"""
import argparse, csv, json, math, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from placement_gt import load_data, dedupe, draw_inst, A_SIZE          # noqa: E402

G = 16                                  # コマ格子
NS = 16                                 # 方向の区分
L0, L1, LS = -2.0, 8.0, 0.5             # log2 距離: <-2 を 0 環、-2..8 を 0.5 刻み
NR = 1 + int((L1 - L0) / LS)            # 21 環。最外環は 7.5–8.0(それ以遠は最外環に丸める)
NB = NR * NS
SUB = 4
KAPPA = 20.0
R_EDGE = np.r_[0.0, 2.0 ** (L0 + LS * np.arange(NR))]                  # 環の内外半径
BIN_AREA = np.repeat(np.pi * (R_EDGE[1:] ** 2 - R_EDGE[:-1] ** 2) / NS, NS)   # 区画の面積(a の大きさ単位)

KB = 144                                # 区画あたりの点(12×12 層別)
def _bin_points():
    """区画ごとに面積一様の点 KB 個(層別)。最外環の外縁は 2^8 とする。返る (NB, KB) の x, y"""
    g = int(np.sqrt(KB)); ur = (np.arange(g) + 0.5) / g
    bx = np.zeros((NB, KB)); by = np.zeros((NB, KB))
    for r in range(NR):
        r0, r1 = R_EDGE[r], R_EDGE[r + 1]
        rad = np.sqrt(r0 ** 2 + ur * (r1 ** 2 - r0 ** 2))              # 面積一様
        for sct in range(NS):
            th = -np.pi + (sct + ur) / NS * 2 * np.pi
            RR, TT = np.meshgrid(rad, th, indexing="ij")
            bx[r * NS + sct] = (RR * np.cos(TT)).ravel(); by[r * NS + sct] = (RR * np.sin(TT)).ravel()
    return bx, by


_u = (np.arange(G)[:, None] + (np.arange(SUB) + 0.5)[None] / SUB) / G
UX = np.broadcast_to(_u[None, :, None, :], (G, G, SUB, SUB)).reshape(G * G, SUB * SUB)
UY = np.broadcast_to(_u[:, None, :, None], (G, G, SUB, SUB)).reshape(G * G, SUB * SUB)


def polar_bin(dx, dy, s):
    d = np.hypot(dx, dy) / s
    ld = np.log2(np.maximum(d, 1e-9))
    r = np.where(ld < L0, 0, 1 + ((ld - L0) / LS).astype(int)).clip(0, NR - 1)
    ang = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * NS
    sct = ang.astype(int).clip(0, NS - 1)
    return r * NS + sct


BX, BY = _bin_points()


def blur2(h, rows, cols, sigma, wrap_cols=False):
    """(rows*cols,) をぼかす(総和保存)。wrap_cols で列方向を周回"""
    if sigma <= 0:
        return h.astype(float)
    im = h.reshape(rows, cols).astype(np.float32)
    if wrap_cols:
        im = np.concatenate([im, im, im], 1)
    k = int(max(3, 2 * round(3 * sigma) + 1))
    out = cv2.GaussianBlur(im, (k, k), sigma, borderType=cv2.BORDER_REPLICATE if wrap_cols
                           else cv2.BORDER_CONSTANT).astype(float)
    if wrap_cols:
        out = out[:, cols:2 * cols]
    return out.ravel() * (h.sum() / max(out.sum(), 1e-12))


def size_of(D, i):
    return A_SIZE / D["scale"][i]


def cell_of(D, i):
    cx = min(max(int(D["x"][i] / D["Wd"][i] * G), 0), G - 1)
    cy = min(max(int(D["y"][i] / D["H"][i] * G), 0), G - 1)
    return cy * G + cx



def rel_size(D, i):
    """基準語の相対的な大きさ = 直径 / コマ短辺"""
    return 2 * A_SIZE / D["scale"][i] / min(D["H"][i], D["Wd"][i])


class Model:
    """学習側(src の panel 群)の度数と、ペアの関係の種類"""

    def __init__(self, D, panels, words_ok):
        self.words_ok = words_ok
        self.m1 = np.zeros(G * G)
        self.m2 = np.zeros((500, G * G))
        keys, bins, lds, refs = [], [], [], []
        for p, ii in panels.items():
            ii = [i for i in ii if words_ok[D["word"][i]]]
            for i in ii:
                c = cell_of(D, i); self.m1[c] += 1; self.m2[D["word"][i], c] += 1
            if len(ii) < 2:
                continue
            ii = np.array(ii)
            a, b = np.meshgrid(ii, ii, indexing="ij"); off = a != b; a, b = a[off], b[off]
            s = A_SIZE / D["scale"][a]
            dx, dy = D["x"][b] - D["x"][a], D["y"][b] - D["y"][a]
            keys.append(D["word"][a].astype(np.int64) * 500 + D["word"][b])
            bins.append(polar_bin(dx, dy, s)); lds.append(np.log2(np.maximum(np.hypot(dx, dy) / s, 1e-9)))
            refs.append(a)
        k = np.concatenate(keys); bn = np.concatenate(bins); ld = np.concatenate(lds); ra = np.concatenate(refs)
        # 基準語の相対的な大きさの 4 段階(学習側のまとまりの四分位)
        allc = np.array([i for ii in panels.values() for i in ii if words_ok[D["word"][i]]])
        self.size_edges = np.quantile([rel_size(D, i) for i in allc], [0.25, 0.5, 0.75])
        rc = np.searchsorted(self.size_edges, 2 * A_SIZE / D["scale"][ra] / np.minimum(D["H"][ra], D["Wd"][ra]))
        g = np.bincount(rc * NB + bn, minlength=4 * NB).reshape(4, NB).astype(float)
        self.gen = (g + 1e-3 * g.sum(1, keepdims=True) / NB) / (g.sum(1, keepdims=True) * (1 + 1e-3))
        uk, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
        self.pair_n = dict(zip(uk.tolist(), cnt.tolist()))
        order = np.argsort(inv, kind="stable"); starts = np.r_[0, np.cumsum(cnt)[:-1]]
        self.pair_bins, self.pair_dmed, self.pair_iqr = {}, {}, {}
        for j, key in enumerate(uk.tolist()):
            if cnt[j] < 5:
                continue
            sl = order[starts[j]:starts[j] + cnt[j]]
            self.pair_bins[key] = np.bincount(bn[sl], minlength=NB).astype(float)
            if cnt[j] >= KAPPA:
                q25, q50, q75 = np.percentile(ld[sl], [25, 50, 75])
                self.pair_dmed[key] = float(2 ** q50); self.pair_iqr[key] = float(q75 - q25)
        # 関係の種類(事前登録の境目)
        self.iqr_t1 = float(np.quantile(list(self.pair_iqr.values()), 1 / 3))
        self.pair_type = {}
        for key, dm in self.pair_dmed.items():
            iq = self.pair_iqr[key]
            self.pair_type[key] = 1 if (dm <= 4 and iq <= self.iqr_t1) else 2 if dm <= 16 else 3
        self.m1p = (self.m1 + 1e-3) / (self.m1 + 1e-3).sum()
        self.set_params(1.0, KAPPA, 0.5, KAPPA)

    def set_params(self, sa, ka, sr, kr, rel=True):
        self.sa, self.ka, self.sr, self.kr = sa, ka, sr, kr
        nb = self.m2.sum(1, keepdims=True)
        bl = np.stack([blur2(h, G, G, sa) for h in self.m2])
        pw = bl / np.maximum(bl.sum(1, keepdims=True), 1e-12)
        w = nb / (nb + ka)
        self.m2p = w * pw + (1 - w) * self.m1p[None]
        self.m2p = (self.m2p + 1e-9) / (self.m2p + 1e-9).sum(1, keepdims=True)
        self._pc = {}

    def pair_dist_key(self, key, cls):
        ck = (key, cls)
        if ck not in self._pc:
            n = self.pair_n.get(key, 0)
            if key in self.pair_bins:
                hb = blur2(self.pair_bins[key], NR, NS, self.sr, wrap_cols=True)
                w = n / (n + self.kr)
                self._pc[ck] = w * hb / hb.sum() + (1 - w) * self.gen[cls]
            else:
                self._pc[ck] = self.gen[cls]
        return self._pc[ck]

    def rel_cells(self, D, a, wb, x_a=None, y_a=None):
        """基準 a(位置は x_a,y_a で上書き可)から見た語 wb のコマ格子分布 (G*G,)"""
        key = int(D["word"][a]) * 500 + int(wb)
        cls = int(np.searchsorted(self.size_edges, rel_size(D, a)))
        p = self.pair_dist_key(key, cls); n = self.pair_n.get(key, 0)
        s = A_SIZE / D["scale"][a]
        xa = D["x"][a] if x_a is None else x_a; ya = D["y"][a] if y_a is None else y_a
        W, H = D["Wd"][a], D["H"][a]
        # 内側の環: 各区画の内部の点(面積一様)をコマ格子へ落とし、区画の確率を配る
        inner = slice(0, (NR - 1) * NS)
        px = xa + BX[inner] * s; py = ya + BY[inner] * s
        inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        cell = (np.clip((py / H * G).astype(int), 0, G - 1) * G + np.clip((px / W * G).astype(int), 0, G - 1))
        wgt = np.where(inside, (p[inner] / KB)[:, None], 0.0)
        q = np.bincount(cell.ravel(), weights=wgt.ravel(), minlength=G * G)
        # 最外環(=それより遠い全部): コマ側の標本点のうち最外環に入る点へ、方向ごとに等分
        bi = polar_bin(UX * W - xa, UY * H - ya, s)
        outer = bi >= (NR - 1) * NS
        if outer.any():
            cnt = np.bincount(bi[outer], minlength=NB)
            q = q + np.where(outer, p[bi] / np.maximum(cnt[bi], 1), 0.0).sum(1)
        q = q + 1e-12
        return q / q.sum(), n

    def rank(self, D, a, wb):
        """基準としての優先度(小さいほど優先): (種類, IQR, −大きさ)。関係が無ければ None"""
        key = int(D["word"][a]) * 500 + int(wb)
        t = self.pair_type.get(key)
        if t is None:
            return None
        return (t, self.pair_iqr[key], D["scale"][a])

    def best_ref(self, D, cands, wb, only_type=None):
        best, br = None, None
        for a in cands:
            r = self.rank(D, a, wb)
            if r is None or (only_type is not None and r[0] != only_type):
                continue
            if br is None or r < br:
                best, br = a, r
        return best, br


def bits(p, c):
    return -math.log2(max(p[c], 1e-300))


def eval_orderfree(M, D, panels, rng=None, swap_type1=False, only_m2=False, by_type=False):
    """順序なし。全体(M0/M1/M2/M3a/M3b)と、種類別(各種類から基準を 1 つ選んだ M2 / M3a)"""
    out = defaultdict(list)
    stats = dict(n=0, no_ref=0)
    for p, ii in panels.items():
        ii = [i for i in ii if M.words_ok[D["word"][i]]]
        if len(ii) < 2:
            continue
        acc = defaultdict(float)
        for t in ii:
            c = cell_of(D, t); wb = D["word"][t]
            acc["M0"] += math.log2(G * G); acc["M1"] += bits(M.m1p, c)
            p2 = M.m2p[wb]; acc["M2"] += bits(p2, c)
            if only_m2:
                continue
            cands = [a for a in ii if a != t]
            assert t not in cands                                    # T3
            cache = {}

            def rel(a, swap=False):
                if (a, swap) not in cache:
                    xa = ya = None
                    if swap:
                        o = [j for j in ii if j not in (t, a)]
                        if o:
                            j = o[rng.integers(len(o))]; xa, ya = D["x"][j], D["y"][j]
                    cache[(a, swap)] = M.rel_cells(D, a, wb, xa, ya)
                return cache[(a, swap)]
            a, _ = M.best_ref(D, cands, wb)
            stats["n"] += 1
            if a is None:
                stats["no_ref"] += 1
                acc["M3a"] += bits(p2, c); acc["M3b"] += bits(p2, c)
            else:
                q, n = rel(a); w = n / (n + M.kr)
                acc["M3a"] += bits(q, c); acc["M3b"] += bits(w * q + (1 - w) * p2, c)
            if by_type or swap_type1:
                for ty in ((1,) if swap_type1 and not by_type else (1, 2, 3)):
                    at, _ = M.best_ref(D, cands, wb, only_type=ty)
                    if at is None:
                        continue
                    q, _n = rel(at, swap=swap_type1)
                    acc[f"T{ty}_M3a"] += bits(q, c); acc[f"T{ty}_M2"] += bits(p2, c); acc[f"T{ty}_n"] += 1
        for k in ("M0", "M1", "M2", "M3a", "M3b", "T1_M3a", "T1_M2", "T1_n", "T2_M3a", "T2_M2", "T2_n",
                  "T3_M3a", "T3_M2", "T3_n"):
            out[k].append(acc.get(k, 0.0))
        out["n"].append(len(ii))
    return {k: np.array(v) for k, v in out.items()}, stats


def eval_order(M, D, panels, mode, rng=None):
    tot = []; ns = []
    for p, ii in panels.items():
        ii = [i for i in ii if M.words_ok[D["word"][i]]]
        if len(ii) < 2:
            continue
        by_size = sorted(ii, key=lambda i: (D["scale"][i], i))
        if mode == "O1":
            seq = by_size
        elif mode == "O3":
            seq = [int(x) for x in rng.permutation(ii)]
        else:                                   # O2: 配置済みと最も優先度の高い関係を持つ未配置の語を次に
            seq = [by_size[0]]; rest = set(ii) - {by_size[0]}
            while rest:
                best, br = None, None
                for t in sorted(rest):
                    _, r = M.best_ref(D, seq, D["word"][t])
                    if r is not None and (br is None or r < br):
                        best, br = t, r
                if best is None:
                    best = next(i for i in by_size if i in rest)
                seq.append(best); rest.discard(best)
        s = 0.0
        for k, t in enumerate(seq):
            c = cell_of(D, t); wb = D["word"][t]; p2 = M.m2p[wb]
            if k == 0:
                s += bits(p2, c); continue
            a, _ = M.best_ref(D, seq[:k], wb)
            if a is None:
                s += bits(p2, c); continue
            q, n = M.rel_cells(D, a, wb); w = n / (n + M.kr)
            s += bits(w * q + (1 - w) * p2, c)
        tot.append(s); ns.append(len(ii))
    return np.array(tot), np.array(ns)


def boot_diff(a, b, n, rng, B=1000):
    k = len(n); idx = rng.integers(0, k, (B, k))
    d = (a[idx].sum(1) - b[idx].sum(1)) / np.maximum(n[idx].sum(1), 1)
    return float((a.sum() - b.sum()) / max(n.sum(), 1)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


SIG_A, KAP_A = (0.5, 1.0, 2.0, 4.0, 8.0), (5.0, 20.0, 80.0, 320.0, 1280.0)
SIG_R, KAP_R = (0.0, 0.5, 1.0, 2.0), (5.0, 20.0, 80.0, 320.0)


def tune(D, src_panels, ok, groups, n_eval=300):
    """学習側の群の中だけで、シリーズを 2 つに分けた交差検証で M2・M3a の平滑化を独立に選ぶ"""
    gs = sorted(set(groups[p] for p in src_panels), key=lambda g: (-sum(groups[p] == g for p in src_panels), g))
    fold = {g: k % 2 for k, g in enumerate(gs)}
    F = [{p: v for p, v in src_panels.items() if fold[groups[p]] == f} for f in (0, 1)]
    sc2 = defaultdict(float); sc3 = defaultdict(float)
    for f in (0, 1):
        M = Model(D, F[f], ok)
        ev = {p: F[1 - f][p] for p in sorted(F[1 - f])[:n_eval]}
        for sg in SIG_A:
            for kp in KAP_A:
                M.set_params(sg, kp, M.sr, M.kr, rel=False)
                r, _ = eval_orderfree(M, D, ev, only_m2=True)
                sc2[(sg, kp)] += r["M2"].sum() / r["n"].sum() / 2
        for sg in SIG_R:
            for kp in KAP_R:
                M.set_params(1.0, KAPPA, sg, kp)
                r, _ = eval_orderfree(M, D, ev)
                sc3[(sg, kp)] += r["M3a"].sum() / r["n"].sum() / 2
    b2 = min(sc2, key=sc2.get); b3 = min(sc3, key=sc3.get)
    edge = dict(M2=b2[0] in (SIG_A[0], SIG_A[-1]) or b2[1] in (KAP_A[0], KAP_A[-1]),
                M3a=b3[0] in (SIG_R[0], SIG_R[-1]) or b3[1] in (KAP_R[0], KAP_R[-1]))
    return b2, b3, edge, {str(k): v for k, v in sc2.items()}, {str(k): v for k, v in sc3.items()}


def type_overlays(D, P, M_, M, panels, out, rng):
    """種類ごとに代表ペア(出現の多い順に 8 つ)を、a 中心・b の実インスタンスを相対位置に重ねて描く"""
    from placement_gt import pair_overlay
    A, Bj, Pn = [], [], []
    for p, ii in panels.items():
        ii = np.array([i for i in ii if M.words_ok[D["word"][i]]])
        if len(ii) < 2:
            continue
        a, b = np.meshgrid(ii, ii, indexing="ij"); off = a != b
        A.append(a[off]); Bj.append(b[off]); Pn.append(np.full(off.sum(), p))
    A, Bj, Pn = np.concatenate(A), np.concatenate(Bj), np.concatenate(Pn)
    for ty in (1, 2, 3):
        ks = sorted([k for k, t in M.pair_type.items() if t == ty], key=lambda k: -M.pair_n[k])
        ks = [k for k in ks if k // 500 != k % 500][:8]              # 同語ペアは除く
        ims = []
        for k in ks:
            r = dict(a=k // 500, b=k % 500, panels=M.pair_n[k])
            im = pair_overlay(P, M_, D, (A, Bj, None, Pn), r, rng)
            cv2.putText(im, f"d_med {M.pair_dmed[k]:.1f} IQR {M.pair_iqr[k]:.2f}", (4, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
            ims.append(im)
        if not ims:
            continue
        ims += [np.full_like(ims[0], 255)] * (-len(ims) % 4)
        grid = np.vstack([np.hstack([np.pad(im, ((0, 4), (0, 4), (0, 0)), constant_values=200)
                                     for im in ims[i:i + 4]]) for i in range(0, len(ims), 4)])
        cv2.imwrite(str(out / f"pairs_type{ty}.png"), grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/relpos_20260925")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=20260925)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    _cm, P, W_, M_, D = load_data(dev)
    keep, n_drop = dedupe(D)
    rel = {int(r["word"]) for r in csv.DictReader(open("results/word_reliability_v2_20260924/word_reliability_v2.csv"))
           if r["verdict"] == "信頼"}
    print(f"重複除去 {n_drop} コマ / 信頼語 {len(rel)}", flush=True)
    halves = {}
    for h in (0, 1):
        d = defaultdict(list)
        for i in np.flatnonzero(keep & (D["half"] == h)):
            d[int(D["panel"][i])].append(int(i))
        halves[h] = dict(d)
    groups = {int(D["panel"][i]): D["group"][i] for i in range(len(D["panel"]))}
    rng = np.random.default_rng(a.seed)
    sets = {"全語": np.ones(500, bool), "信頼49語": np.isin(np.arange(500), list(rel))}
    res, tuned, checks = {}, {}, {}
    for sname, ok in sets.items():
        for src, dst, lab in ((0, 1, "A→B"), (1, 0, "B→A")):
            key = f"{sname} {lab}"
            M = Model(D, halves[src], ok)
            b2, b3, edge, s2, s3 = tune(D, halves[src], ok, groups, n_eval=100 if a.check else 300)
            M.set_params(b2[0], b2[1], b3[0], b3[1])
            tcount = {t: sum(v == t for v in M.pair_type.values()) for t in (1, 2, 3)}
            tuned[key] = dict(M2=b2, M3a=b3, edge=edge, cv_M2=s2, cv_M3a=s3, iqr_t1=M.iqr_t1,
                              size_edges=M.size_edges.tolist(), type_counts=tcount)
            print(f"調整 {key}: M2 σ,κ={b2} / M3a σ,κ={b3} / 候補の端 {edge} / 種類のペア数 {tcount}", flush=True)
            ev = halves[dst]
            if a.check:
                ks = sorted(ev)[:200]; ev = {k: ev[k] for k in ks}
                Mt = Model(D, ev, ok); Mt.set_params(M.sa, M.ka, M.sr, M.kr, rel=False)
                r, _ = eval_orderfree(Mt, D, ev, only_m2=True)
                n = r["n"].sum(); m1p = Mt.m1p
                H1 = float(-(m1p[m1p > 0] * np.log2(m1p[m1p > 0])).sum())
                r0, st = eval_orderfree(M, D, ev, by_type=True)
                r2, _ = eval_orderfree(M, D, ev, rng=np.random.default_rng(0), swap_type1=True)
                n0 = r0["n"].sum(); n1 = r0["T1_n"]
                t1m3 = r0["T1_M3a"].sum() / max(n1.sum(), 1)
                t2 = boot_diff(r2["T1_M3a"], r0["T1_M3a"], n1, rng)
                c = dict(M0=r["M0"].sum() / n, M1=r["M1"].sum() / n, H1=H1, M2=r0["M2"].sum() / n0,
                         M3b=r0["M3b"].sum() / n0, T1_M3a=t1m3, T1_M2=r0["T1_M2"].sum() / max(n1.sum(), 1),
                         T1_n=int(n1.sum()), T2_diff=t2)
                c["T1b"] = bool(c["T1_M3a"] <= 8.0 and c["M3b"] <= 8.0); c["T2"] = bool(t2[1] > 0)
                checks[key] = c
                print(f"T1 {key}: M0 {c['M0']:.3f} (8.000) / M1 {c['M1']:.3f} (周辺 {H1:.3f})")
                print(f"T1b/T2 {key}: 種類(1) n={c['T1_n']} M3a {c['T1_M3a']:.3f} M2 {c['T1_M2']:.3f} / 全体 M3b "
                      f"{c['M3b']:.3f} / 入替の悪化 {t2[0]:+.3f} [{t2[1]:+.3f}, {t2[2]:+.3f}] → T1b {c['T1b']} T2 {c['T2']}",
                      flush=True)
                continue
            r, st = eval_orderfree(M, D, ev, by_type=True)
            n = r["n"]
            row = {k: float(r[k].sum() / n.sum()) for k in ("M0", "M1", "M2", "M3a", "M3b")}
            row["no_ref_rate"] = st["no_ref"] / max(st["n"], 1)
            row["n_clusters"] = int(n.sum()); row["n_panels"] = int(len(n))
            row["M2-M3a"] = boot_diff(r["M2"], r["M3a"], n, rng)
            row["M2-M3b"] = boot_diff(r["M2"], r["M3b"], n, rng)
            for ty in (1, 2, 3):
                nt = r[f"T{ty}_n"]
                row[f"type{ty}"] = dict(n=int(nt.sum()), M2=float(r[f"T{ty}_M2"].sum() / max(nt.sum(), 1)),
                                        M3a=float(r[f"T{ty}_M3a"].sum() / max(nt.sum(), 1)),
                                        gain=boot_diff(r[f"T{ty}_M2"], r[f"T{ty}_M3a"], nt, rng))
            o1, no = eval_order(M, D, ev, "O1"); o2, _ = eval_order(M, D, ev, "O2")
            o3 = [eval_order(M, D, ev, "O3", np.random.default_rng(a.seed + k))[0] for k in range(10)]
            o3m = np.mean(o3, 0)
            row["O1"] = float(o1.sum() / no.sum()); row["O2"] = float(o2.sum() / no.sum())
            row["O3"] = [float(x.sum() / no.sum()) for x in o3]
            row["O3m-O1"] = boot_diff(o3m, o1, no, rng); row["O3m-O2"] = boot_diff(o3m, o2, no, rng)
            res[key] = row
            print(key, json.dumps({k: v for k, v in row.items() if k != "O3"}, ensure_ascii=False, default=str),
                  flush=True)
            print(f"   O3 10通り: {min(row['O3']):.4f}–{max(row['O3']):.4f}", flush=True)
            json.dump(dict(results=res, tuned=tuned), open(out / "relpos.json", "w"), indent=1,
                      ensure_ascii=False, default=str)
    if a.check:
        json.dump(dict(checks=checks, tuned=tuned), open(out / "check.json", "w"), indent=1, ensure_ascii=False,
                  default=str)
        return
    M = Model(D, halves[0], sets["全語"])
    t0 = tuned["全語 A→B"]; M.set_params(t0["M2"][0], t0["M2"][1], t0["M3a"][0], t0["M3a"][1])
    type_overlays(D, P, M_, M, halves[0], out, np.random.default_rng(5))
    print("out ->", out)


if __name__ == "__main__":
    main()
