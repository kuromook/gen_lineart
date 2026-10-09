#!/usr/bin/env python
"""位置ビンの歪みの影響範囲(2026-10-09 事前登録。共通基盤 第3回通達の割り当て)。

corpus の位置ビンは prep_grammar_corpus.py で
  sy = cy/dims[0]、sx = cx/dims[1]
と作られている。strokes が (y, x) 順なので dims = (W, H)、meta は cy = y・cx = x。
よって sy = y/W、sx = x/H でどちらも分母が食い違う。
補正は「dims のどちらの列で割るか」を入れ替えるだけ: sy = y/H、sx = x/W。

コマ内の並び順 (-(1/scale), cy, cx) は分母に依存しないので、語・scale・並び・分割・
コマ集合は不変で、位置の 2 トークンだけが変わる。

--check : 道具確認 K1a / K1b / N1 / N2 / N3 / K2(学習なし)
--k1c   : 原corpus での再学習が記録値を再現するか(cloze 1 本)
(既定)  : 道具確認 → 合格なら corpus_fixed.npz を書く → 本測定 → 図

Track F のファイルは読むだけで、一切書かない。
"""
import argparse, csv, json, shutil, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TRACKF = Path("/home/sh1/deepl/lineart-stroke-grammar")
CLUSTERS = TRACKF / "results/cluster_set_20260919"
CORPUS = TRACKF / "results/grammar_corpus_20260920/corpus.npz"
FACES = Path("/home/sh1/deepl/lineart-face-words/results/h3_face_parts_20261004/faces.csv")

OFF_Y, OFF_X, OFF_S = 512, 528, 544
BOS, EOS = 556, 557
N_BINS = 16
S_BINS, S_LO, S_HI = 12, -7.0, 4.0
SEED = 20261009

# 記録値(事前登録に明記したもの)
REC = dict(unigram=7.348380741334939, pos_marginal=7.562884283205894,
           cloze_pos=6.314605334367116, cloze_word=7.398188148710611,
           cloze_tags_pos=6.353530597693522, cloze_type_pos=6.387861989437070,
           twostage_pos=6.658, setar_pos=6.680)
TOL_BITS = 0.030          # K1c の許容
BIG, SMALL = 0.270, 0.090  # 判定の閾値(事前登録)


# ---------------------------------------------------------------- 並びの再現
def meta_order():
    """prep_grammar_corpus が見た行の並び = train(meta 順) のあと test(meta 順)"""
    rows = list(csv.DictReader(open(CLUSTERS / "meta.csv")))
    tr = [i for i, r in enumerate(rows) if r["split"] == "train"]
    te = [i for i, r in enumerate(rows) if r["split"] == "test"]
    return rows, np.array(tr + te)


def panels_in_order(rows, order):
    """コマごとに (meta 行番号, scale, y, x) を prep と同じ並びで返す。パネル ID 昇順"""
    by = defaultdict(list)
    for i in order:
        r = rows[i]
        by[int(r["panel"])].append((i, float(r["scale"]), float(r["cy"]), float(r["cx"])))
    out = []
    for pid in sorted(by):
        cs = sorted(by[pid], key=lambda t: (-(1.0 / t[1]), t[2], t[3]))   # 安定ソート
        out.append((pid, cs))
    return out


def bins_of(cs, dims_row, mode):
    """mode: orig(y/W, x/H) / fixed(y/H, x/W) / mirror(x/W, y/H)"""
    d0, d1 = float(dims_row[0]), float(dims_row[1])          # dims = (W, H)
    y = np.array([c[2] for c in cs]); x = np.array([c[3] for c in cs])
    if mode == "orig":
        fy, fx = y / max(d0, 1), x / max(d1, 1)              # y/W, x/H
    elif mode == "fixed":
        fy, fx = y / max(d1, 1), x / max(d0, 1)              # y/H, x/W
    elif mode == "mirror":
        fy, fx = x / max(d0, 1), y / max(d1, 1)              # x/W, y/H(比は正しく軸が鏡像)
    else:
        raise ValueError(mode)
    sy = np.minimum((fy * N_BINS).astype(int), N_BINS - 1)
    sx = np.minimum((fx * N_BINS).astype(int), N_BINS - 1)
    return sy, sx


def scale_bin(cs):
    sc = np.array([c[1] for c in cs])
    return np.clip(((np.log2(sc) - S_LO) / (S_HI - S_LO) * S_BINS).astype(int), 0, S_BINS - 1)


def build_seq(pans, dims, words, mode, L):
    """prep と同じ形のトークン行列を作る"""
    seq = np.full((len(pans), L), -1, np.int16)
    for i, (pid, cs) in enumerate(pans):
        sy, sx = bins_of(cs, dims[pid], mode)
        ss = scale_bin(cs)
        w = words[[c[0] for c in cs]]
        toks = np.empty(len(cs) * 4 + 2, np.int64)
        toks[0] = BOS
        toks[1:-1:4] = w; toks[2:-1:4] = OFF_Y + sy
        toks[3:-1:4] = OFF_X + sx; toks[4:-1:4] = OFF_S + ss
        toks[-1] = EOS
        seq[i, :min(len(toks), L)] = toks[:L]
    return seq


# ---------------------------------------------------------------- 周辺 bits
def marginal_bits(seq, split_mask):
    """common.baselines と同式: H(py) + H(px)。train 側のみ"""
    tr = seq[split_mask]
    yc = np.zeros(N_BINS); xc = np.zeros(N_BINS); wc = np.zeros(500); n_eos = 0
    for row in tr:
        t = row[row >= 0]
        cl = t[1:-1].reshape(-1, 4)
        wc += np.bincount(cl[:, 0], minlength=500)
        yc += np.bincount(cl[:, 1] - OFF_Y, minlength=N_BINS)
        xc += np.bincount(cl[:, 2] - OFF_X, minlength=N_BINS)
        n_eos += 1
    h = lambda p: float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    uni = wc / wc.sum()
    uni_f = np.append(wc, n_eos); uni_f = uni_f / uni_f.sum()
    return h(yc / yc.sum()) + h(xc / xc.sum()), h(uni_f), yc / yc.sum(), xc / xc.sum()


def top_bin_share(seq):
    """最上位ビン(15)の占有率(合計 / y / x)"""
    yv = seq[:, 2::4]; xv = seq[:, 3::4]
    yv = yv[yv >= 0] - OFF_Y; xv = xv[xv >= 0] - OFF_X
    n = len(yv) + len(xv)
    return float(((yv == N_BINS - 1).sum() + (xv == N_BINS - 1).sum()) / n), \
        float((yv == N_BINS - 1).mean()), float((xv == N_BINS - 1).mean())


def clipped_share(pans, dims, mode):
    """比が 1 以上で切り詰められた割合(y, x)。2026-09-23 の記録 15.9% / 20.1% と同じ量"""
    ny = nx = cy_ = cx_ = 0
    for pid, cs in pans:
        d0, d1 = float(dims[pid][0]), float(dims[pid][1])
        y = np.array([c[2] for c in cs]); x = np.array([c[3] for c in cs])
        if mode == "orig":
            fy, fx = y / max(d0, 1), x / max(d1, 1)
        elif mode == "fixed":
            fy, fx = y / max(d1, 1), x / max(d0, 1)
        else:
            fy, fx = x / max(d0, 1), y / max(d1, 1)
        cy_ += int((fy >= 1).sum()); cx_ += int((fx >= 1).sum()); ny += len(y); nx += len(x)
    return cy_ / ny, cx_ / nx


# ---------------------------------------------------------------- 道具確認
def encode_words(dev="cuda"):
    """cluster_set 全行を meta 順に符号化(prep_grammar_corpus と同じ経路)"""
    import torch
    sys.path.insert(0, str(HERE.parent / "trackf")); sys.path.insert(0, str(HERE))
    from common import CODEBOOK                                        # noqa: E402
    from train_codebook2 import Codebook2                              # noqa: E402
    P = np.load(CLUSTERS / "pts.npy", mmap_mode="r")
    W = np.load(CLUSTERS / "width.npy", mmap_mode="r")
    M = np.load(CLUSTERS / "mask.npy", mmap_mode="r")
    cb = torch.load(CODEBOOK, map_location=dev, weights_only=False)
    cm = Codebook2(levels=tuple(cb["levels"]), coord_bins=cb["args"].get("coord_bins", 0)).to(dev)
    cm.load_state_dict(cb["model"]); cm.eval(); cm.rfsq.stages = cb["args"].get("stages", 3)
    ws = []
    with torch.no_grad():
        for i in range(0, len(P), 4096):
            p = torch.from_numpy(np.asarray(P[i:i + 4096])).to(dev).float()
            w = torch.from_numpy(np.asarray(W[i:i + 4096])).to(dev).float()
            m = torch.from_numpy(np.asarray(M[i:i + 4096])).to(dev)
            ws.append(cm.encode(p, w, m, 1)[1].cpu().numpy())
    return np.concatenate(ws)


def k2_dims_vs_png(dims, pans):
    """K2: dims の順を線画 PNG の寸法(解析系の外)と照合。閾値は事前登録どおり"""
    png = {}
    for t in csv.DictReader(open(FACES)):
        if t["png_h"]:
            png[int(t["panel"])] = (int(t["png_h"]), int(t["png_w"]))
    pid = [p for p, _ in pans if p in png]
    d = dims[pid]
    ph = np.array([png[p][0] for p in pid]); pw = np.array([png[p][1] for p in pid])
    SL = 2            # panel_dims() は ceil(インク最大) + 2 なので PNG を最大 1px 上回る
    as_named = float((((d[:, 0] <= pw + SL) & (d[:, 1] <= ph + SL))).mean())   # dims = (W, H)
    swapped = float((((d[:, 0] <= ph + SL) & (d[:, 1] <= pw + SL))).mean())    # (H, W) と読む
    nonsq = float((np.abs(ph - pw) > 0.05 * np.minimum(ph, pw)).mean())
    return (as_named >= 0.990 and swapped <= 0.700), dict(
        n=len(pid), as_WH=round(as_named, 4), as_HW=round(swapped, 4),
        nonsquare=round(nonsq, 4), slack_px=SL,
        ratio_WH=[round(float(np.median(d[:, 0] / pw)), 4), round(float(np.median(d[:, 1] / ph)), 4)],
        ratio_HW=[round(float(np.median(d[:, 0] / ph)), 4), round(float(np.median(d[:, 1] / pw)), 4)],
        thresholds=dict(as_WH_min=0.990, as_HW_max=0.700))


def run_checks(out, dev="cuda"):
    z = np.load(CORPUS)
    seq0, split, pid_c, dims = z["seq"], z["split"], z["panels"], z["dims"]
    L = seq0.shape[1]
    rows, order = meta_order()
    pans = panels_in_order(rows, order)
    chk = {}

    # K1a: 原式で作り直した seq が既存と全トークン一致
    words = encode_words(dev)
    seqA = build_seq(pans, dims, words, "orig", L)
    same = bool(seqA.shape == seq0.shape and np.array_equal(seqA, seq0))
    diff = int((seqA != seq0).sum()) if seqA.shape == seq0.shape else -1
    pid_ok = bool(np.array_equal(np.array([p for p, _ in pans]), pid_c))
    chk["K1a"] = dict(passed=bool(same and pid_ok), tokens_total=int((seq0 >= 0).sum()),
                      tokens_differing=diff, panel_ids_match=pid_ok, shape=list(seqA.shape))
    print("K1a(原式で seq 完全一致):", "合格" if chk["K1a"]["passed"] else "不合格",
          f"差分トークン {diff}", flush=True)
    if not chk["K1a"]["passed"]:
        json.dump(chk, open(out / "check.json", "w"), ensure_ascii=False, indent=1); return chk, None

    # K1b: 周辺と unigram
    mb, ub, _, _ = marginal_bits(seq0, split)
    ok_b = bool(abs(mb - REC["pos_marginal"]) < 5e-7 and abs(ub - REC["unigram"]) < 5e-7)
    chk["K1b"] = dict(passed=ok_b, pos_marginal=mb, unigram=ub,
                      recorded=dict(pos_marginal=REC["pos_marginal"], unigram=REC["unigram"]))
    print("K1b(周辺/unigram の再現):", "合格" if ok_b else "不合格",
          f"{mb:.6f} / {ub:.6f}", flush=True)

    # N1 / N2 / N3: 誤りの型ごとの陰性対照
    seqF = build_seq(pans, dims, words, "fixed", L)
    seqM = build_seq(pans, dims, words, "mirror", L)
    sat = {k: top_bin_share(s) for k, s in (("orig", seqA), ("fixed", seqF), ("mirror", seqM))}
    clip = {k: clipped_share(pans, dims, k) for k in ("orig", "fixed", "mirror")}
    # N1: 判定する量は「切り詰められた割合」(2026-09-23 の記録 15.9% / 20.1% と同じ量)
    n1 = bool(abs(clip["orig"][0] - 0.159) <= 0.005 and abs(clip["orig"][1] - 0.201) <= 0.005)
    chk["N1"] = dict(passed=n1, clipped_orig=[round(v, 4) for v in clip["orig"]],
                     recorded=[0.159, 0.201], tolerance=0.005,
                     clipped_fixed=[round(v, 4) for v in clip["fixed"]],
                     top_bin_orig=[round(v, 4) for v in sat["orig"]],
                     top_bin_fixed=[round(v, 4) for v in sat["fixed"]])
    print("N1(分母の取り違えで切り詰め):", "合格" if n1 else "不合格",
          f"{clip['orig'][0]:.4f} / {clip['orig'][1]:.4f}(記録 0.159 / 0.201)", flush=True)
    # N2: 合否ではなく実演。鏡像は補正版の 2 トークンを入れ替えただけなので対称な統計量は定義上同一
    mb_f, _, _, _ = marginal_bits(seqF, split); mb_m, _, _, _ = marginal_bits(seqM, split)
    # トークン ID ではなくビン番号で比べる(OFF_Y と OFF_X が 16 違う)。
    # 2::4 と 3::4 は長さが 1 違うので短い方に揃える
    def _bins(sq, off, step):
        v = sq[:, step::4]
        return np.where(v >= 0, v - off, -1)
    my, mx = _bins(seqM, OFF_Y, 2), _bins(seqM, OFF_X, 3)
    fy_, fx_ = _bins(seqF, OFF_Y, 2), _bins(seqF, OFF_X, 3)
    k = min(my.shape[1], mx.shape[1], fy_.shape[1], fx_.shape[1])
    swapped_identical = bool(np.array_equal(my[:, :k], fx_[:, :k])
                             and np.array_equal(mx[:, :k], fy_[:, :k]))
    chk["N2_demo"] = dict(judged=False, mirror_is_fixed_with_tokens_swapped=swapped_identical,
                          top_bin_fixed=round(sat["fixed"][0], 4), top_bin_mirror=round(sat["mirror"][0], 4),
                          marginal_fixed=round(mb_f, 6), marginal_mirror=round(mb_m, 6),
                          clipped_fixed=[round(v, 4) for v in clip["fixed"]],
                          clipped_mirror=[round(v, 4) for v in clip["mirror"]],
                          note="鏡像は補正版の位置2トークンを入れ替えただけ。対称な統計量は定義上同一で、"
                               "この対照は設計上必ず通る=空振り。鏡像を捕まえるのは外部の基準に当てた K2 だけ")
    n2 = True
    print("N2(実演・合否にしない): 鏡像 = 補正版のトークン入替", swapped_identical,
          f"| 飽和 {sat['fixed'][0]:.4f} 対 {sat['mirror'][0]:.4f}"
          f" | 周辺 {mb_f:.6f} 対 {mb_m:.6f}", flush=True)

    rng = np.random.default_rng(SEED)
    seqR = seq0.copy()
    m = seqR[:, 2::4] >= 0
    seqR[:, 2::4][m] = OFF_Y + rng.integers(0, N_BINS, m.sum())
    m2 = seqR[:, 3::4] >= 0
    seqR[:, 3::4][m2] = OFF_X + rng.integers(0, N_BINS, m2.sum())
    mr, _, _, _ = marginal_bits(seqR, split)
    sr = top_bin_share(seqR)[0]
    n3 = bool(abs(mr - 8.0) <= 0.010 and abs(sr - 0.0625) <= 0.005)
    chk["N3"] = dict(passed=n3, uniform_marginal=round(mr, 5), uniform_top_bin=round(sr, 5))
    print("N3(採点の健全性):", "合格" if n3 else "不合格", f"{mr:.5f} / {sr:.5f}", flush=True)

    ok2, det2 = k2_dims_vs_png(dims, pans)
    chk["K2"] = dict(passed=ok2, **det2)
    print("K2(dims の順を線画 PNG と照合):", "合格" if ok2 else "不合格",
          f"(W,H) {det2['as_WH']} / (H,W) {det2['as_HW']}", flush=True)

    chk["passed_no_training"] = bool(chk["K1a"]["passed"] and ok_b and n1 and n2 and n3 and ok2)
    json.dump(chk, open(out / "check.json", "w"), ensure_ascii=False, indent=1)
    print("道具確認(学習なしの部分):", "合格" if chk["passed_no_training"] else "不合格", flush=True)
    return chk, (seq0, seqF, split, pid_c, dims, pans, words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/posbins_20261009")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    chk, data = run_checks(out)
    if data is None or not chk.get("passed_no_training"):
        print("主測定に進まず停止。", flush=True); return 1
    if a.check:
        return 0
    seq0, seqF, split, pid_c, dims, pans, words = data
    np.savez_compressed(out / "corpus_fixed.npz", seq=seqF, split=split, panels=pid_c, dims=dims)
    print("書き出し:", out / "corpus_fixed.npz", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
