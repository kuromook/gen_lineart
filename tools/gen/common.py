#!/usr/bin/env python
"""Track G 生成モデル共通: corpus 読み込み・ビン定数・baseline・語原型復号・描画。
ビン化・正準順・bits 定義はすべて Track F 仕様を踏襲
(prep_grammar_corpus.py / train_cloze.py / render_cloze.py と同式)。
"""
from pathlib import Path

import numpy as np
import torch

TRACKF = Path("/home/sh1/deepl/lineart-stroke-grammar")
CORPUS = TRACKF / "results/grammar_corpus_20260920/corpus.npz"
LABELS = TRACKF / "results/panel_composition_20260921/labels_k12.npy"
TAGFEATS = TRACKF / "results/panel_composition_20260921/tags_top40.npy"
CODEBOOK = TRACKF / "results/invariance_20260920/l4_w500/codebook2.pt"

N_WORDS = 500
OFF_Y, OFF_X, OFF_S = 512, 528, 544
BOS, EOS = 556, 557
EOS_W = 500                     # 語 head に付けた生成停止クラス (0..499 が語彙)
MAXS = 170                      # 1コマの最大まとまり数
N_BINS = 16                     # pos_y / pos_x のビン数
S_BINS, S_LO, S_HI = 12, -7.0, 4.0   # scale ビン (log2 scale)


def unpack(row):
    """corpus 行 -> (w, py, px, sc) int64 配列 (Track F train_cloze.unpack と同じ)"""
    t = row[row >= 0]
    cl = t[1:-1].reshape(-1, 4)
    return (cl[:, 0].astype(np.int64), (cl[:, 1] - OFF_Y).astype(np.int64),
            (cl[:, 2] - OFF_X).astype(np.int64), (cl[:, 3] - OFF_S).astype(np.int64))


def load_all(limit=0):
    """corpus + 型ラベル + タグを読み、まとまり列のリストを返す。
    limit>0 のとき学習側インデックスだけ先頭 limit コマに絞る(評価は常に test 全量)。"""
    z = np.load(CORPUS)
    seq, split = z["seq"], z["split"]
    lab = np.load(LABELS).astype(np.int64)
    tg = np.load(TAGFEATS).astype(np.float32)
    assert len(lab) == len(seq) and len(tg) == len(seq), "labels/tags と corpus の行数が不一致"
    panels = [unpack(r) for r in seq]
    itr, ite = np.flatnonzero(split), np.flatnonzero(~split)
    if limit:
        itr = itr[:limit]
    return dict(panels=panels, lab=lab, tg=tg, itr=itr, ite=ite,
                n_types=int(lab.max()) + 1, n_tagf=tg.shape[1])


def baselines(panels, idx):
    """train 頻度から unigram エントロピーと pos 周辺分布の bits。
    Track F 式(tr[:,1::4] に EOS=557 が混入する)と EOS なし(生成モデルの語 bits と
    同じ母集団)の両方を返す。pos 周辺は両式で一致(7.563)。"""
    wc = np.zeros(N_WORDS); yc = np.zeros(N_BINS); xc = np.zeros(N_BINS)
    n_eos = 0
    for i in idx:
        w, py, px, _sc = panels[i]
        wc += np.bincount(w[w < N_WORDS], minlength=N_WORDS)
        n_eos += 1                              # 各コマ末尾に EOS が1つ
        yc += np.bincount(py[py < N_BINS], minlength=N_BINS)
        xc += np.bincount(px[px < N_BINS], minlength=N_BINS)
    uni = wc / wc.sum()
    H_uni = float(-(uni[uni > 0] * np.log2(uni[uni > 0])).sum())
    p_eos = n_eos / (wc.sum() + n_eos)
    uni_f = np.append(wc, n_eos); uni_f = uni_f / uni_f.sum()
    H_uni_f = float(-(uni_f[uni_f > 0] * np.log2(uni_f[uni_f > 0])).sum())
    yp, xp = yc / yc.sum(), xc / xc.sum()
    H_pos = float(-(yp[yp > 0] * np.log2(yp[yp > 0])).sum()
                  - (xp[xp > 0] * np.log2(xp[xp > 0])).sum())
    return H_uni, H_pos, H_uni_f


def load_protos(dev):
    """語ID→原型ストローク座標+keep マスク (Track F render_cloze.py:46-59 と同じ復号経路)。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trackf"))
    from train_codebook2 import Codebook2, topn_mask
    cb = torch.load(CODEBOOK, map_location=dev, weights_only=False)
    levels = tuple(cb["levels"])
    cm = Codebook2(levels=levels, coord_bins=cb["args"].get("coord_bins", 0)).to(dev)
    cm.load_state_dict(cb["model"]); cm.eval(); cm.rfsq.stages = cb["args"].get("stages", 3)
    basis = np.cumprod((1,) + levels[:-1])
    half_w = np.floor(np.array(levels) / 2)
    qq = np.zeros((N_WORDS, len(levels)), np.float32)
    for w in range(N_WORDS):
        d = (w // basis) % np.array(levels)
        qq[w] = (d - half_w) / half_w
    with torch.no_grad():
        ex, pp, _pw, cl, _lg = cm.decode(torch.from_numpy(qq).to(dev))
        keep = topn_mask(ex.float(), cl.float())
    return pp.float().cpu().numpy(), keep.cpu().numpy()


def nat_place(proto, keep, wi, py, px, sc, H, Wd):
    """語原型を (py,px,sc) ビン中心に配置 (Track F render_cloze.nat_place と同式)。
    返る座標は (x,y)=(col,row) 順。"""
    scale = 2.0 ** (S_LO + (sc + 0.5) / S_BINS * (S_HI - S_LO))
    cy = (py + 0.5) / 16 * H
    cx = (px + 0.5) / 16 * Wd
    return proto[wi] / scale + np.array([cy, cx]), keep[wi]
