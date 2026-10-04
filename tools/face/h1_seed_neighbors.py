#!/usr/bin/env python
"""実験 H1(2026-10-04 事前登録): 種に形が近いまとまりを全体から集めて並べる。

近さ = 正規化枠(±28)の点群どうしの対称 chamfer 距離(Track G 層1 X4 と同じ定義)。大きさ・語は条件に使わない。
種と同じ絵(chamfer < 0.5 かつ線の本数が同じ)は順位から除いて記録する。
種ごとに、同じ作品から近い順 20 個・他の作品から近い順 20 個。1 組 = 近景・遠景・コマ全体の 3 枠。

道具確認: (1) 種自身との距離 0 (2) r89070 の重複に r2270 が出る (3) 元の線画に乗る割合の中央値 >= 0.8
"""
import argparse, csv, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_eye_sizes as C                                           # noqa: E402

SEEDS = {89070: "closed_eye", 163364: "ear"}      # 作者(ユーザー)の判定
KNOWN_DUP = {89070: 2270}
CELL = 200
PER_LINE = 4


def chamfer_to_all(P, M, seed, dev, bs=2048):
    """seed と全まとまりの対称 chamfer(正規化枠の単位)"""
    a = torch.from_numpy(np.asarray(P[seed], np.float32)[np.asarray(M[seed])]).reshape(-1, 2).to(dev)
    out = np.zeros(len(M), np.float32)
    for i in range(0, len(M), bs):
        b = torch.from_numpy(np.asarray(P[i:i + bs], np.float32)).to(dev)          # (B, 32, 16, 2)
        m = torch.from_numpy(np.asarray(M[i:i + bs])).to(dev)                      # (B, 32)
        B = b.shape[0]
        pm = m[:, :, None].expand(-1, -1, b.shape[2]).reshape(B, -1)               # (B, 512) 点の有効
        d = torch.cdist(a[None].expand(B, -1, -1), b.reshape(B, -1, 2))            # (B, na, 512)
        d = d.masked_fill(~pm[:, None, :], 1e6)
        a2b = d.min(2).values.mean(1)
        b2a = d.min(1).values
        b2a = (b2a * pm).sum(1) / pm.sum(1).clamp(min=1)
        r = 0.5 * (a2b + b2a)
        r[pm.sum(1) == 0] = 1e6
        out[i:i + B] = r.cpu().numpy()
    return out


def one_set(pan, P, M, meta, rs, i, text, no):
    """1 組 = 近景 | 遠景 | コマ全体。枠で囲み番号を振る"""
    C.CELL = CELL
    pid = int(meta[i]["panel"])
    H, Wd = (float(v) for v in pan.dims[pid])
    inst = C.inst_points(P, M, meta, i)
    allp = np.concatenate(inst)
    c = (allp.min(0) + allp.max(0)) / 2
    d = rs[i] * min(H, Wd)
    lines = pan.lines(pid)
    near = C.crop_cell(pan, lines, inst, c, max(d * 2, 8.0), pid)
    far = C.crop_cell(pan, lines, inst, c, min(max(d * 6, 0.3 * min(H, Wd)), max(H, Wd)), pid)
    side = max(H, Wd) * 1.04
    whole = C.crop_cell(pan, lines, inst, (Wd / 2, H / 2), side, pid)
    q = (c - (np.array([Wd / 2, H / 2]) - side / 2)) * CELL / side
    cv2.circle(whole, tuple(np.round(q).astype(int)), max(8, int(d * CELL / side)), (0, 0, 220), 1, cv2.LINE_AA)
    for im, t in ((near, "zoom"), (far, "wide"), (whole, "panel")):
        cv2.putText(im, t, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 110, 40), 1, cv2.LINE_AA)
    body = np.hstack([near, np.full((CELL, 1, 3), 210, np.uint8), far, np.full((CELL, 1, 3), 210, np.uint8), whole])
    head = np.full((22, body.shape[1], 3), 235, np.uint8)
    cv2.putText(head, f"[{no}] {text}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    box = np.vstack([head, body])
    box = cv2.copyMakeBorder(box, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(90, 90, 90))
    return cv2.copyMakeBorder(box, 6, 10, 6, 14, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def grid(sets):
    blank = np.full_like(sets[0], 255)
    sets = sets + [blank] * (-len(sets) % PER_LINE)
    return np.vstack([np.hstack(sets[j:j + PER_LINE]) for j in range(0, len(sets), PER_LINE)])


def banner(w, text, h=34, color=(0, 0, 160)):
    bar = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(bar, text, (6, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/h1_seed_neighbors_20261004")
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    out = C.ROOT.parent / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    words, meta, P, M = C.encode_all(dev)
    pan = C.Panels()
    rs = np.array([C.rel_size(meta, pan.dims, i) for i in range(len(meta))])
    work = np.array([m["work"] for m in meta])
    nstk = np.array([int(m["n"]) for m in meta])

    plan, chk, rows = {}, {}, []
    for seed, name in SEEDS.items():
        d = chamfer_to_all(P, M, seed, dev)
        dup = np.flatnonzero((d < 0.5) & (nstk == nstk[seed]))
        dup = dup[dup != seed]
        cand = np.ones(len(d), bool); cand[seed] = False; cand[dup] = False
        order = np.argsort(d, kind="stable")
        order = order[cand[order]]
        same = order[work[order] == work[seed]][:a.k]
        other = order[work[order] != work[seed]][:a.k]
        plan[seed] = (d, same, other, dup)
        shown = np.concatenate([[seed], same, other])
        fr = [C.on_line_frac(pan, P, M, meta, int(i)) for i in shown]
        chk[seed] = dict(name=name, self_dist=float(d[seed]), self_ok=bool(d[seed] < 1e-4),
                         dups=[int(x) for x in dup], n_dup=int(len(dup)),
                         known_dup_ok=bool(KNOWN_DUP[seed] in dup) if seed in KNOWN_DUP else None,
                         on_line_median=round(float(np.median(fr)), 4), on_line_ok=bool(np.median(fr) >= 0.8),
                         d_same_range=[round(float(d[same[0]]), 3), round(float(d[same[-1]]), 3)],
                         d_other_range=[round(float(d[other[0]]), 3), round(float(d[other[-1]]), 3)],
                         n_words_top=int(len(set(words[np.concatenate([same, other])].tolist()))),
                         n_seed_word_top=int((words[np.concatenate([same, other])] == words[seed]).sum()))
        print(seed, chk[seed], flush=True)
    ok = all(c["self_ok"] and c["on_line_ok"] and c["known_dup_ok"] is not False for c in chk.values())
    json.dump(dict(check=chk, passed=ok), open(out / "check.json", "w"), indent=1)
    print("道具確認:", "合格" if ok else "不合格", flush=True)
    if not ok:
        return

    for seed, name in SEEDS.items():
        d, same, other, dup = plan[seed]
        lab = lambda i, rk: f"{rk} r{i} w{words[i]} d={d[i]:.2f} s={rs[i]:.3f} {work[i][:22]}"
        s0 = one_set(pan, P, M, meta, rs, seed, lab(seed, "SEED"), "seed")
        parts = [banner(s0.shape[1] * PER_LINE, f"seed r{seed} ({name})  w{words[seed]}  {work[seed]}   "
                        f"same-picture duplicates removed: {len(dup)}", color=(0, 0, 0)), grid([s0])]
        no = 0
        for title, idx, grp in ((f"SAME WORK: nearest {a.k} by shape", same, "same_work"),
                                (f"OTHER WORKS: nearest {a.k} by shape", other, "other_work")):
            sets = []
            for rk, i in enumerate(idx, 1):
                no += 1; i = int(i)
                sets.append(one_set(pan, P, M, meta, rs, i, lab(i, f"#{rk}"), no))
                rows.append(dict(seed=seed, seed_name=name, no=no, group=grp, rank=rk, row=i, word=int(words[i]),
                                 chamfer=round(float(d[i]), 4), rel_size=round(float(rs[i]), 4),
                                 panel=int(meta[i]["panel"]), work=work[i]))
            parts += [banner(sets[0].shape[1] * PER_LINE, title), grid(sets)]
        path = out / f"seed_r{seed}_{name}.png"
        cv2.imwrite(str(path), np.vstack(parts))
        print("->", path, flush=True)
    with open(out / "neighbors.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main()
