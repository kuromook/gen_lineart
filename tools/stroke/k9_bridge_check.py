#!/usr/bin/env python
"""K9 器具確認(基盤 reply_k8 §5/§6/§7腕2)。**存在の主張の前に**回す。

§6 交絡: `ink_mask(thr=128, grow=200)` は 128 未満の種に 8 近傍で繋がる 200 未満を墨にする。
だから**灰 190 の画素が 1 つ挟まっているだけの 2 本の線が「地続き」になる**。
その場合の機構は剪定ではなく**前処理器のインク規則**。

印字(棒は付けない。登録定数は動かさない。感度の印字):
  - 14 件を **thr=128・grow 無し**で再判定して何件生き残るか
  - 繋いでいる墨の**最大灰**(最小最大経路のボトルネック)と、そこの**局所インク幅**
    線 1 本の幅(約 2.7px)より細い橋は線ではない
§7 腕2(塊の腕): コマごとに最大インク成分が全インクに占める割合。記述、帰無値なし。
"""
import argparse, csv, heapq, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import k2_box_b_gray as K

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def minimax_path(g, mask, pa, pb):
    """mask の中を 8 近傍で歩いて pa->pb、経路上の**最大灰を最小化**する。
    戻り値 (ボトルネックの灰, その画素, 経路長)。届かなければ (None, None, None)。"""
    H, W = g.shape
    a = (int(round(pa[0])), int(round(pa[1])))
    b = (int(round(pb[0])), int(round(pb[1])))
    if not mask[a] or not mask[b]:
        return None, None, None
    best = np.full((H, W), 1 << 30, np.int32)
    prev = {}
    best[a] = int(g[a])
    h = [(int(g[a]), a)]
    while h:
        d, u = heapq.heappop(h)
        if d > best[u]:
            continue
        if u == b:
            break
        for dr, dc in NB:
            v = (u[0] + dr, u[1] + dc)
            if not (0 <= v[0] < H and 0 <= v[1] < W) or not mask[v]:
                continue
            nd = max(d, int(g[v]))
            if nd < best[v]:
                best[v] = nd
                prev[v] = u
                heapq.heappush(h, (nd, v))
    if best[b] >= (1 << 30):
        return None, None, None
    # ボトルネック画素(経路上で灰が最大の点)を取り出す
    path, cur = [b], b
    while cur != a:
        cur = prev[cur]
        path.append(cur)
    k = max(path, key=lambda q: int(g[q]))
    return int(best[b]), k, len(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--contig", default="results/k8_contiguity_20261018/per_pair_contig.csv")
    ap.add_argument("--out", default="results/k9_bridge_check_20261018")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(ROOT / a.contig))
            if r["box"] == "B" and r["missing"] == "0" and r["comp_same"] == "1"]
    print(f"対象: 箱 B の comp_same=1 {len(rows)} 件", flush=True)
    sys.path.insert(0, str(K.TRACK_H / "tools/face"))
    import group_server as G
    G.load_state(Path(a.labels))
    P = G.S["panels"]
    byp = defaultdict(list)
    for r in rows:
        byp[int(r["panel"])].append(r)

    res, blob = [], {}
    for pid, rs in byp.items():
        g = cv2.imread(str(K.DATASET / P[pid]["source"] / "line" / P[pid]["name"]), 0)
        Q = K.pipeline(g)
        ink_h = Q["ink"].astype(bool)                      # 登録どおり thr=128, grow=200
        ink_p = (g < K.THR)                                # thr のみ、grow 無し
        _n, lab_p = cv2.connectedComponents(ink_p.astype(np.uint8), connectivity=8)
        dist_p = cv2.distanceTransform(ink_p.astype(np.uint8), cv2.DIST_L2, 5) * 2.0
        # 腕2: 最大インク成分が全インクに占める割合(ヒステリシスのマスクで)
        nh, lab_h = cv2.connectedComponents(ink_h.astype(np.uint8), connectivity=8)
        cnt = np.bincount(lab_h.ravel())
        cnt[0] = 0
        blob[pid] = dict(n_ink=int(ink_h.sum()), n_comp=int(nh - 1),
                         largest_share=round(float(cnt.max() / max(1, ink_h.sum())), 4))
        for r in rs:
            i, j, ei, ej = int(r["s0"]), int(r["s1"]), int(r["e0"]), int(r["e1"])
            pa, pb = Q["tips"][i][ei], Q["tips"][j][ej]
            gb, kp, plen = minimax_path(g, ink_h, pa, pb)
            # thr のみのマスクでの同一成分判定(tip が外れたら 2px 以内の最近傍インクへ)
            def plain_lab(p):
                rr, cc = int(round(p[0])), int(round(p[1]))
                if lab_p[rr, cc]:
                    return int(lab_p[rr, cc]), 0.0
                sub = lab_p[max(0, rr - 2):rr + 3, max(0, cc - 2):cc + 3]
                nz = sub[sub > 0]
                return (int(np.bincount(nz).argmax()), 1.0) if len(nz) else (0, -1.0)
            la, da = plain_lab(pa); lb, db = plain_lab(pb)
            same_p = int(la > 0 and la == lb)
            res.append(dict(panel=pid, order=r["order"], label=r["label"], gap=r["gap"],
                            gap_bin=r["gap_bin"], s0=i, s1=j,
                            bridge_max_gray=gb if gb is not None else -1,
                            bridge_iw=round(float(dist_p[kp]), 2) if kp else -1.0,
                            bridge_iw_hyst=round(float(Q["iw"][kp]), 2) if kp else -1.0,
                            bridge_px=kp[0] if kp else -1, bridge_py=kp[1] if kp else -1,
                            path_len=plen if plen else -1,
                            same_plain=same_p, tip_off_plain=int(da > 0 or db > 0),
                            tip_missing_plain=int(da < 0 or db < 0),
                            largest_share=blob[pid]["largest_share"]))
    with open(out / "per_case.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)

    surv = [r for r in res if r["same_plain"]]
    gr = [r["bridge_max_gray"] for r in res if r["bridge_max_gray"] >= 0]
    iw = [r["bridge_iw_hyst"] for r in res if r["bridge_iw_hyst"] >= 0]
    summary = dict(
        registered=dict(note="棒は付けない。登録定数は動かさない。感度の印字のみ",
                        thr=K.THR, grow=K.GROW),
        n_cases=len(res),
        plain_thr_only=dict(survived=len(surv), lost=len(res) - len(surv),
                            tip_off_plain=sum(r["tip_off_plain"] for r in res),
                            tip_missing_plain=sum(r["tip_missing_plain"] for r in res),
                            note="thr=128・grow 無しでも 2 つの tip が同じ成分に入る件数"),
        bridge=dict(max_gray=sorted(gr),
                    max_gray_median=float(np.median(gr)) if gr else None,
                    n_ge_128=sum(1 for x in gr if x >= 128),
                    iw_at_bottleneck=sorted(iw),
                    iw_median=float(np.median(iw)) if iw else None,
                    n_iw_lt_2p7=sum(1 for x in iw if 0 <= x < 2.7),
                    note="最大灰 >= 128 は、ヒステリシスが無ければ繋がらない橋。"
                         "iw < 2.7px は線 1 本の幅より細い橋"),
        arm2_blob=dict(per_panel=blob,
                       largest_share_median=round(float(np.median(
                           [v["largest_share"] for v in blob.values()])), 4),
                       note="コマ全体が 1 つの塊なら comp_same=1 は塊の性質で機構ではない"))
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
