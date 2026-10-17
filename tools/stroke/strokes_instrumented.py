#!/usr/bin/env python
"""Track F `strokes.py` の複製に、橋 span の記録だけを足したもの(Track K、2026-10-16)。

**Track F のファイルは改変しない**ので複製した(Track J が `stroke_chain_touch.py` で
取った形と同じ)。規則は 1 つも変えていない: 返り値の最初の 4 つ
`(spans, stroke_of_span, keep, info)` は原本と完全に一致する。
5 つ目の `extra` だけが新しく、これは IC-k5a で原本と突き合わせて確かめる。

`extra` に入るもの(K4a が数えるために必要で、原本が外に出していないもの):
  bridges   畳み込まれた橋 span ごとの (span 番号, 画素数, 局所インク幅の中央値, 畳み込み閾値)
  spans_all 全 span の (画素数, 局所インク幅の中央値, 付いている節点の数)
  att       (span 番号, 節点番号, 付着点 row, 付着点 col) —— 原本の `attachments` の出力そのまま
  node_root 畳み込み後の節点代表 `nuf.find(n)`(K2 の箱 A と箱 C の区別に必要)
  arms      畳み込み後の節点 -> [(span 番号, 腕の単位ベクトル)] —— 原本が対を作るのに使う候補集合
  linked_pairs 原本が実際に対にした span 対(K2 の条件 (d)「規則が対にしなかった」に必要)
  linked_at    その対がどの(畳み込み後の)節点で作られたか(K4 の「橋を通るか」に必要)
  merge_bridges 畳み込み後の節点 -> その節点を畳んだ橋 span の記録(K4 の尾根長)
原本 `strokes_from_skeleton` の該当箇所に行を足しただけで、判定の式には触れていない。
"""

import cv2
import numpy as np

INK_THRESHOLD = 128
HOLE_MAX_PX = 30


def ink_mask(gray, hole_max=HOLE_MAX_PX, thr=INK_THRESHOLD, grow=None):
    """gray < thr, with enclosed background holes of <= hole_max px filled.

    A few-pixel hole inside a thick or overlapping stroke turns its skeleton
    into a loop or ladder of junctions (seen 2026-09-17 on hatching).

    `grow` turns the threshold into hysteresis: a pixel joins the ink if it is
    below `grow` AND its 8-connected component contains a pixel below `thr`.
    Measured 2026-09-18: a plain `< 128` cut breaks faint anti-aliased lines
    into dots -- one panel went from 258 components at 128 to 64 at 200 -- and a
    line broken into dots is not one token, which matters here more than it did
    for the discriminative work.
    """
    ink = gray < thr
    if grow is not None and grow > thr:
        wide = gray < grow
        n, lab = cv2.connectedComponents(wide.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[lab[ink]] = True
        keep[0] = False
        ink = keep[lab]
    if hole_max <= 0:
        return ink
    n, lab, st, _ = cv2.connectedComponentsWithStats((~ink).astype(np.uint8), connectivity=4)
    h, w = ink.shape
    x, y, bw, bh, area = st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4]
    border = (x == 0) | (y == 0) | (x + bw == w) | (y + bh == h)
    small = (area <= hole_max) & ~border
    small[0] = False
    return ink | small[lab]


OFFS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def junction_mask(skel):
    s = np.pad(skel.astype(np.uint8), 1)
    h, w = skel.shape
    ring = [s[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc] for dr, dc in OFFS]
    trans = sum(((ring[k] == 0) & (ring[(k + 1) % 8] == 1)).astype(np.uint8) for k in range(8))
    return skel & (trans >= 3)


def neighbour_count(skel):
    k = np.ones((3, 3), np.float32); k[1, 1] = 0
    return cv2.filter2D(skel.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)


def group_pixels(labels):
    """label image -> (sorted flat indices, start offsets) per label 1..n."""
    flat = labels.ravel()
    idx = np.flatnonzero(flat)
    lab = flat[idx]
    o = np.argsort(lab, kind="stable")
    idx, lab = idx[o], lab[o]
    n = int(labels.max())
    starts = np.searchsorted(lab, np.arange(1, n + 2))
    return idx, starts


def split(skel):
    j = cv2.dilate(junction_mask(skel).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    nodes_mask = j & skel
    nn, node_lab = cv2.connectedComponents(nodes_mask.astype(np.uint8), connectivity=8)
    ns, span_lab = cv2.connectedComponents((skel & ~j).astype(np.uint8), connectivity=8)
    return node_lab, nn - 1, span_lab, ns - 1


def attachments(node_lab, span_lab, n_spans):
    """(span, node, attach_row, attach_col) for every span end touching a node."""
    h, w = node_lab.shape
    big = cv2.dilate(node_lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
    touch = (span_lab > 0) & (big > 0)
    rr, cc = np.nonzero(touch)
    sp = span_lab[rr, cc]; nd = big[rr, cc]
    base = int(node_lab.max()) + 1
    key = sp.astype(np.int64) * base + nd
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    mr = np.bincount(inv, weights=rr, minlength=len(uk)) / cnt
    mc = np.bincount(inv, weights=cc, minlength=len(uk)) / cnt
    return [(int(k // base), int(k % base), float(a), float(b)) for k, a, b in zip(uk, mr, mc)]


def arm_direction(pix_rc, ar, ac, reach):
    """Unit vector pointing from the attach point INTO the span, from pixels
    within `reach` px of it."""
    d = np.hypot(pix_rc[:, 0] - ar, pix_rc[:, 1] - ac)
    sel = pix_rc[d <= reach]
    if len(sel) < 2:
        sel = pix_rc[np.argsort(d)[: max(2, min(len(d), 4))]]
    v = sel.mean(0) - np.array([ar, ac])
    nv = np.hypot(*v)
    return v / nv if nv > 1e-6 else np.array([0.0, 0.0])


class UF:
    def __init__(self, n): self.p = np.arange(n)
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def strokes_from_skeleton(skel, ink_width, spur_max=6.0, max_bend=35.0, reach=10.0, min_len=8):
    """skel: bool HxW. ink_width: float HxW, local ink width (2 x distance).
    Returns (spans, stroke_of_span, keep, info): spans is a list of (rows, cols),
    stroke_of_span[i] the stroke index of span i, keep[k] whether stroke k
    reaches min_len. Junction node pixels belong to no stroke."""
    skel = skel.copy()
    # --- 2. prune spurs (once) ---
    node_lab, nn, span_lab, ns = split(skel)
    nb = neighbour_count(skel)
    idx, starts = group_pixels(span_lab)
    W = skel.shape[1]
    att = attachments(node_lab, span_lab, ns)
    touching = np.zeros(ns + 1, np.int32)
    for s, n, _, _ in att:
        touching[s] += 1
    endpoint = (nb == 1) & skel
    removed = 0
    for s in range(1, ns + 1):
        p = idx[starts[s - 1]:starts[s]]
        if touching[s] != 1 or len(p) == 0:
            continue
        r, c = p // W, p % W
        if not endpoint[r, c].any():
            continue
        lim = max(spur_max, float(np.median(ink_width[r, c])))
        if len(p) <= lim:
            skel[r, c] = False; removed += 1
    # --- 1. split again on the pruned skeleton ---
    node_lab, nn, span_lab, ns = split(skel)
    idx, starts = group_pixels(span_lab)
    spans = []
    for s in range(1, ns + 1):
        p = idx[starts[s - 1]:starts[s]]
        spans.append((p // W, p % W))
    att = attachments(node_lab, span_lab, ns)
    # --- 3. collapse short spans bridging two nodes inside one thick crossing ---
    nodes_of = {}
    for s, n, ar, ac in att:
        nodes_of.setdefault(s, []).append((n, ar, ac))
    nuf = UF(nn + 1)
    bridge = np.zeros(ns + 1, bool)
    bridges = []                                                           # K: 記録だけ
    for s, lst in nodes_of.items():
        ids = {n for n, _, _ in lst}
        if len(ids) == 2:
            r, c = spans[s - 1]
            thr = max(3.0, 1.5 * float(np.median(ink_width[r, c])))        # K: 閾値を控える
            if len(r) <= thr:
                a, b = list(ids); nuf.union(a, b); bridge[s] = True
                bridges.append((int(s), int(len(r)), float(np.median(ink_width[r, c])), float(thr)))
    # --- 4. link arms at each (collapsed) node ---
    arms = {}
    for s, n, ar, ac in att:
        if bridge[s]:
            continue
        r, c = spans[s - 1]
        v = arm_direction(np.stack([r, c], 1).astype(np.float32), ar, ac, reach)
        arms.setdefault(nuf.find(n), []).append((s, v))
    suf = UF(ns + 1)
    linked_pairs = set(); linked_at = {}                                   # K: 記録だけ
    cos_lim = -np.cos(np.deg2rad(max_bend))  # straight continuation: dot = -1
    n_links = 0
    for node, lst in arms.items():
        cand = []
        for i in range(len(lst)):
            for k in range(i + 1, len(lst)):
                if lst[i][0] == lst[k][0]:
                    continue
                dot = float(lst[i][1] @ lst[k][1])
                if dot <= cos_lim:
                    cand.append((dot, i, k))
        cand.sort()
        used = set()
        for dot, i, k in cand:
            if i in used or k in used:
                continue
            used.update((i, k)); suf.union(lst[i][0], lst[k][0]); n_links += 1
            linked_pairs.add((min(lst[i][0], lst[k][0]),
                              max(lst[i][0], lst[k][0])))                  # K: 記録だけ
            linked_at[(min(lst[i][0], lst[k][0]),
                       max(lst[i][0], lst[k][0]))] = int(node)             # K: 記録だけ
    # bridge spans are not linked: they stay their own (short, usually dropped)
    # token, so a thick crossing leaves a gap of about one ink width
    root = np.array([suf.find(s) for s in range(ns + 1)])
    uniq, stroke_of_span = np.unique(root[1:], return_inverse=True)
    lengths = np.bincount(stroke_of_span, weights=[len(r) for r, _ in spans], minlength=len(uniq))
    keep = lengths >= min_len
    info = {"spurs_removed": removed, "spans": ns, "nodes": nn, "links": n_links,
            "bridges": int(bridge.sum()), "strokes": int(keep.sum()),
            "spans_kept": int(sum(1 for r, _ in spans if len(r) >= min_len))}
    merge_bridges = {}                     # K: 畳み込み後の節点 -> そこを畳んだ橋(複数可)
    for b_ in bridges:
        merge_bridges.setdefault(int(nuf.find(nodes_of[b_[0]][0][0])), []).append(b_)
    spans_all = [(int(len(r)), float(np.median(ink_width[r, c])),
                  len({n for n, _, _ in nodes_of.get(i + 1, [])}))         # K: 相異なる節点の数
                 for i, (r, c) in enumerate(spans)]                        # K: 記録だけ
    extra = {"bridges": bridges, "spans_all": spans_all, "bridge_flag": bridge.copy(),
             # K2 が必要とするもの(いずれも記録だけ。規則は 1 行も変えていない)
             "att": [(int(s_), int(n_), float(ar_), float(ac_)) for s_, n_, ar_, ac_ in att],
             "node_root": {int(n_): int(nuf.find(n_)) for n_ in range(nn + 1)},
             "arms": {int(k_): [(int(s_), (float(v_[0]), float(v_[1])))
                                for s_, v_ in lst_] for k_, lst_ in arms.items()},
             "linked_pairs": sorted(linked_pairs),
             "linked_at": {f"{a}-{b}": n for (a, b), n in linked_at.items()},
             "merge_bridges": merge_bridges}
    return spans, stroke_of_span, keep, info, extra
