#!/usr/bin/env python
"""Strokes from a cached 1px skeleton: split at junctions, then link through them.

Track F restart, step 2. The old token was a junction-to-junction span
(median 28px): a drawn line crossed by another line became several tokens.
Here spans are re-joined when they continue each other through a junction.

  1. split    crossing-number junctions (stroke_churn rule), 3x3 dilated, so
              each junction cluster is a node and each remaining 8-connected
              run is a span.
  2. prune    spans with a free end and length <= spur limit that hang off a
              node are skeletonization spurs, not strokes; removed, then the
              skeleton is re-split once.
  3. collapse spans shorter than the local ink width joining two nodes are the
              inside of one thick crossing; their two nodes become one.
  4. link     at every node, arms are paired greedily by how straight the
              continuation is (angle between outgoing directions closest to
              180 deg), up to --max-bend. Unpaired arms end there, so a T keeps
              its bar as one stroke and its stem as another.

Everything returns arrays; nothing is cached but the skeleton.
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
    for s, lst in nodes_of.items():
        ids = {n for n, _, _ in lst}
        if len(ids) == 2:
            r, c = spans[s - 1]
            if len(r) <= max(3.0, 1.5 * float(np.median(ink_width[r, c]))):
                a, b = list(ids); nuf.union(a, b); bridge[s] = True
    # --- 4. link arms at each (collapsed) node ---
    arms = {}
    for s, n, ar, ac in att:
        if bridge[s]:
            continue
        r, c = spans[s - 1]
        v = arm_direction(np.stack([r, c], 1).astype(np.float32), ar, ac, reach)
        arms.setdefault(nuf.find(n), []).append((s, v))
    suf = UF(ns + 1)
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
    # bridge spans are not linked: they stay their own (short, usually dropped)
    # token, so a thick crossing leaves a gap of about one ink width
    root = np.array([suf.find(s) for s in range(ns + 1)])
    uniq, stroke_of_span = np.unique(root[1:], return_inverse=True)
    lengths = np.bincount(stroke_of_span, weights=[len(r) for r, _ in spans], minlength=len(uniq))
    keep = lengths >= min_len
    info = {"spurs_removed": removed, "spans": ns, "nodes": nn, "links": n_links,
            "bridges": int(bridge.sum()), "strokes": int(keep.sum()),
            "spans_kept": int(sum(1 for r, _ in spans if len(r) >= min_len))}
    return spans, stroke_of_span, keep, info
