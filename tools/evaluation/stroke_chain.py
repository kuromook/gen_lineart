"""Chain skeleton fragments through junctions by curve continuity.

Registered 2026-10-10 in doc/work_log.md ("Item 4 step 2 pre-registration").
H-curve passed on the junction-to-junction FRAGMENT, which is not the unit the
user's hypothesis is about (median 16-31px, nearly straight -- lesson 11). To
test "a drawn stroke has a recoverable equation" a drawn stroke has to exist
first, and the same operation is what item 2 needs, since continuing a stroke
across a junction and continuing it across a gap are one mechanism.

  node = connected component of the junction-pixel set (an X gives a cluster,
         not a pixel)
  arm  = (fragment, which end), for each fragment end 8-adjacent to a node

At an arm, geometry is read from a local window of the last TAN_WIN px, read
inside -> outward, in a frame built explicitly (u = outward tangent from the
window's PCA, v = rot90(u)) rather than inherited from SVD sign luck:

  outward tangent  T = (u + m v)/sqrt(1+m^2),  m = 2 a s_e + b
  signed curvature K = 2a / (1+m^2)^{3/2}

Joining i to j traverses i outward and enters j AGAINST its outward direction.
Signed curvature of a plane curve flips under reversal, so continuity is
K_i ~ -K_j, not K_i ~ K_j.

  cost = theta/THETA0 + W_KAPPA * dK + d/D0
  theta = angle(T_i, -T_j);  dK = |K_i+K_j| / (|K_i|+|K_j|+1/KAPPA_REF)
  d     = mean perpendicular distance from each arm's extrapolated parabola to
          the other arm's tip -- the term that separates a continuation from
          two strokes passing the junction side by side

The decision at a node is a maximum-weight matching with unmatched arms
allowed (that is the whole point at a T: the bar continues, the stem stops),
enumerated exhaustively up to MAX_ARMS. Every arm is used at most once, so the
result is a set of disjoint simple paths and cycles; no stroke ever branches.

Arms
  real   the chainer
  nc_j1  (T, K) replaced by a different node's, same arm count, seed fixed --
         tip positions kept, so this isolates the continuity evidence from
         mere proximity
  nc_j2  a uniformly random matching
Both controls are constrained to the SAME number of pairs per node as `real`,
so the registered NC-j3 length-fairness condition holds by construction
rather than by luck (amendment recorded in the work log with its reason).

Usage:
  stroke_chain.py --root <panel dir> --which line --label ako5 --limit 40
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_mask, ink_of  # noqa: E402
from stroke_curves import (LONG_BRANCH_PX, RESID_PX, SPLIT_GAIN, SPLIT_MARGIN,  # noqa: E402
                           SPLIT_MIN_SIDE, SPLIT_STEPS, branch_in_fill,
                           order_path, poly_rms)

R = Path("results/stroke_chain_20261010")
SEED = 20261010

TAN_WIN = 12        # px of each fragment end used to read its tangent/curvature
TAN_SKIP = 0        # px of collar skipped at the tip before fitting; see the
                    # IC-j6 calibration. Skeletonizing pulls ink toward a
                    # junction's centre, so the last few px of a fragment bend
                    # into the junction and a tangent read there is biased.
THETA0 = 30.0       # deg; the cost unit for a kink
D0 = 1.5            # px; the cost unit for a sideways offset
W_KAPPA = 1.0
KAPPA_REF = 200.0   # px; curvature floor, so two straight ends are not punished
THETA_MAX = 60.0    # hard gate
D_MAX = 3.0         # hard gate, px
C_UNMATCHED = 1.0   # leaving an arm alone; a pair must cost under 2 x this
MAX_ARMS = 8        # a bigger node leaves every arm unmatched, and is counted
MERGE_PX = 6.0      # junction clusters joined by a fragment this short are one
                    # node. Added after IC-j1 failed: see the amendment in
                    # doc/work_log.md. Skeletonizing an X gives TWO degree-3
                    # clusters bridged by a stub whose length is on the order
                    # of the stroke width (1.0px in IC-j1), and that stub
                    # belongs to both strokes, so no matching that uses each
                    # arm once can recover them until the stub is absorbed.
MULTI_BP_PER_PX = 60.0   # exploratory only: breakpoint budget = len/this
MULTI_BP_MAX = 3


# --------------------------------------------------------------------------
# fragments and nodes

def fragments(ink):
    """Returns (frags, nodes, skel, stats).

    frags: list of (N,2) float arrays, x,y, ordered. Short pieces are KEPT --
    dropping a 4px fragment would sever a stroke that runs through it.
    nodes: list, per fragment, of [node_at_end0, node_at_end1], -1 for none.
    """
    skel = skeletonize(ink)
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    nb = cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    jmask = skel & (nb >= 3)
    seg = skel & ~jmask

    _, lab = cv2.connectedComponents(seg.astype(np.uint8), connectivity=8)
    _, jlab = cv2.connectedComponents(jmask.astype(np.uint8), connectivity=8)

    by = defaultdict(list)
    ys, xs = np.nonzero(lab)
    for y, x, i in zip(ys, xs, lab[ys, xs]):
        by[int(i)].append((int(y), int(x)))

    h, w = skel.shape
    frags, nodes = [], []
    stats = {"blobs": 0, "ends_two_nodes": 0}
    for pix in by.values():
        p = order_path(pix, min_px=1)
        if p is None:
            stats["blobs"] += 1
            continue
        ends = []
        for tip in (p[0], p[-1]):
            x0, y0 = int(round(tip[0])), int(round(tip[1]))
            hit = defaultdict(int)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y0 + dy, x0 + dx
                    if 0 <= yy < h and 0 <= xx < w and jlab[yy, xx] > 0:
                        hit[int(jlab[yy, xx])] += 1
            if not hit:
                ends.append(-1)
            else:
                if len(hit) > 1:
                    stats["ends_two_nodes"] += 1
                ends.append(max(sorted(hit), key=lambda n: hit[n]))
        frags.append(p)
        nodes.append(ends)
    return frags, nodes, skel, stats


def path_len(p):
    return float(np.hypot(*(np.diff(p, axis=0).T)).sum()) if len(p) > 1 else 0.0


def merge_nodes(frags, nodes):
    """Union junction clusters bridged by a fragment of at most MERGE_PX. The
    bridge itself is absorbed into the node (its ink joins no stroke), so the
    absorbed length is returned and reported rather than vanishing."""
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for n0, n1 in nodes:
        for n in (n0, n1):
            if n >= 0:
                parent.setdefault(n, n)
    absorbed, apx = set(), 0.0
    for fi, (n0, n1) in enumerate(nodes):
        if n0 >= 0 and n1 >= 0 and n0 != n1 and path_len(frags[fi]) <= MERGE_PX:
            ra, rb = find(n0), find(n1)
            if ra != rb:
                parent[ra] = rb
            absorbed.add(fi)
            apx += path_len(frags[fi])
    return {n: find(n) for n in parent}, absorbed, apx


# --------------------------------------------------------------------------
# arm geometry

def arm_geometry(p, end):
    """(tip, T, K, frame) for one end of a fragment, read inside -> outward.

    The tip itself is always the fragment's true end; the FIT may start a
    TAN_SKIP collar behind it and extrapolate forward, which is how the
    junction's deformation is kept out of the tangent."""
    n = len(p)
    q = p[::-1] if end == 0 else p
    tip = q[-1]
    skip = min(TAN_SKIP, max(0, n - 4))
    q = q[:n - skip] if skip else q
    L = min(TAN_WIN, len(q))
    w = q[-L:]
    if n < 3 or L < 3:
        d = w[-1] - w[0]
        nrm = float(np.hypot(*d))
        if nrm <= 1e-9:
            return tip, None, 0.0, None   # no direction exists; never matched
        return tip, d / nrm, 0.0, None
    c = w.mean(0)
    q = w - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    u = vt[0]
    if float((w[-1] - w[0]) @ u) < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    s = q @ u
    t = q @ v
    rms, co = poly_rms(s, t, 2)
    if co is None:
        d = w[-1] - w[0]
        nrm = float(np.hypot(*d))
        return tip, (d / nrm if nrm > 1e-9 else u), 0.0, None
    a, b = float(co[0]), float(co[1])
    s_e = float((tip - c) @ u)   # evaluated AT THE TIP, past the collar
    m = 2 * a * s_e + b
    T = (u + m * v) / np.sqrt(1 + m * m)
    K = 2 * a / (1 + m * m) ** 1.5
    return tip, T, float(K), (c, u, v, co, s_e)


def perp_offset(frame, tip_other):
    """Perpendicular distance from this arm's extrapolated parabola to the
    other arm's tip, in this arm's own frame."""
    if frame is None:
        return 0.0
    c, u, v, co, _ = frame
    q = tip_other - c
    return abs(float(q @ v) - float(np.polyval(co, float(q @ u))))


def pair_cost(ai, aj):
    """None if a gate rejects the pair."""
    (_, Ti, Ki, Fi), (tj, Tj, Kj, Fj) = ai, aj
    ti = ai[0]
    # A directionless arm (a 1px fragment) must not fall back to a fixed axis:
    # IC-j3 on a real panel caught exactly that, with a theta delta equal to
    # the rotation angle. Such an arm is simply never matched.
    if Ti is None or Tj is None:
        return None, float("inf"), float("nan"), float("nan"), float("nan")
    dot = float(np.clip(Ti @ (-Tj), -1.0, 1.0))
    theta = float(np.degrees(np.arccos(dot)))
    dK = abs(Ki + Kj) / (abs(Ki) + abs(Kj) + 1.0 / KAPPA_REF)
    d = 0.5 * (perp_offset(Fi, tj) + perp_offset(Fj, ti))
    gated = theta > THETA_MAX or d > D_MAX
    cost = theta / THETA0 + W_KAPPA * dK + d / D0
    return (None if gated else cost), cost, theta, dK, d


# --------------------------------------------------------------------------
# matching

def best_matching(k, cost, force_pairs=None):
    """Minimum-cost matching over k arms. cost[(i,j)] with i<j, missing = gated.
    force_pairs=None lets arms stay unmatched at C_UNMATCHED each; an integer
    requires exactly that many pairs (used by the controls, for length
    fairness)."""
    @lru_cache(maxsize=None)
    def rec(rem, need):
        if not rem:
            return (0.0, ()) if (need is None or need == 0) else (float("inf"), ())
        i, rest = rem[0], rem[1:]
        best = (float("inf"), ())
        if need is None or need <= len(rest) // 2:
            c0, m0 = rec(rest, need)
            if np.isfinite(c0):
                best = (c0 + C_UNMATCHED, m0)
        for idx, j in enumerate(rest):
            c = cost.get((i, j))
            if c is None:
                continue
            c1, m1 = rec(rest[:idx] + rest[idx + 1:],
                         None if need is None else need - 1)
            if np.isfinite(c1) and c + c1 < best[0]:
                best = (c + c1, m1 + ((i, j),))
        return best
    out = rec(tuple(range(k)), force_pairs)
    rec.cache_clear()
    return list(out[1])


def match_node(arms, mode, rng, donor=None):
    """arms: list of arm geometry tuples. Returns list of index pairs."""
    k = len(arms)
    if k < 2 or k > MAX_ARMS:
        return [], k > MAX_ARMS
    geo = donor if (mode == "nc_j1" and donor is not None) else arms
    cost = {}
    for i in range(k):
        for j in range(i + 1, k):
            gi = (arms[i][0], geo[i][1], geo[i][2], arms[i][3])
            gj = (arms[j][0], geo[j][1], geo[j][2], arms[j][3])
            gated, raw, *_ = pair_cost(gi, gj)
            cost[(i, j)] = ((raw if np.isfinite(raw) else None)
                            if mode != "real" else gated)
    if mode == "real":
        return best_matching(k, cost), False
    return None, False  # controls are filled in by the caller, needing n_pairs


# --------------------------------------------------------------------------
# stroke assembly

def assemble(frags, pairs_by_frag, skip=frozenset()):
    """pairs_by_frag: {(frag, end): (frag, end)}. Returns list of
    (points, is_cycle, n_frags)."""
    link = pairs_by_frag
    used = set(skip)
    out = []

    def walk(f, e_out):
        seq, cur_f, cur_e, cyc = [], f, e_out, False
        while True:
            nxt = link.get((cur_f, cur_e))
            if nxt is None:
                break
            g, ge = nxt
            if g in used:
                cyc = g == f
                break
            used.add(g)
            seq.append((g, ge == 1))
            cur_f, cur_e = g, 1 - ge
        return seq, cyc

    for f in range(len(frags)):
        if f in used:
            continue
        used.add(f)
        fwd, c1 = walk(f, 1)
        bwd, c2 = walk(f, 0)
        chain = [(g, not fl) for g, fl in reversed(bwd)] + [(f, False)] + fwd
        pts = np.vstack([frags[g][::-1] if fl else frags[g] for g, fl in chain])
        out.append((pts, c1 or c2, len(chain)))
    return out


# --------------------------------------------------------------------------
# fitting

def fit_stroke(p):
    """The registered model, unchanged from stroke_curves.fit_branch, so the
    stroke number is directly comparable to the fragment number."""
    c = p.mean(0)
    q = p - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    s, t = q @ vt[0], q @ vt[1]
    line_rms = float(np.sqrt(np.mean(t ** 2)))
    par_rms, par_co = poly_rms(s, t, 2)
    best_split, best_at = float("inf"), -1
    lo, hi = int(SPLIT_MARGIN * len(p)), int((1 - SPLIT_MARGIN) * len(p))
    step = max(1, (hi - lo) // SPLIT_STEPS)
    for b in range(max(lo, SPLIT_MIN_SIDE), max(hi, SPLIT_MIN_SIDE + 1), step):
        if b < SPLIT_MIN_SIDE or len(p) - b < SPLIT_MIN_SIDE:
            continue
        r1, _ = poly_rms(s[:b], t[:b], 2)
        r2, _ = poly_rms(s[b:], t[b:], 2)
        if not (np.isfinite(r1) and np.isfinite(r2)):
            continue
        r = float(np.sqrt((b * r1 ** 2 + (len(p) - b) * r2 ** 2) / len(p)))
        if r < best_split:
            best_split, best_at = r, b
    if line_rms <= RESID_PX:
        cls = "line"
    elif par_rms <= RESID_PX:
        cls = "arc"
    elif best_split <= SPLIT_GAIN * par_rms and best_split <= RESID_PX:
        cls = "split"
    else:
        cls = "unexplained"
    return {"n": len(p), "cls": cls, "line_rms": round(line_rms, 3),
            "par_rms": round(par_rms, 3), "split_rms": round(best_split, 3),
            "split_at": best_at,
            "curvature": round(abs(float(par_co[0])), 6) if par_co is not None else "",
            "a_coef": round(float(par_co[0]), 6) if par_co is not None else ""}


def seg_rms(p):
    """One parabola in the segment's OWN PCA frame."""
    if len(p) < 4:
        return 0.0
    c = p.mean(0)
    q = p - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    r, _ = poly_rms(q @ vt[0], q @ vt[1], 2)
    return r if np.isfinite(r) else float("inf")


def fit_multi(p, budget):
    """EXPLORATORY ONLY (registered as such): recursive greedy splitting, each
    piece fitted in its own frame. Reported, never allowed to move the bar."""
    if seg_rms(p) <= RESID_PX or budget <= 0 or len(p) < 2 * SPLIT_MIN_SIDE:
        return seg_rms(p) <= RESID_PX
    lo = max(int(SPLIT_MARGIN * len(p)), SPLIT_MIN_SIDE)
    hi = min(int((1 - SPLIT_MARGIN) * len(p)), len(p) - SPLIT_MIN_SIDE)
    step = max(1, (hi - lo) // SPLIT_STEPS)
    best, at = float("inf"), -1
    for b in range(lo, max(hi, lo + 1), step):
        r = max(seg_rms(p[:b]), seg_rms(p[b:]))
        if r < best:
            best, at = r, b
    if at < 0:
        return False
    return fit_multi(p[:at], budget - 1) and fit_multi(p[at:], budget - 1)


# --------------------------------------------------------------------------

def build_links(ink, mode, rng, pre=None):
    """The chainer proper: returns (frags, link, absorbed, stats) with no
    fitting, so calibration runs do not pay for it. `pre` supplies an already
    computed fragments() result, which is what makes a constants sweep
    affordable -- skeletonizing is the whole cost and it does not depend on
    any of the swept constants."""
    frags, nodes, skel, stats = pre if pre is not None else fragments(ink)
    stats = dict(stats)
    nmap, absorbed, apx = merge_nodes(frags, nodes)
    stats["absorbed_frags"] = len(absorbed)
    stats["absorbed_px"] = apx
    arms_by_node = defaultdict(list)   # merged node -> [(frag, end)]
    for fi, (n0, n1) in enumerate(nodes):
        if fi in absorbed:
            continue
        if n0 >= 0:
            arms_by_node[nmap[n0]].append((fi, 0))
        if n1 >= 0:
            arms_by_node[nmap[n1]].append((fi, 1))

    geo = {}
    for node, arms in arms_by_node.items():
        for (fi, e) in arms:
            geo[(fi, e)] = arm_geometry(frags[fi], e)

    node_ids = sorted(arms_by_node)
    # NC-j1 donors: another node with the same arm count, seeded
    by_count = defaultdict(list)
    for nd in node_ids:
        by_count[len(arms_by_node[nd])].append(nd)
    donor_of = {}
    for cnt, nds in by_count.items():
        shuf = list(nds)
        rng.shuffle(shuf)
        if len(shuf) > 1:
            shuf = shuf[1:] + shuf[:1]      # a derangement for a cycle of >1
        donor_of.update(dict(zip(nds, shuf)))

    link, nstat = {}, {"over_max": 0, "pairs": 0, "arms": 0}
    for nd in node_ids:
        arms = arms_by_node[nd]
        g = [geo[a] for a in arms]
        nstat["arms"] += len(arms)
        real_pairs, over = match_node(g, "real", rng)
        if over:
            nstat["over_max"] += 1
            continue
        if mode == "real":
            pairs = real_pairs
        elif mode == "nc_j1":
            k = len(arms)
            if k < 2 or k > MAX_ARMS:
                pairs = []
            else:
                d_arms = arms_by_node[donor_of[nd]]
                dg = [geo[a] for a in d_arms]
                cost = {}
                for i in range(k):
                    for j in range(i + 1, k):
                        gi = (g[i][0], dg[i][1], dg[i][2], g[i][3])
                        gj = (g[j][0], dg[j][1], dg[j][2], g[j][3])
                        c = pair_cost(gi, gj)[1]
                        cost[(i, j)] = c if np.isfinite(c) else None
                pairs = best_matching(k, cost, force_pairs=len(real_pairs))
        else:  # nc_j2
            idx = list(range(len(arms)))
            rng.shuffle(idx)
            pairs = [(idx[2 * i], idx[2 * i + 1]) for i in range(len(real_pairs))]
        for (i, j) in pairs:
            a, b = arms[i], arms[j]
            link[a], link[b] = b, a
            nstat["pairs"] += 1

    return frags, link, absorbed, {**stats, **nstat, "frags": len(frags)}


def chain_image(ink, fill, mode, rng):
    frags, link, absorbed, stats = build_links(ink, mode, rng)
    strokes = assemble(frags, link, skip=absorbed)
    rows = []
    for pts, cyc, nf in strokes:
        step = np.hypot(*(np.diff(pts, axis=0).T)) if len(pts) > 1 else np.array([0.0])
        path_px = float(step.sum())
        r = fit_stroke(pts) if (len(pts) >= 4 and not cyc) else {
            "n": len(pts), "cls": "cycle" if cyc else "tooshort",
            "line_rms": "", "par_rms": "", "split_rms": "", "split_at": -1,
            "curvature": "", "a_coef": ""}
        r["path_px"] = round(path_px, 1)
        r["end_px"] = round(float(np.hypot(*(pts[-1] - pts[0]))), 1)
        r["straight"] = round(r["end_px"] / path_px, 3) if path_px > 0 else ""
        r["n_frags"] = nf
        r["cycle"] = int(cyc)
        r["in_fill"] = round(branch_in_fill(pts, fill), 3)
        budget = min(int(path_px / MULTI_BP_PER_PX), MULTI_BP_MAX)
        r["multi_ok"] = int(fit_multi(pts, budget)) if (len(pts) >= 4 and not cyc) else 0
        r["multi_bp"] = budget
        rows.append(r)
    return rows, {**stats, "strokes": len(strokes)}


EXPLAINED = ("line", "arc", "split")


def summarise(rows, label, stat):
    def block(sel):
        tot = sum(r["n"] for r in sel)
        if not tot:
            return None
        sh = {c: 100.0 * sum(r["n"] for r in sel if r["cls"] == c) / tot
              for c in EXPLAINED + ("unexplained", "cycle", "tooshort")}
        multi = 100.0 * sum(r["n"] for r in sel if r["multi_ok"]) / tot
        return len(sel), tot, sh, multi
    print(f"\n== {label} ==  frags {stat['frags']}  arms {stat['arms']} "
          f"pairs {stat['pairs']}  strokes {stat['strokes']}  "
          f"blobs {stat['blobs']}  nodes>{MAX_ARMS}arms {stat['over_max']}  "
          f"absorbed stubs {stat['absorbed_frags']} ({stat['absorbed_px']:.0f}px)  "
          f"ends touching 2 nodes {stat['ends_two_nodes']}")
    print(f"{'subset':>26}{'strokes':>9}{'px':>9}{'line':>8}{'arc':>7}{'split':>7}"
          f"{'EXPL':>8}{'unexpl':>8}{'cycle':>7}{'multi*':>8}")
    long_out = [r for r in rows if r["path_px"] >= LONG_BRANCH_PX and r["in_fill"] < 0.5]
    subsets = [("all", rows),
               (">=%dpx" % LONG_BRANCH_PX, [r for r in rows if r["path_px"] >= LONG_BRANCH_PX]),
               (">=30px outside fill", long_out),
               (">=30px inside fill",
                [r for r in rows if r["path_px"] >= LONG_BRANCH_PX and r["in_fill"] >= 0.5])]
    for name, sel in subsets:
        b = block(sel)
        if b is None:
            continue
        n, tot, sh, multi = b
        print(f"{name:>26}{n:9d}{tot:9d}{sh['line']:7.1f}%{sh['arc']:6.1f}%"
              f"{sh['split']:6.1f}%{sum(sh[c] for c in EXPLAINED):7.1f}%"
              f"{sh['unexplained']:7.1f}%{sh['cycle']:6.1f}%{multi:7.1f}%")
    lens = sorted(r["path_px"] for r in rows)
    st = sorted(r["straight"] for r in rows if r["straight"] != "")
    if lens:
        print(f"  median stroke path {lens[len(lens) // 2]:.0f}px   "
              f"median straightness {st[len(st) // 2] if st else float('nan'):.3f}   "
              f"median frags/stroke {sorted(r['n_frags'] for r in rows)[len(rows) // 2]}")
    return long_out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--which", default="line", choices=("rough", "line"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--modes", default="real,nc_j1,nc_j2")
    args = ap.parse_args()

    paths = sorted(p for p in (Path(args.root) / args.which).iterdir()
                   if p.suffix.lower() in (".png", ".jpg"))[:args.limit or None]
    R.mkdir(parents=True, exist_ok=True)
    for mode in args.modes.split(","):
        rng = random.Random(SEED)
        rows, agg = [], defaultdict(int)
        for p in paths:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            ink = ink_of(g)
            rs, st = chain_image(ink, fill_mask(ink), mode, rng)
            for r in rs:
                r["image"] = p.name
            rows += rs
            for k, v in st.items():
                agg[k] += v
        tag = f"{args.label}_{args.which}_{mode}"
        out = R / f"strokes_{tag}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "image", "n", "cls", "line_rms", "par_rms", "split_rms", "split_at",
                "curvature", "a_coef", "path_px", "end_px", "straight", "n_frags",
                "cycle", "in_fill", "multi_ok", "multi_bp"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        summarise(rows, f"{tag}  ({len(paths)} images)", agg)
        print(f"saved: {out}")
    print("\n* multi = EXPLORATORY length-scaled breakpoint budget, "
          "registered as not able to move the bar")


if __name__ == "__main__":
    main()
