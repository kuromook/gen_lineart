"""IC-j8: split IC-j7's `topology` bucket, which is 81.1% of IC-j6's rejoin
failures.

Registered 2026-10-11 in doc/work_log.md (commit f62c0ed) before this file
existed, with amendments B1-B5 (commit 8dd3a60) made from code and arithmetic
before any real panel was read. Read that entry first; the short form:

    t_part_absorbed        a true half was absorbed into a node.
                           CONSTRUCTION ZERO, asserted: PART_MIN_PX = 8
                           ordered pixels and no resampling give a part
                           path_len >= 7.0 > MERGE_PX = 6.0
    t_no_node              a half touches no junction cluster at either end
                           -> free_end        (A NEW MECHANISM, not a repair)
    t_touch_split          under the touch closure the two halves share a
                           node: the fragments() mis-assignment (A6)
                           -> merge_touch                     (a defect repair)
    t_relay_other          a path between the halves exists that uses no
                           occluder fragment
                           -> relay_force      (an UPPER BOUND: gates off and
                              pairing forced at EVERY node, not just the
                              path's)
    t_relay_occluder_only  every path uses an occluder fragment, so rejoining
                           through it is scored `wrong` by IC-j6's own
                           definition -> no repair; all_levels is the
                           registered falsifier
    t_disconnected         no path at all -> all_levels is the falsifier
    t_residual             VACUOUS: the decision tree is exhaustive

`merge_px_inf` is a diagnostic only, never a matched repair, because
absorbing a bridge of any length is a constant change and not a defect
repair; `all_levels` excludes it on purpose.

Nothing here changes a chainer constant (C4). `stroke_chain.py`'s SHA-256 is
printed by every run.

Usage:
  diagnose_topology.py --check              # IC-j8 only
  diagnose_topology.py --root <panel dir>   # IC-j8, then the split
"""

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stroke_chain as sc  # noqa: E402
import calibrate_stroke_chain as cal  # noqa: E402
import diagnose_icj6 as dj  # noqa: E402

OUT = dj.OUT
SEED = cal.SEED
FREE_R = 8.0             # B5: free-tip pairing radius, a new mechanism's px

SUB = ("t_part_absorbed", "t_no_node", "t_touch_split", "t_relay_other",
       "t_relay_occluder_only", "t_disconnected", "t_residual")
LEVELS = ("merge_touch", "free_end", "relay_force", "relay_only",
          "merge_px_inf", "all_levels")
MATCHED = {"t_no_node": "free_end",
           "t_touch_split": "merge_touch",
           "t_relay_other": "relay_force",
           "t_part_absorbed": None,          # B2 construction zero
           "t_relay_occluder_only": None,    # B4 no repair can be clean
           "t_disconnected": None,
           "t_residual": None}
FORCE = {"C_UNMATCHED": 1e9, "THETA_MAX": 1e9, "D_MAX": 1e9}


# --------------------------------------------------------------------------
# the sub-classifier, pure so it can be asserted without images

def reach(adj, src):
    """Clusters reachable from any of `src` in the cluster adjacency."""
    seen, q = set(src), deque(src)
    while q:
        c = q.popleft()
        for n in adj.get(c, ()):
            if n not in seen:
                seen.add(n)
                q.append(n)
    return seen


def sub_bucket(absorbed_part, r0, r1, closure, adj_other, adj_all):
    """Which sub-bucket a `topology` failure falls in.

    r0/r1  merged cluster ids at the two halves' ends (sets, may be empty)
    closure  cluster -> touch-closure root (B1)
    adj_other / adj_all  cluster adjacency, without / with the fragments that
                         belong to the occluder
    The order is the registration's and is load-bearing.
    """
    if absorbed_part:
        return "t_part_absorbed"
    if not r0 or not r1:
        return "t_no_node"
    if {closure[c] for c in r0} & {closure[c] for c in r1}:
        return "t_touch_split"
    if reach(adj_other, r0) & r1:
        return "t_relay_other"
    if reach(adj_all, r0) & r1:
        return "t_relay_occluder_only"
    return "t_disconnected"


def _self_check():
    """One assert cell; the B2 arithmetic is a hand-computed known value."""
    # B2: the construction zero, as arithmetic rather than as an expectation
    assert cal.PART_MIN_PX == 8 and sc.MERGE_PX == 6.0
    assert (cal.PART_MIN_PX - 1) * 1.0 == 7.0 > sc.MERGE_PX, (
        "a part shorter than MERGE_PX would make t_part_absorbed reachable")
    ident = {1: 1, 2: 2, 3: 3, 4: 4}
    assert sub_bucket(True, {1}, {2}, ident, {}, {}) == "t_part_absorbed"
    assert sub_bucket(False, set(), {2}, ident, {}, {}) == "t_no_node"
    assert sub_bucket(False, {1}, set(), ident, {}, {}) == "t_no_node"
    # closure puts 1 and 2 on the same root -> the mis-assignment signature
    assert sub_bucket(False, {1}, {2}, {1: 1, 2: 1}, {}, {}) == "t_touch_split"
    # a relay through non-occluder ink: 1 - 3 - 2
    a = {1: {3}, 3: {1, 2}, 2: {3}}
    assert sub_bucket(False, {1}, {2}, ident, a, a) == "t_relay_other"
    # the same path, but its middle fragment is the occluder
    assert sub_bucket(False, {1}, {2}, ident, {}, a) == "t_relay_occluder_only"
    assert sub_bucket(False, {1}, {2}, ident, {}, {}) == "t_disconnected"
    # the tree is exhaustive: t_residual is unreachable, which is the point
    assert reach({1: {2}, 2: {1}}, {1}) == {1, 2}
    return True


_self_check()


# --------------------------------------------------------------------------
# the structures the classifier reads, built from one trial

def cluster_adj(nodes, nmap, skip=()):
    """Cluster adjacency: a fragment joins the merged clusters at its ends."""
    adj = defaultdict(set)
    for fi, (n0, n1) in enumerate(nodes):
        if fi in skip or n0 < 0 or n1 < 0:
            continue
        a, b = nmap[n0], nmap[n1]
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    return dict(adj)


def classify(ctx):
    """(sub-bucket, extras) for one `topology` failure."""
    frags, nodes, skel = ctx["frags"], ctx["nodes"], ctx["skel"]
    f0, f1 = ctx["fp"]
    nmap, absorbed, _ = sc.merge_nodes(frags, nodes)
    r0 = {nmap[n] for n in nodes[f0] if n >= 0}
    r1 = {nmap[n] for n in nodes[f1] if n >= 0}
    if not (f0 in absorbed or f1 in absorbed):
        assert not (r0 & r1), "a topology failure cannot already share a node"
    nmap_t, _, _, _ = dj.node_view_touch(frags, nodes, skel)
    closure = {c: nmap_t[c] for c in nmap_t}
    for c in r0 | r1:
        closure.setdefault(c, c)
    occ = set(ctx["op"])
    adj_all = cluster_adj(nodes, nmap)
    adj_other = cluster_adj(nodes, nmap, skip=occ)
    b = sub_bucket(f0 in absorbed or f1 in absorbed, r0, r1,
                   closure, adj_other, adj_all)
    return b, {"n_clusters": len(set(nmap.values())),
               "r0": len(r0), "r1": len(r1)}


# --------------------------------------------------------------------------
# the repairs. Every one rebuilds the WHOLE link map, so unlike IC-j7's
# node-local flips these are not node-local and the caveat does not apply.

def free_pairs(frags, nodes, geo_cache):
    """Synthetic nodes holding two free tips within FREE_R of each other."""
    free = [(fi, e) for fi, ns in enumerate(nodes)
            for e in (0, 1) if ns[e] < 0]
    for a in free:
        geo_cache.setdefault(a, sc.arm_geometry(frags[a[0]], a[1]))
    cand = []
    for i in range(len(free)):
        for j in range(i + 1, len(free)):
            if free[i][0] == free[j][0]:
                continue
            d = float(np.hypot(*(geo_cache[free[i]][0] - geo_cache[free[j]][0])))
            if d <= FREE_R:
                cand.append((d, free[i], free[j]))
    cand.sort(key=lambda t: (t[0], t[1], t[2]))
    used, out = set(), []
    for _, a, b in cand:
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        out.append([a, b])
    return out


def relay_path(nodes, nmap, r0, r1, occ):
    """The fragments on a shortest cluster path between the two halves that
    uses no occluder fragment, or None. B6."""
    edge = {}
    for fi, (n0, n1) in enumerate(nodes):
        if fi in occ or n0 < 0 or n1 < 0:
            continue
        a, b = nmap[n0], nmap[n1]
        if a != b:
            edge.setdefault((a, b), fi)
            edge.setdefault((b, a), fi)
    adj = defaultdict(set)
    for (a, b) in edge:
        adj[a].add(b)
    if r0 & r1:
        # the halves already share a node: there is nothing to relay, so the
        # intervention is "drop the rivals at that node and force a pair".
        # Without this, relay_only returned "not joined" for every trial that
        # already rejoins and NC-j8c read it as a 63% breakage. Corrected
        # 2026-10-12 and recorded in doc/work_log.md.
        return set(), set(r0 & r1)
    prev, q = {c: None for c in r0}, deque(sorted(r0))
    hit = None
    while q and hit is None:
        c = q.popleft()
        for n in sorted(adj.get(c, ())):
            if n in prev:
                continue
            prev[n] = c
            if n in r1:
                hit = n
                break
            q.append(n)
    if hit is None:
        return None, None
    chain, c = [hit], hit
    while prev[c] is not None:
        c = prev[c]
        chain.append(c)
    chain.reverse()
    frs = {edge[(chain[i], chain[i + 1])] for i in range(len(chain) - 1)}
    return frs, set(chain)


def relay_stats(ctx):
    """How far away the relay is. Added 2026-10-12, before the result was
    read as anything but a diagnosis of the instrument.

    A `t_relay_other` trial says only that SOME non-occluder path exists.
    Two very different things satisfy that: the crossing's own middle ridge,
    a few px between the two halves, which is the right join; or a detour
    through the rest of the drawing, which would satisfy IC-j6's success
    test (`roots` equal) without being the join we want. These three numbers
    separate them and are printed as a distribution, never pooled into one
    mean:

      n_frags  fragments on the shortest non-occluder path
      px       their total length
      reach    the farthest any path pixel strays from the midpoint between
               the two halves' facing tips
    """
    frags, nodes = ctx["frags"], ctx["nodes"]
    f0, f1 = ctx["fp"]
    nmap, _, _ = sc.merge_nodes(frags, nodes)
    r0 = {nmap[n] for n in nodes[f0] if n >= 0}
    r1 = {nmap[n] for n in nodes[f1] if n >= 0}
    frs, _ = relay_path(nodes, nmap, r0, r1, set(ctx["op"]))
    if not frs:
        return None
    # the facing tips: the closest pair of ends of the two halves
    best, m = None, None
    for i in (0, -1):
        for j in (0, -1):
            d = float(np.hypot(*(frags[f0][i] - frags[f1][j])))
            if best is None or d < best:
                best, m = d, (frags[f0][i] + frags[f1][j]) / 2.0
    px = sum(sc.path_len(frags[fi]) for fi in frs)
    reach = max(float(np.hypot(*(q - m))) for fi in frs for q in frags[fi])
    return {"n_frags": len(frs), "px": px, "reach": reach, "tip_gap": best}


def relink(ctx, level, rng):
    """(joined, clean) under one repair level, over the whole image."""
    frags, nodes, skel = ctx["frags"], ctx["nodes"], ctx["skel"]
    f0, f1 = ctx["fp"]
    touch = level in ("merge_touch", "all_levels")
    freee = level in ("free_end", "all_levels")
    force = level in ("relay_force", "relay_only", "all_levels")
    if level == "merge_px_inf":
        old = sc.MERGE_PX
        sc.MERGE_PX = 1e9
        try:
            _, _, abn, geo = dj.node_view(frags, nodes)
        finally:
            sc.MERGE_PX = old
    elif touch:
        _, _, abn, geo = dj.node_view_touch(frags, nodes, skel)
    else:
        _, _, abn, geo = dj.node_view(frags, nodes)
    abn = {k: list(v) for k, v in abn.items()}
    geo = dict(geo)
    if freee:
        for k, pair in enumerate(free_pairs(frags, nodes, geo)):
            abn[-(k + 1)] = pair
    ov = FORCE if force else {}
    drop_at = {}
    if level == "relay_only":
        nmap, _, _ = sc.merge_nodes(frags, nodes)
        r0 = {nmap[n] for n in nodes[f0] if n >= 0}
        r1 = {nmap[n] for n in nodes[f1] if n >= 0}
        frs, chain = relay_path(nodes, nmap, r0, r1, set(ctx["op"]))
        if frs is None:
            return False, True
        keep = frs | {f0, f1}
        for nd in chain:
            drop_at[nd] = {a[0] for a in abn.get(nd, ()) if a[0] not in keep}
    pairs = {nd: dj.node_pairs(arms, geo, rng, overrides=ov,
                               drop=drop_at.get(nd, ()))
             for nd, arms in abn.items()}
    link = dj.link_from(pairs, abn)
    r = cal.roots(len(frags), link)
    joined = r[f0] == r[f1]
    clean = not any(r[f] == r[o] for f in (f0, f1) for o in ctx["op"])
    return joined, clean


# --------------------------------------------------------------------------
# IC-j8: the constructed trials, labels written in the registration

def _case(fpieces, others, occl, occ_t=2):
    img = np.zeros((220, 220), np.uint8)
    fm = np.zeros((220, 220), np.uint8)
    om = np.zeros((220, 220), np.uint8)
    for p, q in fpieces:
        dj._line(img, p, q, 1)
        dj._line(fm, p, q, 1)
    for p, q in occl:
        dj._line(img, p, q, occ_t)
        dj._line(om, p, q, occ_t)
    for p, q in others:
        dj._line(img, p, q, 1)
    return img.astype(bool), fm.astype(bool), om.astype(bool)


N0 = dj.N0
OCC55 = (dj._poly(N0, -55.0, -22), dj._poly(N0, -55.0, 22))
FAR = ((20, 20), (20, 56))        # an occluder that touches nothing

CASES = [
    # A: IC-j7 case 1b verbatim -- the mechanism is known by construction
    ("A straight crossing at -55deg (IC-j7 case 1b)",
     [((40, 100), N0), (N0, (160, 100))], [], [OCC55],
     "t_touch_split", "merge_touch"),
    # C: a 6px gap and no crossing at all, so neither half has a node
    ("C 6px gap, occluder touches nothing",
     [((40, 100), (100, 100)), ((106, 100), (170, 100))], [], [FAR],
     "t_no_node", "free_end"),
    # D: two separate junctions with a bent relay between them, no occluder
    #    on the path
    ("D two junctions, bent relay of other ink between them",
     [((40, 100), (100, 100)), ((104, 123), (104, 190))],
     [((100, 100), (104, 123)), ((100, 60), (100, 100)),
      ((104, 123), (150, 123))], [FAR],
     "t_relay_other", None),
    # E: two crossed strokes, far apart, nothing connecting them
    ("E two crossings in separate components",
     [((40, 100), (100, 100)), ((40, 170), (100, 170))],
     [((100, 72), (100, 128)), ((100, 142), (100, 198))], [FAR],
     "t_disconnected", None),
    # F: IC-j7 case 1 verbatim -- rejoins, so the sub-classifier is not asked
    ("F the same crossing at -70deg (IC-j7 case 1)",
     [((40, 100), N0), (N0, (160, 100))], [], [dj.OCC],
     None, None),
]


def icj8(verbose=True):
    rng = random.Random(SEED)
    ok_all = True
    print("IC-j8 B2 arithmetic: a part is >= %d ordered pixels, steps >= 1px, "
          "so path_len >= %.1f > MERGE_PX %.1f -- t_part_absorbed is a "
          "CONSTRUCTION ZERO -> asserted"
          % (cal.PART_MIN_PX, (cal.PART_MIN_PX - 1) * 1.0, sc.MERGE_PX))
    for name, fpieces, others, occl, want, want_flip in CASES:
        ink, fm, om = _case(fpieces, others, occl)
        pre = sc.fragments(ink)
        fp = cal.parts_of(pre[0], fm, om)
        op = cal.parts_of(pre[0], om, fm)
        if len(fp) != 2 or not op:
            print(f"IC-j8 {name}: UNUSABLE ({len(fp)} F parts, {len(op)} "
                  f"occluder parts) -> FAIL")
            ok_all = False
            continue
        row, ctx = dj.analyse((ink, pre, fp, op, 0.0), rng, verify=False)
        if want is None:
            good = row["bucket"] != "topology"
            print(f"IC-j8 {name}\n      top bucket {row['bucket']} "
                  f"(expected anything but topology) -> "
                  f"{'PASS' if good else 'FAIL'}")
            ok_all &= good
            continue
        if row["bucket"] != "topology":
            print(f"IC-j8 {name}: top bucket {row['bucket']}, expected "
                  f"topology -> FAIL (fixture, not classifier)")
            ok_all = False
            continue
        got, extra = classify(ctx)
        good = got == want
        fl = {lv: relink(ctx, lv, rng) for lv in LEVELS}
        if want_flip is not None:
            good &= bool(fl[want_flip][0])
        ok_all &= good
        if verbose:
            s = "  ".join(f"{lv}={'Y' if fl[lv][0] else 'n'}"
                          f"{'' if fl[lv][1] else '*dirty'}" for lv in LEVELS)
            print(f"IC-j8 {name}\n      sub-bucket {got} (expected {want})  "
                  f"clusters {extra['n_clusters']}  r0 {extra['r0']} "
                  f"r1 {extra['r1']}\n      flips: {s}"
                  f"{'' if want_flip is None else f'  [criterion: {want_flip}]'}"
                  f"  -> {'PASS' if good else 'FAIL'}")
    print(f"IC-j8: {'PASS' if ok_all else 'FAIL'} "
          f"({len(CASES)} constructed trials + the B2 arithmetic, labels "
          f"written down in the registration)")
    return ok_all


# --------------------------------------------------------------------------

def run(root, which, lo, cnt, limit=0):
    rng = random.Random(SEED)
    crops = cal.load_crops(root, which, lo, cnt, rng)
    trials = cal.build_trials(crops)
    if limit:
        trials = trials[:limit]
    rows, ctxs = [], []
    topo = []
    n = rejoined = 0
    for t in trials:
        row, ctx = dj.analyse(t, rng)
        n += 1
        rejoined += row["bucket"] == "rejoined"
        rows.append(row)
        ctxs.append(ctx)
        if row["bucket"] == "topology":
            topo.append(len(rows) - 1)
    fails = n - rejoined
    print(f"\npanels {which}[{lo}:{lo + cnt}] -> {n} trials, rejoin "
          f"{rejoined / n:.3f}, {fails} failures, of which "
          f"{len(topo)} are `topology` ({len(topo) / max(fails, 1):.1%})")

    sub, extras = {}, {}
    for i in topo:
        b, e = classify(ctxs[i])
        sub[i] = b
        extras[i] = e
    cnt_sub = Counter(sub.values())
    assert cnt_sub.get("t_part_absorbed", 0) == 0, (
        "B2 says t_part_absorbed is impossible; it fired")
    assert sum(cnt_sub.values()) == len(topo)
    print(f"C2: sub-buckets sum {sum(cnt_sub.values())} = topology "
          f"{len(topo)} -> PASS; t_residual 0 is VACUOUS (the decision tree "
          f"is exhaustive), t_part_absorbed 0 is the asserted CONSTRUCTION "
          f"ZERO of B2")

    flips = {}
    for i in topo:
        flips[i] = {lv: relink(ctxs[i], lv, rng) for lv in LEVELS}

    print(f"\n{'sub-bucket':<24}{'n':>5}{'of topo':>9}  "
          + "".join(f"{lv:>14}" for lv in LEVELS))
    for b in SUB:
        idx = [i for i in topo if sub[i] == b]
        if not idx:
            continue
        cells = []
        for lv in LEVELS:
            j = sum(flips[i][lv][0] for i in idx) / len(idx)
            c = sum(flips[i][lv][0] and flips[i][lv][1] for i in idx) / len(idx)
            cells.append(f"{j:>7.0%}/{c:<6.0%}")
        print(f"{b:<24}{len(idx):>5}{len(idx) / len(topo):>8.1%}  "
              + "".join(f"{x:>14}" for x in cells))
    print("  each cell is flip / flip_clean (B4): joined, and joined without "
          "an occluder fragment in the chain.\n  `relay_force` is gates off and "
          "pairing forced at every node; it does NOT force the RIGHT pair, "
          "which IC-j8 case D measures.\n  `relay_only` additionally drops "
          "every rival arm at the path's nodes and is the true upper bound "
          "for forming a relay chain. `merge_px_inf` is a CONSTANT CHANGE, "
          "not a repair.")

    print("\nwhere the relay is (t_relay_other), as a distribution and not "
          "a mean:")
    rs = [relay_stats(ctxs[i]) for i in topo if sub[i] == "t_relay_other"]
    rs = [r for r in rs if r is not None]
    if rs:
        for k, unit in (("n_frags", ""), ("px", "px"), ("reach", "px"),
                        ("tip_gap", "px")):
            v = sorted(r[k] for r in rs)
            q = lambda f: v[min(len(v) - 1, int(f * len(v)))]  # noqa: E731
            print(f"  {k:<8} min {v[0]:>6.1f}{unit}  q25 {q(0.25):>6.1f}  "
                  f"median {q(0.5):>6.1f}  q75 {q(0.75):>6.1f}  "
                  f"max {v[-1]:>6.1f}")
        one = sum(r["n_frags"] == 1 for r in rs)
        near = sum(r["reach"] <= 20.0 for r in rs)
        print(f"  a single relay fragment: {one}/{len(rs)} = "
              f"{one / len(rs):.1%};  the whole path within 20px of the cut: "
              f"{near}/{len(rs)} = {near / len(rs):.1%}")
        print("  a short, near path is the crossing's own middle ridge and the "
              "join is the right one;\n  a long, far path would satisfy "
              "IC-j6's success test by connectivity elsewhere in the\n  "
              "drawing, which is a different claim. Both are printed because "
              "the bucket's\n  definition does not distinguish them.")

    # C3, conditional form: within each bucket >= 10% of topology
    print("\nC3 (conditional): a sub-bucket holding >= 10% of `topology` must "
          "flip >= 50% under its own repair and <= 20% under each other "
          "bucket's repair")
    own = {}
    for b in SUB:
        idx = [i for i in topo if sub[i] == b]
        if not idx or len(idx) / len(topo) < 0.10:
            continue
        lv = MATCHED[b]
        if lv is None:
            print(f"  {b:<24} {len(idx):>4} trials, no matched repair by "
                  f"registration -> NOT ACTIONABLE; falsifier all_levels "
                  f"{sum(flips[i]['all_levels'][0] for i in idx) / len(idx):.1%}"
                  f" (clean "
                  f"{sum(flips[i]['all_levels'][0] and flips[i]['all_levels'][1] for i in idx) / len(idx):.1%})")
            continue
        f = sum(flips[i][lv][0] for i in idx) / len(idx)
        # C3's specificity leg is "each OTHER BUCKET's repair", so the pool
        # is the matched repairs only. `relay_only` and `merge_px_inf` are
        # diagnostics matched to no bucket and are excluded; including
        # relay_only would have failed this leg with a number that is close
        # to an identity. Corrected 2026-10-12, recorded in the work log; the
        # bars did not move.
        pool_m = sorted({l for l in MATCHED.values() if l is not None})
        others = [sum(flips[i][o][0] for i in idx) / len(idx)
                  for o in pool_m if o != lv]
        good = f >= 0.50 and max(others) <= 0.20
        own[b] = f
        print(f"  {b:<24} {len(idx):>4} trials  own {lv} {f:.1%} (bar 50%)  "
              f"others max {max(others):.1%} (bar 20%)  -> "
              f"{'PASS' if good else 'NOT ACTIONABLE'}")

    # NC-j8b: within-bucket shuffle -- immune to population dominance
    print("\nNC-j8b within-bucket shuffle: a repair drawn from the OTHER "
          "buckets' repairs, same trials")
    shuf = random.Random(SEED + 8)
    pool = [l for b, l in MATCHED.items() if l is not None]
    for b in SUB:
        idx = [i for i in topo if sub[i] == b]
        lv = MATCHED[b]
        if not idx or lv is None:
            continue
        alt = [l for l in pool if l != lv]
        k = sum(flips[i][shuf.choice(alt)][0] for i in idx)
        r = k / len(idx)
        star = "  STARVED (< 20 samples) -> NOT ACTIONABLE" if len(idx) < 20 else ""
        gap = own.get(b)
        g = f"gap {100 * (gap - r):+.1f} points (bar 20)" if gap is not None else ""
        print(f"  {b:<24} {k}/{len(idx)} = {r:.1%}  {g}{star}")

    # NC-j8c: the price, on the trials that already rejoin
    print("\nNC-j8c price, on the trials that already rejoin:")
    ok_idx = [i for i, r in enumerate(rows) if r["bucket"] == "rejoined"]
    for lv in LEVELS:
        br = sum(not relink(ctxs[i], lv, rng)[0] for i in ok_idx)
        dirty = sum(not relink(ctxs[i], lv, rng)[1] for i in ok_idx)
        print(f"  {lv:<14} breaks {br}/{len(ok_idx)} = {br / len(ok_idx):.1%}"
              f"   newly dirty {dirty}/{len(ok_idx)} = "
              f"{dirty / len(ok_idx):.1%}")
    print("  NOT node-local: every level above rebuilds the whole image's "
          "link map, so this sees damage anywhere in the crop. The remaining "
          "blind spot is the panel outside the 160px crop.")

    # C5: the anchor to IC-j7's already-measured merge_touch flips
    bad = [i for i in topo if flips[i]["merge_touch"][0]
           and sub[i] != "t_touch_split"]
    print(f"\nC5 anchor: every merge_touch flip must land in t_touch_split -> "
          f"{len(bad)} violations -> {'PASS' if not bad else 'FAIL'}")

    with open(OUT / "icj8_topology.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["i", "sub", "n_clusters", "r0", "r1"]
                           + [f"{lv}_{k}" for lv in LEVELS
                              for k in ("flip", "clean")])
        w.writeheader()
        for i in topo:
            d = {"i": i, "sub": sub[i], **extras[i]}
            for lv in LEVELS:
                d[f"{lv}_flip"] = int(flips[i][lv][0])
                d[f"{lv}_clean"] = int(flips[i][lv][1])
            w.writerow(d)
    samples = [(b, [ctxs[i] for i in topo if sub[i] == b]) for b in SUB
               if any(sub[i] == b for i in topo)]
    print(f"montage: {dj.montage(samples, OUT / 'icj8_topology.png')}")
    print(f"per-trial: {OUT / 'icj8_topology.csv'}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--which", default="line")
    ap.add_argument("--from-index", type=int, default=40)
    ap.add_argument("--n-panels", type=int, default=131)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    h = hashlib.sha256(
        Path(sc.__file__).read_bytes()).hexdigest()[:16]
    print(f"C4: stroke_chain.py sha256 {h} (unmodified by this tool)")
    print("self-check: PASS (sub_bucket assert cell with the B2 arithmetic)")
    ok = icj8()
    if args.root and ok:
        OUT.mkdir(parents=True, exist_ok=True)
        run(args.root, args.which, args.from_index, args.n_panels, args.limit)
    elif args.root:
        print("IC-j8 FAILED: no real panel is read (C1)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
