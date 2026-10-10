"""IC-j7: split IC-j6's rejoin failures into buckets, each naming a repair.

Registered 2026-10-10 in doc/work_log.md (commit 193effa) before this file
existed. IC-j6 rejoins 65.4% of cut strokes; the other 34.6% is one number
covering several different repairs, and the H-stroke-3 entry put splitting it
on the critical path.

Nothing here changes a chainer constant (criterion C4).

Amendments to the registration, each with its reason. A1 and A2 are design
errors in the registered counterfactuals that the constructed IC-j7 trials
caught before any real data was read, which is what IC-j7 is for.

A1  All four counterfactual levels are now computed for every failure and
    printed side by side -- `gates` (THETA_MAX = D_MAX = inf), `force`
    (C_UNMATCHED = 1e9), `both`, and `drop_rival` -- because the registered
    causes are not independent and the reason is arithmetic, not empirical:
    `THETA0 = 30` and `C_UNMATCHED = 1.0`, so break-even is `2.0` and any
    `theta > THETA_MAX = 60` already puts `theta / THETA0 > 2.0` on its own.
    The angle gate is exactly redundant against break-even at 60 deg and
    bites only in combination with the offset. The *matched* counterfactual
    per bucket stays the minimal one that targets the bucket's named cause
    (`gates` for `gate_*`, `force` for `cost_break_even`, `drop_rival` for
    `wrong_arm_*`) so that the buckets remain distinguishable and NC-j7b
    stays a real control; a near-zero `gates` flip is then a property of the
    constants and is read as such, not as a failure of the diagnosis. IC-j7
    cases 2 and 3 are the demonstration.

A2  `wrong_arm_*` was registered with "delete the occluder's arms from the
    node". The rival that wins need not be the occluder -- IC-j7 case 5 is a
    straight stroke entering the node, and dropping the occluder there changes
    nothing. The counterfactual is now `drop_rival`: delete the fragments the
    two true arms were actually linked to.

A3  IC-j7's occluder was registered at 70 deg from the first part of F. On the
    bend cases that puts it 10-20 deg from the second part, the two merge in
    the skeleton and the node stops existing (measured: 0 arms, bucket
    `topology`). It is now at -70 deg, which gives a single merged node and
    the full arm count in every case. No bucket threshold moved; the fixture
    was broken, not the classifier.

A4  Case 4 was registered as "bent 55 deg with a 0.5px offset" = 2.17, which
    sits 0.17 above the 2.0 break-even, inside the few degrees skeletonisation
    moves a measured tangent. It is now a plain 45 deg bend, which measures
    theta 45.0, d 2.06, cost 2.87: 15 deg clear of THETA_MAX, ~1px clear of
    D_MAX, 0.87 past break-even. The three inequalities that make the case a
    `cost_break_even` under the registered definition are asserted, so a
    drifting fixture cannot be reported as a passing classifier.

A5  A fifth level, `rival_force` (drop the rival AND force pairing), is
    reported as well. Measured reason: in IC-j7 case 5 removing the rival
    leaves the true pair at cost 2.90, still past break-even, so it does not
    rejoin -- a single-cause repair does not exist for that failure. The
    column is a diagnostic, never the matched test.

A6  A sixth constructed trial, case 1b, was added -- and it is a finding, not
    a fixture fix. The same straight crossing that case 1 rejoins becomes
    `topology` when the occluder sits at -55 deg. Mechanism, measured: the
    skeleton puts TWO junction clusters at the crossing with a single 1px
    fragment between them; that fragment touches both clusters, but
    `fragments()` records only the dominant cluster per end, so `merge_nodes`
    sees n0 == n1, cannot union, and the two halves of F end up at nodes that
    no matching can reach across. Case 1b pins the defect so a later repair
    has something to turn green. No constant is changed here (C4); the limit
    is written into doc/shortcuts.md.

Usage:
  diagnose_icj6.py --check              # IC-j7 only
  diagnose_icj6.py --root <panel dir>   # IC-j7, then the split
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stroke_chain as sc  # noqa: E402
import calibrate_stroke_chain as cal  # noqa: E402

OUT = Path("results/stroke_chain_20261010")
SEED = cal.SEED
BREAK_EVEN = 2.0 * sc.C_UNMATCHED
OCC_DEG = -70.0          # A3

BUCKETS = ("gate_theta", "gate_d", "gate_both", "cost_break_even",
           "wrong_arm_occluder", "wrong_arm_other", "over_max", "topology",
           "no_tangent", "residual")
LEVELS = ("gates", "force", "both", "drop_rival", "rival_force")
# the matched counterfactual per bucket: the MINIMAL intervention that
# targets the bucket's named cause (A1). None = unflippable by construction,
# reported as such rather than as a 0%.
MATCHED = {"gate_theta": "gates", "gate_d": "gates", "gate_both": "gates",
           "cost_break_even": "force", "wrong_arm_occluder": "drop_rival",
           "wrong_arm_other": "drop_rival", "over_max": None,
           "topology": None, "no_tangent": None, "residual": None}


# --------------------------------------------------------------------------
# the classifier, as a pure function so it can be asserted without images

def bucket_of(n_arms, shared, part_absorbed, l0, l1, a0, a1, occ,
              theta, d, raw, theta_max=None, d_max=None,
              break_even=BREAK_EVEN):
    """Which registered bucket a failed trial falls in.

    l0/l1 are what the chainer linked the two true arms to (None = unmatched),
    a0/a1 are the two true arms, occ is the set of occluder fragment indices.
    """
    theta_max = sc.THETA_MAX if theta_max is None else theta_max
    d_max = sc.D_MAX if d_max is None else d_max
    if part_absorbed or not shared:
        return "topology"
    if n_arms > sc.MAX_ARMS:
        return "over_max"
    if l0 == a1 or l1 == a0:
        return "residual"            # linked to each other, yet the trial failed
    for l in (l0, l1):
        if l is not None:
            return ("wrong_arm_occluder" if l[0] in occ else "wrong_arm_other")
    if not (np.isfinite(raw) and np.isfinite(theta) and np.isfinite(d)):
        return "no_tangent"
    gt, gd = theta > theta_max, d > d_max
    if gt and gd:
        return "gate_both"
    if gt:
        return "gate_theta"
    if gd:
        return "gate_d"
    if raw >= break_even:
        return "cost_break_even"
    return "residual"                # available and cheap: the matcher had no
                                     # reason to refuse it. Must be 0.


def _self_check():
    """One assert cell, values computed by hand (foundation notice 4)."""
    A, B, W, O = (3, 0), (5, 1), (7, 0), {9}

    def bk(theta, d, l0=None, l1=None, n=4, shared=True, absorbed=False):
        raw = theta / sc.THETA0 + d / sc.D0
        return bucket_of(n, shared, absorbed, l0, l1, A, B, O, theta, d, raw)
    assert bk(80.0, 1.0) == "gate_theta"
    assert bk(10.0, 6.0) == "gate_d"
    assert bk(80.0, 6.0) == "gate_both"
    # 45/30 + 1.71/1.5 = 1.500 + 1.140 = 2.640 >= 2.0, both gates clear
    assert abs(45 / sc.THETA0 + 1.71 / sc.D0 - 2.640) < 1e-3
    assert bk(45.0, 1.71) == "cost_break_even"
    # 10/30 + 0.5/1.5 = 0.667 < 2.0: nothing explains a refusal
    assert bk(10.0, 0.5) == "residual"
    assert bk(10.0, 0.5, l0=(9, 0)) == "wrong_arm_occluder"
    assert bk(10.0, 0.5, l0=W) == "wrong_arm_other"
    assert bk(10.0, 0.5, l0=B) == "residual"      # linked to its true partner
    assert bk(10.0, 0.5, n=9) == "over_max"
    assert bk(10.0, 0.5, shared=False) == "topology"
    assert bk(10.0, 0.5, absorbed=True) == "topology"
    assert bucket_of(4, True, False, None, None, A, B, O,
                     float("nan"), 1.0, float("inf")) == "no_tangent"
    # A1's arithmetic: the angle gate is redundant against break-even alone
    assert sc.THETA_MAX / sc.THETA0 >= BREAK_EVEN, (
        "A1 assumes a gated angle is already past break-even")
    return True


_self_check()


# --------------------------------------------------------------------------
# node view, rebuilt here so stroke_chain.py is not touched; the rebuild is
# asserted identical to build_links' own link on every real trial

def node_view(frags, nodes):
    nmap, absorbed, _ = sc.merge_nodes(frags, nodes)
    arms_by_node = defaultdict(list)
    for fi, (n0, n1) in enumerate(nodes):
        if fi in absorbed:
            continue
        if n0 >= 0:
            arms_by_node[nmap[n0]].append((fi, 0))
        if n1 >= 0:
            arms_by_node[nmap[n1]].append((fi, 1))
    geo = {a: sc.arm_geometry(frags[a[0]], a[1])
           for arms in arms_by_node.values() for a in arms}
    return nmap, absorbed, dict(arms_by_node), geo


def node_pairs(arms, geo, rng, overrides=None, drop=()):
    keep = [i for i, a in enumerate(arms) if a[0] not in drop]
    g = [geo[arms[i]] for i in keep]
    old = {}
    for k, v in (overrides or {}).items():
        old[k] = getattr(sc, k)
        setattr(sc, k, v)
    try:
        pairs, _ = sc.match_node(g, "real", rng)
    finally:
        for k, v in old.items():
            setattr(sc, k, v)
    return [(keep[i], keep[j]) for i, j in (pairs or [])]


def link_from(pairs_by_node, arms_by_node):
    link = {}
    for nd, pairs in pairs_by_node.items():
        arms = arms_by_node[nd]
        for i, j in pairs:
            link[arms[i]], link[arms[j]] = arms[j], arms[i]
    return link


def joined(n_frags, link, f0, f1):
    r = cal.roots(n_frags, link)
    return r[f0] == r[f1]


def shared_node(nodes, nmap, arms_by_node, f0, f1):
    """The merged node holding one end of each part, or None."""
    def ends(f):
        return {nmap[n]: e for e, n in enumerate(nodes[f]) if n >= 0}
    e0, e1 = ends(f0), ends(f1)
    for nd in sorted(set(e0) & set(e1)):
        a0, a1 = (f0, e0[nd]), (f1, e1[nd])
        arms = arms_by_node.get(nd, [])
        if a0 in arms and a1 in arms:
            return nd, a0, a1
    return None, None, None


# --------------------------------------------------------------------------

def analyse(trial, rng, verify=True):
    """One trial -> (row, ctx). ctx carries what the flip tests need."""
    ink2, pre, fp, op, ang = trial
    frags, nodes = pre[0], pre[1]
    nmap, absorbed, arms_by_node, geo = node_view(frags, nodes)
    pairs_by_node = {nd: node_pairs(arms, geo, rng)
                     for nd, arms in arms_by_node.items()}
    link = link_from(pairs_by_node, arms_by_node)
    if verify:
        # the rebuilt node view must reproduce the calibrated chainer exactly
        ref = sc.build_links(ink2, "real", random.Random(0), pre=pre)[1]
        assert link == ref, "node_view diverged from build_links"

    f0, f1 = fp
    ok = joined(len(frags), link, f0, f1)
    nd, a0, a1 = shared_node(nodes, nmap, arms_by_node, f0, f1)
    occ = set(op)
    row = {"ok": ok, "angle": ang, "node": nd, "bucket": None, "n_arms": 0,
           "theta": float("nan"), "d": float("nan"), "raw": float("inf")}
    ctx = {"ink": ink2, "frags": frags, "fp": fp, "op": op, "node": nd,
           "geo": geo, "arms_by_node": arms_by_node,
           "pairs_by_node": pairs_by_node, "nf": len(frags),
           "a0": a0, "a1": a1, "rival": ()}
    if nd is None:
        row["bucket"] = "rejoined" if ok else "topology"
        return row, ctx
    arms = arms_by_node[nd]
    row["n_arms"] = len(arms)
    _, raw, theta, _, d = sc.pair_cost(geo[a0], geo[a1])
    row.update({"theta": theta, "d": d, "raw": raw})
    if ok:
        row["bucket"] = "rejoined"
        return row, ctx
    l0, l1 = link.get(a0), link.get(a1)
    ctx["rival"] = tuple({l[0] for l in (l0, l1) if l is not None})
    row["bucket"] = bucket_of(
        len(arms), True, (f0 in absorbed or f1 in absorbed),
        l0, l1, a0, a1, occ, theta, d, raw)
    return row, ctx


def flip(ctx, level, rng):
    """One counterfactual at the trial's own node: do the two true parts end
    in the same chain afterwards?"""
    nd = ctx["node"]
    if nd is None or level is None:
        return None
    arms = ctx["arms_by_node"][nd]
    kw = {}
    if level == "gates":
        kw["overrides"] = {"THETA_MAX": 1e9, "D_MAX": 1e9}
    elif level == "force":
        kw["overrides"] = {"C_UNMATCHED": 1e9}
    elif level == "both":
        kw["overrides"] = {"THETA_MAX": 1e9, "D_MAX": 1e9, "C_UNMATCHED": 1e9}
    elif level in ("drop_rival", "rival_force"):
        if not ctx["rival"]:
            return None
        kw["drop"] = set(ctx["rival"])
        if level == "rival_force":
            kw["overrides"] = {"C_UNMATCHED": 1e9}
    else:
        raise ValueError(level)
    pbn = dict(ctx["pairs_by_node"])
    pbn[nd] = node_pairs(arms, ctx["geo"], rng, **kw)
    return joined(ctx["nf"], link_from(pbn, ctx["arms_by_node"]),
                  ctx["fp"][0], ctx["fp"][1])


# --------------------------------------------------------------------------
# IC-j7: five constructed trials, expected buckets in the registration

def _line(img, p, q, t=1):
    cv2.line(img, (int(round(p[0])), int(round(p[1]))),
             (int(round(q[0])), int(round(q[1]))), 1, t)


def _poly(p, deg, length):
    r = np.radians(deg)
    return (p[0] + length * np.cos(r), p[1] + length * np.sin(r))


N0 = (100.0, 100.0)
OCC = (_poly(N0, OCC_DEG, -22), _poly(N0, OCC_DEG, 22))
CASES = [
    ("1 straight, occluder clear",
     [((40, 100), N0), (N0, (160, 100))], [], OCC, "rejoined", None),
    ("1b the same crossing at -55deg: two junction clusters, 1px bridge",
     [((40, 100), N0), (N0, (160, 100))], [],
     (_poly(N0, -55.0, -22), _poly(N0, -55.0, 22)), "topology", None),
    ("2 bend 80deg",
     [((40, 100), N0), (N0, _poly(N0, 80, 85))], [], OCC, "gate_theta",
     lambda t, d, c: t > sc.THETA_MAX),
    ("3 parallel, 6px lateral offset",
     [((40, 100), (100, 100)), ((100, 106), (160, 106))], [],
     ((100, 85), (100, 121)), "gate_d",
     lambda t, d, c: d > sc.D_MAX and t <= sc.THETA_MAX),
    ("4 bend 45deg, ungated but past break-even",
     [((40, 100), N0), (N0, _poly(N0, 45, 60))], [], OCC, "cost_break_even",
     lambda t, d, c: t <= sc.THETA_MAX and d <= sc.D_MAX and c >= BREAK_EVEN),
    ("5 bend 50deg with a straight rival",
     [((40, 100), N0), (N0, _poly(N0, 50, 60))], [(N0, (170, 100))], OCC,
     "wrong_arm_other", None),
]


def icj7(verbose=True):
    rng = random.Random(SEED)
    ok_all = True
    for name, fpieces, others, occl, want, inequality in CASES:
        img = np.zeros((200, 200), np.uint8)
        fm = np.zeros((200, 200), np.uint8)
        om = np.zeros((200, 200), np.uint8)
        for p, q in fpieces:
            _line(img, p, q, 1)
            _line(fm, p, q, 1)
        _line(img, occl[0], occl[1], 2)
        _line(om, occl[0], occl[1], 2)
        for p, q in others:
            _line(img, p, q, 1)
        ink = img.astype(bool)
        pre = sc.fragments(ink)
        fp = cal.parts_of(pre[0], fm.astype(bool), om.astype(bool))
        op = cal.parts_of(pre[0], om.astype(bool), fm.astype(bool))
        if len(fp) != 2 or not op:
            print(f"IC-j7 {name}: UNUSABLE ({len(fp)} F parts, "
                  f"{len(op)} occluder parts) -> FAIL")
            ok_all = False
            continue
        row, ctx = analyse((ink, pre, fp, op, 0.0), rng, verify=False)
        got = row["bucket"]
        good = got == want
        # A4: the case must satisfy its bucket's defining inequality too, so a
        # drifting fixture cannot be reported as a passing classifier
        if inequality is not None:
            good &= bool(inequality(row["theta"], row["d"], row["raw"]))
        ok_all &= good
        fl = {lv: flip(ctx, lv, rng) for lv in LEVELS}
        if verbose:
            s = " ".join(f"{lv}={'Y' if fl[lv] else ('-' if fl[lv] is None else 'n')}"
                         for lv in LEVELS)
            print(f"IC-j7 {name}\n      bucket {got} (expected {want})  "
                  f"theta {row['theta']:.1f}  d {row['d']:.2f}  "
                  f"cost {row['raw']:.2f}  arms {row['n_arms']}\n"
                  f"      flips: {s}  -> {'PASS' if good else 'FAIL'}")
    print(f"IC-j7: {'PASS' if ok_all else 'FAIL'} (6 constructed trials, "
          f"buckets written down in the registration)")
    return ok_all


# --------------------------------------------------------------------------

def montage(samples, path, scale=3, per=3):
    """One row per bucket, `per` sampled failures each. F orange, occluder
    blue, the two true arm tips circled green."""
    rows = []
    for bucket, items in samples:
        tiles = []
        for ctx in items[:per]:
            ink, frags = ctx["ink"], ctx["frags"]
            h, w = ink.shape
            vis = np.full((h, w, 3), 255, np.uint8)
            vis[ink] = (40, 40, 40)
            for fi in ctx["op"]:
                xy = np.round(frags[fi]).astype(int)
                vis[xy[:, 1], xy[:, 0]] = (230, 90, 60)
            for fi in ctx["fp"]:
                xy = np.round(frags[fi]).astype(int)
                vis[xy[:, 1], xy[:, 0]] = (40, 140, 240)
            for fi in ctx["rival"]:
                xy = np.round(frags[fi]).astype(int)
                vis[xy[:, 1], xy[:, 0]] = (200, 40, 200)
            vis = cv2.resize(vis, (w * scale, h * scale),
                             interpolation=cv2.INTER_NEAREST)
            for a in (ctx["a0"], ctx["a1"]):
                if a is not None and a in ctx["geo"]:
                    tip = ctx["geo"][a][0] * scale
                    cv2.circle(vis, (int(tip[0]), int(tip[1])), 7,
                               (20, 160, 20), 1)
            cv2.putText(vis, bucket, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(vis)
        if not tiles:
            continue
        while len(tiles) < per:
            tiles.append(np.full_like(tiles[0], 255))
        hh = max(t.shape[0] for t in tiles)
        ww = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, ww - t.shape[1],
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
                 for t in tiles]
        rows.append(np.hstack(tiles))
    if not rows:
        return None
    ww = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 2, 2, 0, ww - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(200, 200, 200))
            for r in rows]
    cv2.imwrite(str(path), np.vstack(rows))
    return path


def run(root, which, lo, n_panels, per_panel):
    rng = random.Random(SEED)
    crops = cal.load_crops(root, which, lo, n_panels, random.Random(SEED),
                           per_panel=per_panel)
    trials = cal.build_trials(crops)
    print(f"{len(crops)} sites from panels [{lo}:{lo + n_panels}] of {which} "
          f"-> {len(trials)} usable trials "
          f"(the H-stroke panels [0:40] are not touched)")

    rows, ctxs = [], []
    for t in trials:
        r, c = analyse(t, rng)
        rows.append(r)
        ctxs.append(c)
    print("node_view reproduced build_links' link on all "
          f"{len(rows)} trials (assert, not a report)")

    n = len(rows)
    rejoined = sum(r["bucket"] == "rejoined" for r in rows)
    cnt = Counter(r["bucket"] for r in rows if r["bucket"] != "rejoined")
    fails = n - rejoined
    assert rejoined + sum(cnt.values()) == n, (rejoined, cnt, n)
    assert cnt.get("residual", 0) == 0, f"C2: residual must be 0, got {cnt}"
    print(f"C2 exhaustive: rejoined {rejoined} + buckets "
          f"{sum(cnt.values())} = {n}, residual 0 -> PASS")
    print(f"IC-j6 recall here: rejoin {rejoined / n:.3f}; "
          f"{fails} failures ({fails / n:.1%}) to split\n")

    lv = {}
    for b in BUCKETS:
        idx = [i for i, r in enumerate(rows) if r["bucket"] == b]
        if not idx:
            continue
        lv[b] = {}
        for level in LEVELS:
            fl = [flip(ctxs[i], level, rng) for i in idx]
            fl = [x for x in fl if x is not None]
            lv[b][level] = (len(fl), sum(fl))

    def pct(t):
        return f"{t[1] / t[0]:.0%}" if t and t[0] else "--"

    print(f"{'bucket':<20}{'n':>5}{'of fails':>9}  "
          f"{'gates':>7}{'force':>7}{'both':>7}{'rival':>7}{'riv+frc':>8}  "
          f"{'theta':>7}{'d':>7}{'cost':>7}")
    for b in BUCKETS:
        if b not in lv:
            continue
        idx = [i for i, r in enumerate(rows) if r["bucket"] == b]
        med = lambda k: np.nanmedian([rows[i][k] for i in idx])  # noqa: E731
        raws = [rows[i]["raw"] for i in idx if np.isfinite(rows[i]["raw"])]
        print(f"{b:<20}{len(idx):>5}{len(idx) / fails:>8.1%}  "
              + "".join(f"{pct(lv[b].get(x)):>7}" for x in LEVELS)
              + f"  {med('theta'):>7.1f}{med('d'):>7.2f}"
              + (f"{np.median(raws):>7.2f}" if raws else f"{'inf':>7}"))
    print("  gates = THETA_MAX/D_MAX removed; force = C_UNMATCHED 1e9; "
          "both = all three;\n  rival = the winning arm deleted; "
          "riv+frc = deleted AND pairing forced. The MATCHED test is the "
          "minimal\n  one per bucket: gates / force / rival respectively "
          "(A1).")
    print("  'force' on cost_break_even is close to an IDENTITY: the pair is "
          "ungated and forcing a pair at an isolated node has nothing else to "
          "choose. The informative number for that bucket is NC-j7c.")

    matched = [(b, lv[b][MATCHED[b]]) for b in lv if MATCHED[b]]
    tot_n = sum(t[0] for _, t in matched)
    tot_k = sum(t[1] for _, t in matched)
    pooled = tot_k / max(tot_n, 1)
    print(f"\nmatched counterfactual, pooled over flippable buckets: "
          f"{tot_k}/{tot_n} = {pooled:.1%}")

    # NC-j7b: each failure is given a bucket drawn from the observed mix
    pool = [b for b in BUCKETS for _ in range(cnt.get(b, 0))]
    shuf = random.Random(SEED + 1)
    nb_n = nb_k = 0
    for i, r in enumerate(rows):
        if r["bucket"] == "rejoined":
            continue
        f = flip(ctxs[i], MATCHED[shuf.choice(pool)], shuf)
        if f is not None:
            nb_n += 1
            nb_k += f
    nb = nb_k / max(nb_n, 1)
    gap = pooled - nb
    print(f"NC-j7b shuffled attribution: {nb_k}/{nb_n} = {nb:.1%}  "
          f"gap {gap * 100:+.1f} points")

    # NC-j7c: the same counterfactuals on trials that already rejoined
    c_n = c_k = 0
    for i, r in enumerate(rows):
        if r["bucket"] != "rejoined":
            continue
        for level in ("both", "drop_rival"):
            f = flip(ctxs[i], level, rng)
            if f is not None:
                c_n += 1
                c_k += (not f)
    print(f"NC-j7c on already-rejoined trials: {c_k}/{c_n} broken by a "
          f"counterfactual = {c_k / max(c_n, 1):.1%} "
          f"(the price of the repair; registered as sensitivity, no bar)")

    print("\nC3: a bucket holding >=10% of failures must flip >=60% under its "
          f"own counterfactual, and NC-j7b must sit >=20 points below {pooled:.1%}")
    big = [b for b in lv if MATCHED[b] and cnt[b] >= 0.10 * fails]
    any60 = False
    for b in sorted(big, key=lambda b: -cnt[b]):
        t = lv[b][MATCHED[b]]
        f = t[1] / max(t[0], 1)
        any60 |= f >= 0.60
        print(f"  {b:<20}{cnt[b]:>5} ({cnt[b] / fails:>5.1%})  "
              f"{MATCHED[b]:<11} flip {f:.1%}  -> "
              f"{'meets 60%' if f >= 0.60 else 'under 60%'}")
    print(f"  NC-j7b gap {gap * 100:+.1f} points -> "
          f"{'at or past 20' if gap >= 0.20 else 'under 20'}")
    actionable = any60 and gap >= 0.20
    print(f"C3: the split is "
          f"{'ACTIONABLE' if actionable else 'NOT ACTIONABLE'} as registered")

    samples = []
    pick = random.Random(SEED)
    for b in BUCKETS:
        items = [ctxs[i] for i, r in enumerate(rows) if r["bucket"] == b]
        if items:
            pick.shuffle(items)
            samples.append((b, items))
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nmontage: {montage(samples, OUT / 'icj7_buckets.png')}")
    with open(OUT / "icj7_failures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bucket", "ok", "angle", "theta",
                                          "d", "raw", "n_arms", "node"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"per-trial rows: {OUT / 'icj7_failures.csv'}")
    return actionable


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--which", default="line", choices=("line", "rough"))
    ap.add_argument("--from-index", type=int, default=cal.CAL_FROM)
    ap.add_argument("--n-panels", type=int, default=cal.CAL_N)
    ap.add_argument("--per-panel", type=int, default=12)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    print("self-check: PASS (bucket_of assert cell, 13 hand-written cases)")
    if not icj7():
        print("C1 failed: nothing is read.")
        sys.exit(1)
    if args.check or not args.root:
        sys.exit(0)
    run(args.root, args.which, args.from_index, args.n_panels, args.per_panel)


if __name__ == "__main__":
    main()
