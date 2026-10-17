"""NC-j8d: the global price of the touch merge, on IC-j6's own two numbers.

Registered in doc/work_log.md (commit f62c0ed) with the adoption bars fixed
BEFORE this measurement existed:

    rejoin >= 0.70     (half of the +10.9 points IC-j7 measured as available)
    wrong  <= 0.059    (today 0.039, an allowance of +2.0 points absolute)
    score  >= 0.675    (today 0.615, so +0.06)
    ALL THREE must hold, or the repair is not adopted and that is the
    headline, not a footnote.

Named error this control exists to catch: two genuinely distinct crossings
within MERGE_PX fuse into one node, so `wrong` rises while `rejoin` also
rises, and a recall-only reading would call that a win.

Both arms run on the SAME trials, and that the trial sets are identical is
asserted rather than assumed. `force` is printed beside them as the
trivial-solution identity: pair everything at every node, which buys recall
with precision and is what the bars exist to refuse.
"""

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stroke_chain as sc  # noqa: E402
import stroke_chain_touch as sct  # noqa: E402
import calibrate_stroke_chain as cal  # noqa: E402

BAR_REJOIN, BAR_WRONG, BAR_SCORE = 0.70, 0.059, 0.675

# IC-j9 (doc/work_log.md, 2026-10-16, commit b29b4c4): the same decision in
# the difference form the foundation prescribed, read ONCE on the never-read
# holdout [0:40]. The absolute bars above belong to IC-j8 and to the
# discovery set [40:171]; they are not re-decided.
J9_SCORE_GAIN = 0.05          # half of the discovery set's +0.109
J9_MIN_TRIALS = 60            # below this: UNDERPOWERED, no verdict
J9_REF_REJOIN, J9_REF_TOL = 0.643, 0.10   # outside this: not comparable

# The reference arm's identity, not its behaviour, is what every bar in this
# file was written against. IC-j9 ADOPTED the touch merge on 2026-10-17, so
# stroke_chain.py now carries the repair and this tool would be comparing the
# candidate against itself. A self-comparison reads as "gained 0, LOST 0,
# newly wrong 0", which is indistinguishable from a repair that does nothing
# -- so the verdict is refused rather than printed. Run the pre-adoption file
# (commit 1a82ea5) as the reference to reproduce IC-j8/IC-j9.
PRE_ADOPT_SHA = "e9e49267c339f885"


def per_trial(trials, chainer):
    """(rejoin set, wrong set) by trial index, so the two arms can be
    compared trial by trial and not only by their counts. Added 2026-10-12:
    the first run showed `wrong` identical to three decimals in both arms,
    and an identical COUNT is not an identical SET -- a repair that fixes two
    contaminations and creates two others reads the same."""
    import random as _r
    rng = _r.Random(cal.SEED)
    ok, bad = set(), set()
    for k, (ink2, pre, fp, op, _) in enumerate(trials):
        frags, link, _, _ = chainer.build_links(ink2, "real", rng, pre=pre)
        r = cal.roots(len(frags), link)
        if r[fp[0]] == r[fp[1]]:
            ok.add(k)
        if any(r[f] == r[o] for f in fp for o in op):
            bad.add(k)
    return ok, bad


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--which", default="line")
    ap.add_argument("--from-index", type=int, default=40)
    ap.add_argument("--n-panels", type=int, default=131)
    ap.add_argument("--holdout", action="store_true",
                    help="read the IC-j9 difference-form bars instead of "
                         "IC-j8's absolute ones. Use once, on [0:40].")
    args = ap.parse_args()
    if args.holdout:
        print("IC-j9: the HOLDOUT reading. [40:171] is the discovery set and "
              "is not re-decided here.\n")

    print(f"stroke_chain.py       sha256 {sha(sc.__file__)}  (calibrated, "
          f"unmodified)")
    print(f"stroke_chain_touch.py sha256 {sha(sct.__file__)}  (the candidate: "
          f"the touch merge and nothing else)")
    for k in ("TAN_WIN", "TAN_SKIP", "THETA0", "D0", "W_KAPPA", "KAPPA_REF",
              "THETA_MAX", "D_MAX", "C_UNMATCHED", "MAX_ARMS", "MERGE_PX"):
        assert getattr(sc, k) == getattr(sct, k), k
    print("every constant identical in both files -> asserted")

    rng = random.Random(cal.SEED)
    crops = cal.load_crops(args.root, args.which, args.from_index,
                           args.n_panels, rng)
    base = cal.build_trials(crops)
    cal.sc = sct
    try:
        cand = cal.build_trials(crops)
    finally:
        cal.sc = sc
    assert len(base) == len(cand), (len(base), len(cand))
    for (_, p1, f1, o1, _), (_, p2, f2, o2, _) in zip(base, cand):
        assert f1 == f2 and o1 == o2 and len(p1[0]) == len(p2[0])
    print(f"the two arms run on the same {len(base)} trials -> asserted "
          f"(same fragments, same F parts, same occluder parts)")

    rows = []
    n, r, w, s, _ = cal.score(base)
    rows.append(("stroke_chain (calibrated)", n, r, w, s))
    cal.sc = sct
    try:
        n2, r2, w2, s2, _ = cal.score(cand)
    finally:
        cal.sc = sc
    rows.append(("+ touch merge (candidate)", n2, r2, w2, s2))
    # the trivial solution, printed as an identity
    old = (sc.C_UNMATCHED, sc.THETA_MAX, sc.D_MAX)
    sc.C_UNMATCHED, sc.THETA_MAX, sc.D_MAX = 1e9, 1e9, 1e9
    try:
        n3, r3, w3, s3, _ = cal.score(base)
    finally:
        sc.C_UNMATCHED, sc.THETA_MAX, sc.D_MAX = old
    rows.append(("force, gates off (IDENTITY)", n3, r3, w3, s3))

    print(f"\n{'arm':<30}{'n':>5}{'rejoin':>9}{'wrong':>8}{'score':>8}")
    for name, nn, rr, ww, ss in rows:
        print(f"{name:<30}{nn:>5}{rr:>9.3f}{ww:>8.3f}{ss:>8.3f}")
    print("  score = rejoin - wrong, IC-j6's own figure of merit.\n"
          "  the third row is the trivial solution and is labelled an "
          "IDENTITY: pairing everything\n  raises rejoin and buys it with "
          "wrong, which is exactly what the three bars refuse.")

    ok0, bad0 = per_trial(base, sc)
    ok1, bad1 = per_trial(cand, sct)
    print(f"\ntrial by trial, not only by count:")
    print(f"  rejoined  base {len(ok0)}  candidate {len(ok1)}  "
          f"gained {len(ok1 - ok0)}  LOST {len(ok0 - ok1)}")
    print(f"  wrong     base {len(bad0)}  candidate {len(bad1)}  "
          f"newly wrong {len(bad1 - bad0)}  no longer wrong "
          f"{len(bad0 - bad1)}")
    print(f"  an identical count is not an identical set; the two numbers "
          f"above are what decides it.")

    if sha(sc.__file__) != PRE_ADOPT_SHA:
        print(f"\n**NO VERDICT.** The reference arm is {sha(sc.__file__)}, not "
              f"the pre-adoption chainer {PRE_ADOPT_SHA} that every bar in "
              f"this file was written against.\n  IC-j9 adopted the touch "
              f"merge into stroke_chain.py on 2026-10-17, so the two arms "
              f"above are the same code path and\n  their agreement is an "
              f"IDENTITY, not evidence. To reproduce IC-j8 or IC-j9, check "
              f"out the pre-adoption file (commit 1a82ea5).")
        return 0

    if args.holdout:
        print("\nIC-j9 bars, fixed before this run (commit b29b4c4):")
        if n < J9_MIN_TRIALS:
            print(f"  {n} usable trials < {J9_MIN_TRIALS} -> **UNDERPOWERED, "
                  f"NOT ACTIONABLE**. The repair stays unadopted and waits "
                  f"for a larger reserved set. No leg is read.")
            return 0
        lo, hi = J9_REF_REJOIN - J9_REF_TOL, J9_REF_REJOIN + J9_REF_TOL
        if not (lo <= r <= hi):
            print(f"  reference rejoin {r:.3f} outside {J9_REF_REJOIN:.3f} "
                  f"+/- {J9_REF_TOL:.2f} -> **NOT COMPARABLE**. The two panel "
                  f"ranges are not the same kind of material; this is a "
                  f"comparability failure, not a verdict. No leg is read.")
            return 0
        print(f"  comparability: reference rejoin {r:.3f} inside "
              f"{lo:.3f}..{hi:.3f}, and {n} trials >= {J9_MIN_TRIALS} -> both "
              f"preconditions met, the legs are read")
        nw = len(bad1 - bad0)
        l1 = len(bad1) <= len(bad0) and nw == 0
        l2 = r2 - r > 0
        l3 = s2 - s >= J9_SCORE_GAIN
        print(f"  cost     wrong count {len(bad1)} vs reference {len(bad0)}, "
              f"newly wrong {nw} -> {'PASS' if l1 else 'FAIL'}  "
              f"(bar: count <= reference AND newly wrong = 0)")
        print(f"  recall   rejoin {r2 - r:+.3f} -> "
              f"{'PASS' if l2 else 'FAIL'}  (bar: > 0)")
        print(f"  merit    score {s2 - s:+.3f} -> "
              f"{'PASS' if l3 else 'FAIL'}  (bar: >= {J9_SCORE_GAIN:+.2f})")
        v = "ADOPT" if (l1 and l2 and l3) else "NOT ADOPTED"
        print(f"  all three required -> **{v}**")
        print(f"  this holdout is used once; it is not re-read under another "
              f"bar.")
        return 0

    ok_r, ok_w, ok_s = r2 >= BAR_REJOIN, w2 <= BAR_WRONG, s2 >= BAR_SCORE
    print(f"\nNC-j8d adoption bars, fixed before this measurement:")
    print(f"  rejoin {r2:.3f} vs bar >= {BAR_REJOIN:.3f} "
          f"(was {r:.3f})  -> {'PASS' if ok_r else 'FAIL'}")
    print(f"  wrong  {w2:.3f} vs bar <= {BAR_WRONG:.3f} "
          f"(was {w:.3f})  -> {'PASS' if ok_w else 'FAIL'}")
    print(f"  score  {s2:.3f} vs bar >= {BAR_SCORE:.3f} "
          f"(was {s:.3f})  -> {'PASS' if ok_s else 'FAIL'}")
    verdict = "ADOPT" if (ok_r and ok_w and ok_s) else "NOT ADOPTED"
    print(f"  all three required -> **{verdict}**")
    if not (ok_r and ok_w and ok_s) and r2 > r:
        print("  the repair raises rejoin without clearing every bar; the "
              "headline is that it buys recall with something else, not that "
              "it helped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
