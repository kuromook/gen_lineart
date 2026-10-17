"""IC-j11: the bounded MERGE_PX / C_UNMATCHED sweep.

Registered in doc/work_log.md (2026-10-17, commit c9074ad) BEFORE this file
existed. Everything below is from that registration, not written after.

  grid        MERGE_PX in {6,8,10,12,16} x C_UNMATCHED in {0.5,1.0,2.0}
              x touch merge {on, off} = 30 cells
  exploration [40:171], the labelled discovery set. NO cell is adopted here.
  reserved    [171:271], 100 panels never read by anything, used ONCE and
              only if a cell is proposable (--verify).

  IC   MERGE_PX=6, C_UNMATCHED=1.0, touch ON  -> rejoin 0.752, wrong 0.061
       MERGE_PX=6, C_UNMATCHED=1.0, touch OFF -> rejoin 0.643, wrong 0.061
       both asserted to +/- 0.005, measured by different code on different
       days. If either fails the sweep is NOT READ AT ALL.

  AMENDMENT G1 (2026-10-17, from a plumbing test on an UNREGISTERED
  4-panel fixture [300:304], before any sweep number was read): the touch
  switch moves `absorbed_frags` (12->13, 3->4, 7->8) and the number of
  distinct merged nodes (5->4, 9->8) while leaving `absorbed_px` IDENTICAL,
  because the extra absorbed fragments are single-pixel ones whose
  `path_len` is 0.0. The registered expected shape 3 was written on
  `absorbed_px` alone, which is blind to them, so `absorbed_frags` is swept
  and printed beside it. No bar changes.

  decisive quantity: `broken` -- trials that rejoin under the installed
  chainer and stop rejoining in the cell. NC-j8c measured merge_px_inf
  breaking 111 of 200 (55.5%), so recall alone cannot decide this.

  proposable (NOT adopted) := broken = 0 by set AND newly_wrong = 0 by set
                              AND score - score_baseline >= +0.05
  adoption  := the same three legs in difference form on [171:271], read
               once, with IC-j9's two refusal preconditions (< 60 trials ->
               UNDERPOWERED; reference rejoin outside its [40:171] value
               +/- 0.10 -> NOT COMPARABLE).

Expected shapes, written down first: rejoin rises with MERGE_PX and flattens
by 12 (the ridge maximum is 11.5px); wrong rises monotonically; absorbed_px
rises monotonically but is PRINTED not asserted (absorption cascades through
node ids); rejoin >= baseline + 0.05 at MERGE_PX = 12, whose failure means
the ridge is not blocked by MERGE_PX; and `broken > 0` somewhere in the grid,
or the grid is too narrow and that is said rather than read as a free win.

Usage:
  sweep_merge_px.py --root <panel dir>                  # explore [40:171]
  sweep_merge_px.py --root <panel dir> --verify M C T    # the reserved set
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_stroke_chain as cal  # noqa: E402
import stroke_chain as sc  # noqa: E402

OUT = Path("results/stroke_chain_20261010")
MERGE_GRID = (6.0, 8.0, 10.0, 12.0, 16.0)
CU_GRID = (0.5, 1.0, 2.0)
TOUCH_GRID = (True, False)

EXPLORE_FROM, EXPLORE_N = 40, 131       # the labelled discovery set
RESERVED_FROM, RESERVED_N = 171, 100    # registered unread

IC_EXPECT = {True: (0.752, 0.061), False: (0.643, 0.061)}
IC_TOL = 0.005
PROPOSE_GAIN = 0.05
J9_MIN_TRIALS, J9_REF_TOL = 60, 0.10
OCC_SURVIVE_MIN = 0.80   # IC-j12: below this, `wrong` is not comparable


def strip_touch(trials):
    """The touch-merge-OFF arm: the pre-adoption path, reached by handing
    build_links a `pre` whose stats carry no touch set. Nothing is patched
    and no constant is changed -- merge_nodes(.., touch=None) is still the
    pre-adoption branch and this is the only caller that uses it."""
    out = []
    for ink2, pre, fp, op, ang in trials:
        frags, nodes, skel, stats = pre
        s2 = {k: v for k, v in stats.items() if k != "touch"}
        out.append((ink2, (frags, nodes, skel, s2), fp, op, ang))
    return out


def run_cell(trials, merge_px, c_unmatched):
    """(rejoin set, wrong set, absorbed_px total) by trial index."""
    old = (sc.MERGE_PX, sc.C_UNMATCHED)
    sc.MERGE_PX, sc.C_UNMATCHED = merge_px, c_unmatched
    rng = random.Random(cal.SEED)
    ok, bad, apx, afr, surv = set(), set(), 0.0, 0, 0
    try:
        for k, (ink2, pre, fp, op, _) in enumerate(trials):  # noqa: B007
            frags, link, absorbed, stats = sc.build_links(ink2, "real", rng,
                                                          pre=pre)
            r = cal.roots(len(frags), link)
            if r[fp[0]] == r[fp[1]]:
                ok.add(k)
            if any(r[f] == r[o] for f in fp for o in op):
                bad.add(k)
            apx += float(stats.get("absorbed_px", 0.0))
            afr += int(stats.get("absorbed_frags", 0))
            # IC-j12 diagnostic: is there still an occluder part to be wrongly
            # chained to? `wrong` counts exactly that, so if loosening the
            # threshold absorbs the occluder's own ridge, `wrong` can fall
            # because the OPPORTUNITY fell.
            if any(o not in absorbed for o in op):
                surv += 1
    finally:
        sc.MERGE_PX, sc.C_UNMATCHED = old
    return ok, bad, apx, afr, surv


def cell_row(trials_on, trials_off, m, c, touch, base_ok, base_bad, n):
    ok, bad, apx, afr, surv = run_cell(trials_on if touch else trials_off,
                                       m, c)
    rejoin, wrong = len(ok) / n, len(bad) / n
    broken = base_ok - ok
    neww = bad - base_bad
    return {
        "merge_px": m, "c_unmatched": c, "touch": int(touch), "n": n,
        "rejoin": round(rejoin, 4), "wrong": round(wrong, 4),
        "score": round(rejoin - wrong, 4),
        "absorbed_px": round(apx, 1), "absorbed_frags": afr,
        "occ_survive": surv,
        "broken": len(broken), "newly_wrong": len(neww),
        "gained": len(ok - base_ok),
    }, ok, bad


def build(root, which, lo, cnt, per_panel):
    crops = cal.load_crops(root, which, lo, cnt, random.Random(cal.SEED),
                           per_panel=per_panel)
    trials = cal.build_trials(crops)
    print(f"panels [{lo}:{lo + cnt}] -> {len(crops)} candidate sites -> "
          f"{len(trials)} usable trials", flush=True)
    return trials


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--which", default="line", choices=("rough", "line"))
    ap.add_argument("--per-panel", type=int, default=12)
    ap.add_argument("--verify", nargs=3, metavar=("MERGE_PX", "C_UNMATCHED",
                                                  "TOUCH"),
                    help="read the RESERVED set [171:271] once for one cell")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.verify:
        m, c, t = float(args.verify[0]), float(args.verify[1]), \
            args.verify[2].lower() in ("1", "on", "true")
        print(f"IC-j11 VERIFICATION on the RESERVED set "
              f"[{RESERVED_FROM}:{RESERVED_FROM + RESERVED_N}], used ONCE.")
        print(f"cell: MERGE_PX={m} C_UNMATCHED={c} touch={'on' if t else 'off'}")
        trials = build(args.root, args.which, RESERVED_FROM, RESERVED_N,
                       args.per_panel)
        n = len(trials)
        toff = strip_touch(trials)
        base_ok, base_bad, _, _, base_surv = run_cell(trials, 6.0, 1.0)
        if n < J9_MIN_TRIALS:
            print(f"  {n} usable trials < {J9_MIN_TRIALS} -> **UNDERPOWERED, "
                  f"NOT ACTIONABLE**. No leg is read.")
            return 0
        ref = len(base_ok) / n
        print(f"  reference (installed chainer, MERGE_PX=6 C_UNMATCHED=1.0 "
              f"touch on): rejoin {ref:.3f}  wrong {len(base_bad) / n:.3f}")
        print(f"  comparability: the [40:171] reference rejoin is 0.752; "
              f"this set gives {ref:.3f}, "
              f"{'inside' if abs(ref - 0.752) <= J9_REF_TOL else 'OUTSIDE'} "
              f"+/- {J9_REF_TOL:.2f}")
        if abs(ref - 0.752) > J9_REF_TOL:
            print(f"  -> **NOT COMPARABLE**, not a verdict. No leg is read.")
            return 0
        row, _, _ = cell_row(trials, toff, m, c, t, base_ok, base_bad, n)
        print(f"  cell: rejoin {row['rejoin']:.3f}  wrong {row['wrong']:.3f}  "
              f"score {row['score']:.3f}  broken {row['broken']}  "
              f"newly_wrong {row['newly_wrong']}  gained {row['gained']}")
        # IC-j12's pre-registered REFUSAL, read before any leg
        sr = row["occ_survive"] / max(base_surv, 1)
        print(f"  diagnostic: trials with an occluder part still unabsorbed -- "
              f"reference {base_surv}/{n}, cell {row['occ_survive']}/{n}, "
              f"ratio {sr:.3f} (refusal bar: < {OCC_SURVIVE_MIN:.2f})")
        wn_ref = (len(base_bad) / n) / max(base_surv / n, 1e-9)
        wn_cell = row["wrong"] / max(row["occ_survive"] / n, 1e-9)
        print(f"  wrong, raw and opportunity-normalised, always together: "
              f"reference {len(base_bad) / n:.3f} / {wn_ref:.3f}   "
              f"cell {row['wrong']:.3f} / {wn_cell:.3f}")
        if sr < OCC_SURVIVE_MIN:
            print(f"  -> **`wrong` IS NOT COMPARABLE between the arms**: the "
                  f"cell absorbed the occluder parts that `wrong` counts. The "
                  f"cost leg cannot be read and the cell stays UNADOPTED. "
                  f"This is a refusal, not a verdict.")
            return 0
        # IC-j12 cost leg: the difference form on counts, plus broken = 0.
        # The IC-j11 clause `newly_wrong = 0 by set` is deliberately NOT here.
        l1 = row["broken"] == 0 and int(round(row["wrong"] * n)) <= len(base_bad)
        l2 = row["rejoin"] - ref > 0
        l3 = row["score"] - (ref - len(base_bad) / n) >= PROPOSE_GAIN
        print(f"  cost    wrong count {int(round(row['wrong'] * n))} vs "
              f"reference {len(base_bad)}, broken {row['broken']} -> "
              f"{'PASS' if l1 else 'FAIL'}  (IC-j12 bar: count <= reference "
              f"AND broken = 0 by set; the `newly_wrong = 0` clause of IC-j11 "
              f"is DROPPED -- see the registration, it is a weakening and the "
              f"reason is written there. newly_wrong here: "
              f"{row['newly_wrong']})")
        print(f"  recall  rejoin {row['rejoin'] - ref:+.3f} -> "
              f"{'PASS' if l2 else 'FAIL'}  (bar: > 0)")
        print(f"  merit   score {row['score'] - (ref - len(base_bad) / n):+.3f}"
              f" -> {'PASS' if l3 else 'FAIL'}  (bar: >= {PROPOSE_GAIN:+.2f})")
        print(f"  all three required -> "
              f"**{'ADOPT' if (l1 and l2 and l3) else 'NOT ADOPTED'}**")
        print("  the reserved set is used once; it is not re-read under "
              "another bar.")
        return 0

    print(f"IC-j11: bounded sweep on the DISCOVERY set "
          f"[{EXPLORE_FROM}:{EXPLORE_FROM + EXPLORE_N}]. "
          f"No cell is adopted here.")
    trials = build(args.root, args.which, EXPLORE_FROM, EXPLORE_N,
                   args.per_panel)
    n = len(trials)
    toff = strip_touch(trials)

    # ---- IC: the two installed cells must reproduce known values ---------
    print("\nIC, known values measured by different code on different days:")
    base = {}
    for touch in (True, False):
        ok, bad, _, _, _ = run_cell(trials if touch else toff, 6.0, 1.0)
        rj, wr = len(ok) / n, len(bad) / n
        er, ew = IC_EXPECT[touch]
        good = abs(rj - er) <= IC_TOL and abs(wr - ew) <= IC_TOL
        print(f"  touch {'on ' if touch else 'off'}: rejoin {rj:.3f} "
              f"(expected {er:.3f})  wrong {wr:.3f} (expected {ew:.3f})  "
              f"-> {'PASS' if good else 'FAIL'}")
        assert good, ("IC FAILED; the sweep is not read at all -- the grid is "
                      "not scoring the chainer that is installed")
        base[touch] = (ok, bad, rj, wr)
    print(f"  PASS (asserted, +/- {IC_TOL})")
    base_ok, base_bad, base_rj, base_wr = base[True]
    base_score = base_rj - base_wr

    rows = []
    for touch in TOUCH_GRID:
        for m in MERGE_GRID:
            for c in CU_GRID:
                row, _, _ = cell_row(trials, toff, m, c, touch,
                                     base_ok, base_bad, n)
                rows.append(row)
                print(f"  touch {'on ' if touch else 'off'}  "
                      f"MERGE_PX {m:4.1f}  C_UNMATCHED {c:3.1f}  "
                      f"rejoin {row['rejoin']:.3f}  wrong {row['wrong']:.3f}  "
                      f"score {row['score']:.3f}  absorbed "
                      f"{row['absorbed_px']:8.1f}px/{row['absorbed_frags']:4d}f"
                      f"  broken {row['broken']:3d}  "
                      f"newly_wrong {row['newly_wrong']:3d}", flush=True)

    with open(OUT / "icj11_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nevery cell: {OUT / 'icj11_sweep.csv'}")

    # ---- the registered expected shapes, each read explicitly ------------
    print(f"\nbaseline (installed): rejoin {base_rj:.3f}  wrong {base_wr:.3f} "
          f" score {base_score:.3f}")
    print("\nexpected shapes, registered before the run:")
    for touch in TOUCH_GRID:
        for c in CU_GRID:
            seq = [r for r in rows if r["touch"] == int(touch)
                   and r["c_unmatched"] == c]
            seq.sort(key=lambda r: r["merge_px"])
            rj = [r["rejoin"] for r in seq]
            wr = [r["wrong"] for r in seq]
            ap_ = [r["absorbed_px"] for r in seq]
            af_ = [r["absorbed_frags"] for r in seq]
            mono_w = all(b >= a - 1e-9 for a, b in zip(wr, wr[1:]))
            mono_a = all(b >= a - 1e-9 for a, b in zip(ap_, ap_[1:]))
            print(f"  touch {'on ' if touch else 'off'} C_UNMATCHED {c:3.1f}: "
                  f"rejoin {[f'{x:.3f}' for x in rj]}  "
                  f"wrong monotone {mono_w}  absorbed_px monotone {mono_a} "
                  f"  absorbed_frags {af_} "
                  f"(printed, not asserted)")
    at12 = [r for r in rows if r["merge_px"] == 12.0 and r["touch"] == 1
            and r["c_unmatched"] == 1.0]
    if at12:
        d = at12[0]["rejoin"] - base_rj
        print(f"\n  shape 4: rejoin at MERGE_PX=12 (touch on, C_UNMATCHED=1.0)"
              f" {d:+.3f} vs the registered floor +{PROPOSE_GAIN:.2f} -> "
              f"{'met' if d >= PROPOSE_GAIN else 'NOT MET'}")
        if d < PROPOSE_GAIN:
            print("    the ridge is NOT blocked by MERGE_PX; the two-node "
                  "cost becomes the only route.")
    tot_broken = sum(r["broken"] for r in rows)
    print(f"  shape 5: total broken over the grid {tot_broken} -> "
          f"{'some cell breaks trials, as expected' if tot_broken else 'NO cell breaks anything -- the grid is TOO NARROW and this is not a free win'}")

    prop = [r for r in rows if r["broken"] == 0 and r["newly_wrong"] == 0
            and r["score"] - base_score >= PROPOSE_GAIN]
    print(f"\nproposable cells (broken = 0 AND newly_wrong = 0 AND score "
          f">= baseline {base_score:.3f} + {PROPOSE_GAIN:.2f}): {len(prop)}")
    for r in sorted(prop, key=lambda r: -r["score"]):
        print(f"  MERGE_PX {r['merge_px']:4.1f}  C_UNMATCHED "
              f"{r['c_unmatched']:3.1f}  touch "
              f"{'on' if r['touch'] else 'off'}  score {r['score']:.3f} "
              f"({r['score'] - base_score:+.3f})  gained {r['gained']}")
    if not prop:
        print("  **NONE. That is the result and it is the headline:** "
              f"MERGE_PX = {sc.MERGE_PX} stands as the value that survived a "
              f"sweep rather than the value nobody tested.")
    else:
        print("  proposable is NOT adopted. Adoption needs "
              f"--verify on the reserved set [{RESERVED_FROM}:"
              f"{RESERVED_FROM + RESERVED_N}], read once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
