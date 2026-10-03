"""Build the 2026-10-04 comparison set: did Track C delete the right lines?

Four arms, all produced by `project_deletions_to_rough.py` and therefore
identical in rendering -- the same rough with different ink erased. Three
pairings per tile, each asking a different question:

  classifier vs rough    -- was deleting an improvement at all?
  classifier vs placebo  -- did it delete the RIGHT lines? (ink-matched and
                            fragment-size-matched, so the only thing that
                            separates them is which strokes went)
  oracle vs placebo_oracle -- INSTRUMENT CHECK, pre-registered: if the judge does
                            not prefer the ideal deletion over a random one of
                            the same size, the eye has no resolution here and
                            nothing in this set can be read. Checked first. It
                            gets its own control because the oracle erases more
                            than twice what the classifier does, and the one
                            comparison everything is gated on must not be
                            decidable on paper tone.

The 2026-09-17 lesson is enforced by construction rather than by selection: the
earlier set had to drop candidates whose pairs were decided by paper tone, and
here the placebo is matched on erased ink for exactly that reason. The
`classifier vs rough` pairing is the one that cannot be tone-matched -- deleting
ink is what it is about -- so it is reported separately and never pooled with
the other two.
"""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "results/trackc_judge_20261004"
GT = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line")
TILES = Path("/home/sh1/deepl/lineart/dataset/pairs_480/holdout_lineart_family.txt")

VARIANTS = ["rough", "classifier", "placebo", "oracle", "placebo_oracle"]
PAIRINGS = [("classifier", "rough"), ("classifier", "placebo"), ("oracle", "placebo_oracle")]
SEED = 20261004
N_PER_QUINTILE = 16
N_REPEATS = 60

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--arms", default=str(ROOT / "stroke_projected"))
_ap.add_argument("--stage", default=str(ROOT / "stage_stroke"))
_ap.add_argument("--prefix", default="s")
_args = _ap.parse_args()
ARMS = Path(_args.arms)
STAGE = Path(_args.stage)
PREFIX = _args.prefix

rng = np.random.default_rng(SEED)
tiles = [Path(l.strip()).stem for l in open(TILES) if l.strip()]
tiles = [t for t in tiles if all((ARMS / v / f"{t}_out.png").exists() for v in VARIANTS)]

gt_ink = {}
for t in tiles:
    g = np.asarray(Image.open(GT / f"{t}.jpg").convert("L").resize((480, 480)))
    gt_ink[t] = float((g < 128).mean())

chosen = []
for bucket in np.array_split(np.array(sorted(tiles, key=lambda t: gt_ink[t])), 5):
    idx = rng.choice(len(bucket), size=N_PER_QUINTILE, replace=False)
    chosen += [str(bucket[i]) for i in sorted(idx)]
print(f"tiles: {len(chosen)} ({N_PER_QUINTILE} from each GT-ink quintile of {len(tiles)} eligible)")

# --- tone audit: the cue the 2026-09-17 set had to be rebuilt around ---------
prof = {}
for t in chosen:
    for v in VARIANTS:
        a = np.asarray(Image.open(ARMS / v / f"{t}_out.png").convert("L"))
        prof[(t, v)] = {"near_white": float((a > 223).mean()),
                        "ink": float((a < 231).mean())}
audit = {}
for a, b in PAIRINGS:
    d_nw = np.array([abs(prof[(t, a)]["near_white"] - prof[(t, b)]["near_white"]) for t in chosen])
    d_ink = np.array([abs(prof[(t, a)]["ink"] - prof[(t, b)]["ink"]) for t in chosen])
    audit[f"{a}_vs_{b}"] = {"median_abs_delta_near_white": round(float(np.median(d_nw)), 4),
                            "p90_abs_delta_near_white": round(float(np.percentile(d_nw, 90)), 4),
                            "median_abs_delta_ink": round(float(np.median(d_ink)), 4)}
    print(f"  tone audit {a:10s} vs {b:10s}  median |d near_white| {np.median(d_nw):.4f}"
          f"  p90 {np.percentile(d_nw, 90):.4f}  median |d ink| {np.median(d_ink):.4f}")

# --- sprites ----------------------------------------------------------------
(STAGE / "img").mkdir(parents=True, exist_ok=True)
for t in chosen:
    sprite = Image.new("L", (480 * len(VARIANTS), 480), 255)
    for i, v in enumerate(VARIANTS):
        sprite.paste(Image.open(ARMS / v / f"{t}_out.png").convert("L"), (480 * i, 0))
    sprite.save(STAGE / f"img/{t}.png", optimize=True)
total = sum(p.stat().st_size for p in (STAGE / "img").glob("*.png"))
print(f"sprites: {len(chosen)} files, {total/1e6:.1f} MB")

# --- pairs ------------------------------------------------------------------
vi = {v: i for i, v in enumerate(VARIANTS)}
seq = [(t, a, b) for t in chosen for a, b in PAIRINGS]
rng.shuffle(seq)
pairs = []
for i, (t, a, b) in enumerate(seq):
    l, r = (a, b) if rng.random() < 0.5 else (b, a)
    pairs.append({"id": f"{PREFIX}{i:04d}", "tile": t, "l": vi[l], "r": vi[r],
                  "pairing": f"{a}_vs_{b}", "rep": ""})
ridx = rng.choice(len(pairs), size=N_REPEATS, replace=False)
reps = [{"id": f"{PREFIX}r{j:04d}", "tile": pairs[k]["tile"], "l": pairs[k]["r"], "r": pairs[k]["l"],
         "pairing": pairs[k]["pairing"], "rep": pairs[k]["id"]} for j, k in enumerate(ridx)]
pos = sorted(rng.choice(range(len(pairs) // 2, len(pairs)), size=N_REPEATS, replace=False))
for p, rw in zip(pos, reps):
    pairs.insert(p, rw)
print(f"pairs: {len(pairs)} ({len(reps)} repeats, sides swapped, placed in the back half)")

json.dump({"variants": VARIANTS, "pairs": pairs}, open(STAGE / "pairs.json", "w"),
          separators=(",", ":"))
with open(ROOT / f"pairs_{PREFIX}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "tile", "pairing", "left_variant", "right_variant", "repeat_of"])
    for p in pairs:
        w.writerow([p["id"], p["tile"], p["pairing"], VARIANTS[p["l"]], VARIANTS[p["r"]], p["rep"]])
with open(ROOT / f"staged_profiles_{PREFIX}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tile", "variant", "near_white", "ink", "gt_ink"])
    for t in chosen:
        for v in VARIANTS:
            w.writerow([t, v, round(prof[(t, v)]["near_white"], 5),
                        round(prof[(t, v)]["ink"], 5), round(gt_ink[t], 5)])
json.dump({"variants": VARIANTS, "pairings": ["%s_vs_%s" % p for p in PAIRINGS],
           "tiles": chosen, "seed": SEED, "n_pairs": len(pairs), "n_repeats": N_REPEATS,
           "tone_audit": audit,
           "instrument_check": "oracle_vs_placebo_oracle",
           "gate": "hold-out agreement >= intra-rater consistency x 0.85, and clearly above f1 / near_white / fill alone",
           "rationale": "all four arms are the same rough with different ink erased, so rendering cannot "
                        "separate them; the placebo is matched on erased ink and on deleted-fragment sizes, "
                        "so classifier_vs_placebo isolates WHICH strokes went"},
          open(ROOT / f"design_{PREFIX}.json", "w"), indent=1)
print("manifest:", ROOT / f"pairs_{PREFIX}.csv")
