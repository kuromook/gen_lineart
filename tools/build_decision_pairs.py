"""Spend the judge's time where the metric cannot already answer.

The 300-pair set was built for a clean experiment rather than for the decision
it was reopened to inform, and the judge said so. Two things were wrong with it:

  * its "decisive" pairing, classifier vs placebo, is one f1 signal already
    answers (+0.0273, classifier higher on 78.6% of tiles). Agreement there
    confirms the metric; it does not add to it.
  * the pairing that actually decides whether Track C should make a structural
    change -- classifier vs rough, "is the deleted drawing better than the one
    it was deleted from" -- got a third of the set, and on the pixel-level set
    43% of its answers were "cannot tell".

So this set is stratified by the metric's own per-tile opinion and weighted
toward the places that opinion is weakest or negative. Every judgement is then
either something the metric cannot supply, or a direct test of the claim Track
C is relying on.

It is sized for a decision, not for fitting a scorer. If the readout says the
eye is not predicted by signal, that is the moment to collect more.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "results/trackc_judge_20261004"
TILES = Path("/home/sh1/deepl/lineart/dataset/pairs_480/holdout_lineart_family.txt")

VARIANTS = ["rough", "classifier", "placebo", "oracle", "placebo_oracle"]
SEED = 20261004

N_ROUGH = 64        # 4 strata x 16, over signal(classifier) - signal(rough)
N_PLACEBO = 24      # where signal is least sure that the classifier won
N_INSTRUMENT = 12   # the pre-registered check, kept small: it passed 17/17
N_REPEATS = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=str(ROOT / "stroke_projected"))
    ap.add_argument("--signal-per-tile", default=str(ROOT / "stroke_projected_signal_per_tile.csv"))
    ap.add_argument("--stage", default=str(ROOT / "stage_decision"))
    ap.add_argument("--prefix", default="v")
    args = ap.parse_args()

    arms, stage, prefix = Path(args.arms), Path(args.stage), args.prefix
    rng = np.random.default_rng(SEED)

    sig = defaultdict(dict)
    sp = Path(args.signal_per_tile)
    if sp.exists():
        for r in csv.DictReader(open(sp)):
            sig[r["arm"]][Path(r["tile"]).stem] = float(r["signal"])
    have_signal = all(len(sig.get(a, {})) for a in VARIANTS)

    tiles = [Path(l.strip()).stem for l in open(TILES) if l.strip()]
    tiles = [t for t in tiles if all((arms / v / f"{t}_out.png").exists() for v in VARIANTS)]
    if have_signal:
        usable = set.intersection(*[set(sig[a]) for a in VARIANTS])
        tiles = [t for t in tiles if t in usable]
    print(f"tiles usable: {len(tiles)}  (per-tile signal available: {have_signal})")

    if have_signal:
        d_rough = {t: sig["classifier"][t] - sig["rough"][t] for t in tiles}
        d_plac = {t: sig["classifier"][t] - sig["placebo"][t] for t in tiles}
    else:
        # Measuring the per-tile signal takes far longer than the judge should be
        # kept waiting, so sampling falls back to GT-ink quintiles. The metric's
        # claim is a property of the tile, not of the sample, so the readout
        # buckets by it once the measurement lands -- the stratification only
        # buys efficiency, never validity.
        d_rough = d_plac = {}

    def spread(pool, k):
        """Even coverage of the pool, by signal delta when known, else by GT ink."""
        if have_signal:
            order = sorted(pool, key=lambda t: d_rough[t])
        else:
            order = sorted(pool, key=lambda t: gt_ink[t])
        out = []
        for s in np.array_split(np.array(order), 4):
            idx = rng.choice(len(s), size=min(k // 4, len(s)), replace=False)
            out += [str(s[i]) for i in sorted(idx)]
        return out

    gt_ink = {}
    if not have_signal:
        GT = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line")
        for t in tiles:
            g = np.asarray(Image.open(GT / f"{t}.jpg").convert("L").resize((480, 480)))
            gt_ink[t] = float((g < 128).mean())

    chosen_rough = spread(tiles, N_ROUGH)
    if have_signal:
        lo = [d_rough[t] for t in chosen_rough]
        print(f"classifier vs rough: {len(chosen_rough)} tiles, signal delta "
              f"{min(lo):+.4f}..{max(lo):+.4f}; "
              f"{sum(1 for t in chosen_rough if d_rough[t] <= 0)} the metric calls no gain")
    else:
        print(f"classifier vs rough: {len(chosen_rough)} tiles over GT-ink quartiles")

    if have_signal:
        weakest = sorted(tiles, key=lambda t: d_plac[t])[:int(len(tiles) * 0.4)]
    else:
        weakest = [t for t in tiles if t not in set(chosen_rough)] or tiles
    idx = rng.choice(len(weakest), size=min(N_PLACEBO, len(weakest)), replace=False)
    chosen_plac = [weakest[i] for i in sorted(idx)]
    print(f"classifier vs placebo: {len(chosen_plac)} tiles"
          + (" from the weakest 40% of the metric's margin" if have_signal else ""))

    idx = rng.choice(len(tiles), size=N_INSTRUMENT, replace=False)
    chosen_inst = [tiles[i] for i in sorted(idx)]
    print(f"instrument check: {len(chosen_inst)} tiles")

    # --- sprites -----------------------------------------------------------
    staged = sorted(set(chosen_rough) | set(chosen_plac) | set(chosen_inst))
    (stage / "img").mkdir(parents=True, exist_ok=True)
    for t in staged:
        sprite = Image.new("L", (480 * len(VARIANTS), 480), 255)
        for i, v in enumerate(VARIANTS):
            sprite.paste(Image.open(arms / v / f"{t}_out.png").convert("L"), (480 * i, 0))
        sprite.save(stage / f"img/{t}.png", optimize=True)
    total = sum(p.stat().st_size for p in (stage / "img").glob("*.png"))
    print(f"sprites: {len(staged)} files, {total/1e6:.1f} MB")

    # --- pairs -------------------------------------------------------------
    vi = {v: i for i, v in enumerate(VARIANTS)}
    def delta(t, a, b):
        return (sig[a][t] - sig[b][t]) if have_signal else ""
    seq = ([(t, "classifier", "rough", delta(t, "classifier", "rough")) for t in chosen_rough]
           + [(t, "classifier", "placebo", delta(t, "classifier", "placebo")) for t in chosen_plac]
           + [(t, "oracle", "placebo_oracle",
               delta(t, "oracle", "placebo_oracle")) for t in chosen_inst])
    rng.shuffle(seq)
    pairs = []
    for i, (t, a, b, delta) in enumerate(seq):
        l, r = (a, b) if rng.random() < 0.5 else (b, a)
        pairs.append({"id": f"{prefix}{i:04d}", "tile": t, "l": vi[l], "r": vi[r],
                      "pairing": f"{a}_vs_{b}", "delta": (round(float(delta), 5) if delta != "" else ""), "rep": ""})
    ridx = rng.choice(len(pairs), size=N_REPEATS, replace=False)
    reps = [{"id": f"{prefix}r{j:04d}", "tile": pairs[k]["tile"], "l": pairs[k]["r"],
             "r": pairs[k]["l"], "pairing": pairs[k]["pairing"],
             "delta": pairs[k]["delta"], "rep": pairs[k]["id"]} for j, k in enumerate(ridx)]
    pos = sorted(rng.choice(range(len(pairs) // 2, len(pairs)), size=N_REPEATS, replace=False))
    for p, rw in zip(pos, reps):
        pairs.insert(p, rw)
    print(f"pairs: {len(pairs)} ({len(reps)} repeats, sides swapped, back half)")

    json.dump({"variants": VARIANTS, "pairs": pairs}, open(stage / "pairs.json", "w"),
              separators=(",", ":"))
    with open(ROOT / f"pairs_{prefix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "tile", "pairing", "left_variant", "right_variant",
                    "signal_delta", "repeat_of"])
        for p in pairs:
            w.writerow([p["id"], p["tile"], p["pairing"], VARIANTS[p["l"]],
                        VARIANTS[p["r"]], p["delta"], p["rep"]])
    json.dump({"variants": VARIANTS, "seed": SEED, "n_pairs": len(pairs),
               "n_repeats": N_REPEATS,
               "allocation": {"classifier_vs_rough": len(chosen_rough),
                              "classifier_vs_placebo": len(chosen_plac),
                              "oracle_vs_placebo_oracle": len(chosen_inst)},
               "purpose": "decide whether Track C's gain is visible, not fit a scorer",
               "stratification": ("signal quartiles" if have_signal else "GT-ink quartiles; the "
                                 "metric claim is attached at readout time instead")},
              open(ROOT / f"design_{prefix}.json", "w"), indent=1)
    print("manifest:", ROOT / f"pairs_{prefix}.csv")


if __name__ == "__main__":
    main()
