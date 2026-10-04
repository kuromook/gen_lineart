"""Ask whether a single image is usable as line art, instead of which of two is better.

The comparison closed on 2026-10-04: two deletion-only outputs of the same rough
are both damaged, so a judge ranks the damage, and damage is computable. The
question that survives is absolute. It is also the one that bounds the direction
rather than the model: if the pixel oracle -- the project's own definition of a
correct deletion -- is not usable either, the ceiling belongs to deletion itself
and no classifier reaches past it.

Four arms, one image at a time, no pairing visible:

  rough      the conditioning, nothing removed -- the floor
  classifier Track C's deletion
  oracle     the pixel oracle's deletion -- the best this direction can do
  gt         the ground-truth line drawing -- the anchor

**Tiles are disjoint across arms.** Showing the same drawing four times would
let the judge remember a cleaner version and compare, which is exactly what
this design exists to avoid. The cost is that arm differences are estimated
between tiles, which is acceptable because the expected effects are large: this
is a threshold question, not a ranking.

The anchor is not blind -- GT is visibly a clean line drawing and the judge will
know it. That is what an anchor is for: if GT is not rated usable, the scale
itself is wrong and nothing else can be read.
"""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "results/trackc_judge_20261004"
ARMS_DIR = ROOT / "stroke_projected"
GT = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line")
TILES = Path("/home/sh1/deepl/lineart/dataset/pairs_480/holdout_lineart_family.txt")
STAGE = ROOT / "stage_absolute"

ARMS = ["rough", "classifier", "oracle", "gt"]
PER_ARM = 15
N_QUINTILES = 5
N_REPEATS = 8
SEED = 20261004


def main():
    rng = np.random.default_rng(SEED)
    tiles = [Path(l.strip()).stem for l in open(TILES) if l.strip()]
    tiles = [t for t in tiles if (GT / f"{t}.jpg").exists()
             and all((ARMS_DIR / a / f"{t}_out.png").exists() for a in ARMS if a != "gt")]

    gt_ink = {}
    for t in tiles:
        g = np.asarray(Image.open(GT / f"{t}.jpg").convert("L").resize((480, 480)))
        gt_ink[t] = float((g < 128).mean())

    per_quintile = PER_ARM // N_QUINTILES * len(ARMS)       # 3 per arm per quintile
    assignment = {a: [] for a in ARMS}
    for bucket in np.array_split(np.array(sorted(tiles, key=lambda t: gt_ink[t])), N_QUINTILES):
        idx = rng.choice(len(bucket), size=per_quintile, replace=False)
        picked = [str(bucket[i]) for i in idx]
        rng.shuffle(picked)
        for k, a in enumerate(ARMS):
            assignment[a] += picked[k * (PER_ARM // N_QUINTILES):(k + 1) * (PER_ARM // N_QUINTILES)]
    for a in ARMS:
        print(f"  {a:11s} {len(assignment[a])} tiles, "
              f"GT ink {min(gt_ink[t] for t in assignment[a]):.3f}-"
              f"{max(gt_ink[t] for t in assignment[a]):.3f}")
    assert len({t for a in ARMS for t in assignment[a]}) == sum(len(v) for v in assignment.values()), \
        "tiles must be disjoint across arms"

    (STAGE / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for a in ARMS:
        for t in assignment[a]:
            src = (GT / f"{t}.jpg") if a == "gt" else (ARMS_DIR / a / f"{t}_out.png")
            name = f"{a}__{t}.png"
            Image.open(src).convert("L").resize((480, 480), Image.LANCZOS).save(
                STAGE / "img" / name, optimize=True)
            items.append({"arm": a, "tile": t, "img": name, "gt_ink": round(gt_ink[t], 5)})
    total = sum(p.stat().st_size for p in (STAGE / "img").glob("*.png"))
    print(f"images: {len(items)} files, {total/1e6:.1f} MB")

    rng.shuffle(items)
    rated = [{"id": f"a{i:04d}", **it, "rep": ""} for i, it in enumerate(items)]
    ridx = rng.choice(len(rated), size=N_REPEATS, replace=False)
    reps = [{"id": f"ar{j:04d}", **{k: rated[k_][k] for k in ("arm", "tile", "img", "gt_ink")},
             "rep": rated[k_]["id"]} for j, k_ in enumerate(ridx)]
    pos = sorted(rng.choice(range(len(rated) // 2, len(rated)), size=N_REPEATS, replace=False))
    for p, rw in zip(pos, reps):
        rated.insert(p, rw)
    print(f"items: {len(rated)} ({N_REPEATS} repeats for self-consistency, placed in the back half)")

    json.dump({"arms": ARMS, "items": rated}, open(STAGE / "items.json", "w"),
              separators=(",", ":"))
    with open(ROOT / "ratings_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "arm", "tile", "gt_ink", "repeat_of"])
        for r in rated:
            w.writerow([r["id"], r["arm"], r["tile"], r["gt_ink"], r["rep"]])
    json.dump({"arms": ARMS, "per_arm": PER_ARM, "seed": SEED, "n_items": len(rated),
               "n_repeats": N_REPEATS, "tiles_disjoint_across_arms": True,
               "question": "is this usable as line art",
               "anchor": "gt is deliberately recognisable; if it is not rated usable the scale is wrong",
               "purpose": "whether any deletion-only output clears the bar, and so whether the "
                          "ceiling belongs to the deletion direction rather than to Track C"},
              open(ROOT / "design_absolute.json", "w"), indent=1)
    print("manifest:", ROOT / "ratings_manifest.csv")


if __name__ == "__main__":
    main()
