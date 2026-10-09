"""IC1: reproduce a known cell before reading anything else.

The anchor is a CSV this pipeline did not produce --
../lineart-controlnet-sdxl-fidelity/results/pool_inventory_20260911/inventory.csv,
written by another track on 2026-09-11 -- read against `fill_mask.py`'s own
implementation of the same definition. Pre-registered in doc/work_log.md
2026-10-09: pass is 5 of 7 shared-pool cells within 0.002 on the same seeded
sample; if the sample cannot be reproduced, the fallback is the all-tile mean
with the published value inside +/- 2 SE of a 120-tile draw.

Leg 2 is the sensitivity, not a pass bar: the same tiles through scipy's exact
EDT, expected within 0.02 of leg 1. A fill number that moves more than that
between the two is an artefact of cv2's 3x3 approximation, not a property of
the pool.

Usage:  check_fill_instrument.py [--out results/.../ic1.csv]
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import fill_ratio_cv2, fill_ratio_exact, ink_of, load_gray  # noqa: E402

SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
INVENTORY = Path(
    "/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/"
    "pool_inventory_20260911/inventory.csv"
)
SDXL_TRACK = Path("/home/sh1/deepl/lineart-controlnet-sdxl-fidelity")
SAMPLE = 120
SEED = 20260911
TOL_LEG1 = 0.002
TOL_LEG2 = 0.02
MIN_PASS = 5


def source_of(name):
    """Same rule as inventory_pair_pools.source_of -- strip trailing coords."""
    stem = name[:-4] if name.endswith(".jpg") else name
    parts = stem.split("_")
    while parts and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts) or stem


def gather():
    """Rebuilt from inventory_pair_pools.gather(), including the trainlist
    universe -- not because this check needs those pools, but because they
    consume the same shared RNG and dropping them would change every sample
    drawn after them."""
    pools = defaultdict(list)
    for split in ("train", "test"):
        d = SHARED / split / "line"
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.suffix == ".jpg":
                pools[f"shared/{split}:{source_of(p.name)}"].append(p)

    train_list = SDXL_TRACK / "data/train_list.txt"
    if train_list.exists():
        line_dir = SDXL_TRACK / "data/line"
        for name in (l.strip() for l in open(train_list)):
            if name:
                p = line_dir / name
                if p.exists():
                    pools[f"trainlist:{source_of(name)}"].append(p)
    return pools


def replay_sampling(pools, min_tiles=20):
    """Replay the published run's RNG consumption exactly: pools merged below
    min_tiles, iterated by descending size, and for each one profile_pool()
    draws `SAMPLE` then contact_sheet() draws 32."""
    small = [k for k, v in pools.items() if len(v) < min_tiles]
    merged = defaultdict(list)
    for k in small:
        universe = k.split(":")[0]
        merged[f"{universe}:(other <{min_tiles} tiles)"] += pools.pop(k)
    pools.update(merged)

    rng = random.Random(SEED)
    drawn = {}
    for name in sorted(pools, key=lambda k: -len(pools[k])):
        paths = pools[name]
        drawn[name] = paths if len(paths) <= SAMPLE else rng.sample(paths, SAMPLE)
        if len(paths) > 32:  # contact_sheet's own draw, same rng
            rng.sample(paths, 32)
    return pools, drawn


def published():
    with open(INVENTORY) as f:
        return {r["pool"]: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/baseline_fill_20261009/ic1_fill_ratio.csv")
    args = ap.parse_args()

    pub = published()
    cells = [p for p in pub if p.startswith("shared/") and "(other" not in p]
    pools, drawn = replay_sampling(gather())

    rows, n_pass, n_cells = [], 0, 0
    for pool in sorted(cells, key=lambda p: -int(pub[p]["n_total"])):
        want = float(pub[pool]["fill_ratio"])
        want_n = int(pub[pool]["n_total"])
        have = pools.get(pool)
        if have is None:
            rows.append({"pool": pool, "status": "pool_missing_now", "published": want})
            continue
        sample = drawn[pool]
        cv2_vals = [fill_ratio_cv2(ink_of(load_gray(p))) for p in sample]
        exact_vals = [fill_ratio_exact(ink_of(load_gray(p))) for p in sample]
        got, got_exact = float(np.mean(cv2_vals)), float(np.mean(exact_vals))
        delta = abs(got - want)
        ok = delta <= TOL_LEG1
        n_cells += 1
        n_pass += int(ok)
        rows.append({
            "pool": pool,
            "status": "ok" if ok else "off",
            "published": round(want, 4),
            "mine_cv2": round(got, 4),
            "mine_exact": round(got_exact, 4),
            "delta_leg1": round(delta, 4),
            "delta_leg2": round(abs(got - got_exact), 4),
            "n_total_published": want_n,
            "n_total_now": len(have),
            "n_sampled": len(sample),
            "sample_sd": round(float(np.std(cv2_vals, ddof=1)), 4),
        })
        print(f"{pool:28} published={want:.4f} mine={got:.4f} "
              f"(exact {got_exact:.4f})  d1={delta:.4f}  {'ok' if ok else 'OFF'}",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pool", "status", "published", "mine_cv2", "mine_exact", "delta_leg1",
              "delta_leg2", "n_total_published", "n_total_now", "n_sampled", "sample_sd"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    leg2_max = max((r.get("delta_leg2", 0) for r in rows if "delta_leg2" in r), default=0)
    verdict = "PASS" if n_pass >= MIN_PASS else "FAIL -> use the registered fallback"
    print(f"\nIC1 leg 1 (seeded replay, cv2): {n_pass}/{n_cells} cells within "
          f"{TOL_LEG1} -> {verdict}")
    print(f"IC1 leg 2 (cv2 vs exact EDT): max gap {leg2_max:.4f} "
          f"(sensitivity bar {TOL_LEG2}, not a pass bar)")
    print(f"saved: {out}")
    return 0 if n_pass >= MIN_PASS else 1


if __name__ == "__main__":
    sys.exit(main())
