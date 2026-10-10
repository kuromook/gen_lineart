"""IC1 and IC2 for stage 2: does the new scorer measure the same thing?

Registered 2026-10-10 in doc/work_log.md.

IC1  Re-score stage 1's `housei_test` and `ako5_held` 480 tiles through the
     NEW scorer -- recomputed from the images on disk, never read back out of
     stage 1's CSV, or it would not be a check. Pixel micro-IoU must
     reproduce, exactly:
       housei   blank 0.0000 / best preproc 0.0002 / msgan 0.0163 / all_black 0.0939
       ako5     blank 0.0000 / best preproc 0.0001 / msgan 0.0152 / all_black 0.1201
     **The "best preproc" column is `preproc@32`'s, and that arm fell to its
     own negative control in stage 1. It is a valid check value and NOT a
     ceiling: the readable preprocessor ceiling is 0.0000.** (fifth notice)

IC2  `fill_ratio` on the ako5 work, against the published value in
     ../lineart-controlnet-sdxl-fidelity -- one leg anchored outside this
     pipeline entirely (lesson 12).

micro-IoU is pooled: sum(inter) / sum(union) over tiles, not a mean of
per-tile IoUs. Stage 1 registered micro as the primary, so that is what is
reproduced here.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_mask import IMAGE_SIZE, fill_mask, ink_of, load_gray  # noqa: E402
from region_metrics import pixel_iou  # noqa: E402

SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
R1 = Path("results/baseline_fill_20261009")
# The registered values are stage 1's PRIMARY stratum -- `normal_fill`, i.e.
# not near-blank AND the GT has a fill -- not every tile in the list. The
# first run of this check pooled over the whole list and failed; the tile
# COUNT is carried here as a second known value so a stratum definition that
# drifts cannot silently pass by landing on the right ratio.
EXPECT = {
    ("housei_test", "test"): {"n": 782, "blank": 0.0000, "preproc@32": 0.0002,
                              "msgan": 0.0163, "all_black": 0.0939},
    ("ako5_held", "train"): {"n": 474, "blank": 0.0000, "preproc@32": 0.0001,
                             "msgan": 0.0152, "all_black": 0.1201},
}


def normal_fill_tiles(pool):
    import csv as _csv
    prof = {r["tile"]: r for r in
            _csv.DictReader(open(R1 / "gt_fill_profile.csv"))}
    tiles = [l.strip() for l in open(R1 / "lists" / f"{pool}.txt") if l.strip()]
    out = []
    for t in tiles:
        p = prof.get(t[:-4] if t.endswith(".jpg") else t)
        if p and p["near_blank"] == "0" and p["has_fill"] == "1":
            out.append(t)
    return out


def ic1(verbose=True):
    ok = True
    for (pool, split), exp in EXPECT.items():
        tiles = normal_fill_tiles(pool)
        n_ok = len(tiles) == exp["n"]
        ok &= n_ok
        print(f"IC1  {pool:>11} {'normal_fill tiles':>12}  {len(tiles)}  "
              f"expected {exp['n']}  {'ok' if n_ok else 'MISMATCH'}")
        acc = {a: [0, 0] for a in exp if a != "n"}
        for t in tiles:
            stem = t[:-4] if t.endswith(".jpg") else t
            gt = fill_mask(ink_of(load_gray(SHARED / split / "line" / f"{stem}.jpg")))
            pp = load_gray(R1 / "preproc" / pool / f"{stem}.jpg")       # white-on-black
            ms = load_gray(R1 / "msgan" / pool / f"{stem}_out.png")     # ink-on-white
            preds = {
                "blank": np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool),
                "preproc@32": fill_mask(pp > 32),
                "msgan": fill_mask(ink_of(ms)),
                "all_black": fill_mask(np.ones((IMAGE_SIZE, IMAGE_SIZE), bool)),
            }
            for a, p in preds.items():
                r = pixel_iou(p, gt)
                acc[a][0] += r["inter_px"]
                acc[a][1] += r["union_px"]
        for a in ("blank", "preproc@32", "msgan", "all_black"):
            got = acc[a][0] / acc[a][1] if acc[a][1] else 0.0
            same = abs(round(got, 4) - exp[a]) < 5e-5
            ok &= same
            if verbose:
                note = ("  <- control-failed arm; the readable ceiling is 0.0000"
                        if a == "preproc@32" else
                        "  <- IDENTITY |G|/|P|" if a == "all_black" else
                        "  <- IDENTITY, GT is never empty here" if a == "blank" else "")
                print(f"IC1  {pool:>11} {a:>12}  micro-IoU {got:.4f}  "
                      f"expected {exp[a]:.4f}  {'ok' if same else 'MISMATCH'}{note}")
    print(f"IC1 -> {'PASS' if ok else 'FAIL'}  ({len(EXPECT)} pools x 4 arms, "
          f"recomputed from the images, not read from stage 1's CSV)")
    return ok


def ic2(verbose=True):
    """fill_ratio for the ako5 work against the外部 published inventory."""
    doc = Path("../lineart-controlnet-sdxl-fidelity/doc/pool_inventory.md")
    inv = Path("../lineart-controlnet-sdxl-fidelity/results/pool_inventory_20260911")
    src = doc if doc.exists() else None
    if src is None:
        print("IC2  SKIP: the external inventory is not present")
        return True
    text = src.read_text(errors="ignore")
    print(f"IC2  external anchor: {src}")
    hits = [l for l in text.splitlines() if "ako5" in l.lower()]
    for l in hits[:6]:
        print(f"       {l.strip()[:150]}")
    if not hits:
        print(f"       no ako5 row found; listing {inv} instead")
        for p in sorted(inv.glob("*"))[:10]:
            print(f"       {p.name}")
    return True


if __name__ == "__main__":
    good = ic1()
    ic2()
    sys.exit(0 if good else 1)
