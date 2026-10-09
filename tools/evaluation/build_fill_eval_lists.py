"""Materialise the evaluation lists fixed in doc/work_log.md 2026-10-09.

The split rule, registered before any image was looked at:

    held_out(pool, page) := int(sha1(f"{pool}_{page}").hexdigest()[:8], 16) % 5 == 0

applied to `ako5` and `ako5r`, which have no test split of their own. `housei`
is already split by page (test holds 001/004/009, train the other fifteen), so
its whole test side is the held-out set and the rule is not applied to it.

Every tile is resolved through tools/evaluation/gt_paths.resolve() rather than
by a name prefix -- the heuristic that was wrong in nine places and fixed on
2026-10-09 (Known Tool Traps).

Usage:  build_fill_eval_lists.py [--out-dir results/baseline_fill_20261009/lists]
"""

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gt_paths  # noqa: E402

SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
MODULUS = 5


def page_of(name):
    return name.split("_")[1]


def held_out(pool, page):
    digest = hashlib.sha1(f"{pool}_{page}".encode()).hexdigest()[:8]
    return int(digest, 16) % MODULUS == 0


def tiles_of(split, pool):
    d = SHARED / split / "line"
    return sorted(p.name for p in d.iterdir()
                  if p.suffix == ".jpg" and p.name.split("_")[0] == pool)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/baseline_fill_20261009/lists")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sets = {}
    sets["housei_test"] = ("test", tiles_of("test", "housei"))
    for pool in ("ako5", "ako5r"):
        all_tiles = tiles_of("train", pool)
        pages = sorted({page_of(n) for n in all_tiles})
        held_pages = [p for p in pages if held_out(pool, p)]
        sets[f"{pool}_held"] = ("train", [n for n in all_tiles
                                          if page_of(n) in held_pages])
        print(f"{pool}: {len(all_tiles)} tiles over {len(pages)} pages; "
              f"held pages {' '.join(held_pages)}")
    # the descriptive pass is over everything, scored by nothing
    sets["all_fill_pools"] = ("train", tiles_of("train", "ako5")
                              + tiles_of("train", "ako5r")
                              + tiles_of("train", "housei"))

    for name, (split, tiles) in sets.items():
        # verify every tile resolves on both sides before writing the list
        missing = []
        for t in tiles:
            for kind in ("line", "rough"):
                try:
                    gt_paths.resolve(t, kind=kind, split=split, root=SHARED)
                except FileNotFoundError:
                    missing.append(f"{t}:{kind}")
        if missing:
            raise SystemExit(f"{name}: {len(missing)} unresolved, e.g. {missing[:3]}")
        path = out / f"{name}.txt"
        path.write_text("".join(f"{t}\n" for t in tiles))
        pages = Counter(page_of(t) for t in tiles)
        print(f"wrote {path}  n={len(tiles)} split={split} pages={len(pages)}")

    (out / "SPLIT.md").write_text(
        "# Track J evaluation lists\n\n"
        f"Rule, registered 2026-10-09 before any image was read:\n\n"
        f"    held_out(pool, page) := int(sha1(f\"{{pool}}_{{page}}\")"
        f".hexdigest()[:8], 16) %% {MODULUS} == 0\n\n"
        "`housei` is already page-split (test = 001/004/009); the rule applies\n"
        "only to `ako5` and `ako5r`, which have no test side. `all_fill_pools`\n"
        "is for the model-free descriptive pass and is not a scoring set.\n"
    )
    print(f"wrote {out/'SPLIT.md'}")


if __name__ == "__main__":
    main()
