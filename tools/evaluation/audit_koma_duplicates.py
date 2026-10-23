"""Does the koma corpus hold the same page twice, and does any copy cross the split?

The foundation found (2026-10-23) that the corpus keeps the same page at more
than one drawing stage, grouped by page shape. Grouping by shape over-counts:
two unrelated works can share a canvas and a full-page panel box. This audit
asks the question on the content instead -- a panel is a copy of another when
**both** its line and its rough are byte-identical -- which is the form the
leakage question actually takes.

    audit_koma_duplicates.py              # the counts
    audit_koma_duplicates.py --self-test  # assert the registered 2026-10-23 values

What matters for the stage-2 verdict is the last line: no duplicate group has
one copy in train and another in held. The family split is what buys most of
that (a work_id-level split would have leaked), but not all of it -- one
duplicate pair spans two families and landed train/train, which is luck, not
design. See doc/work_log.md 2026-10-23 "The corpus holds the same page twice".
"""

import argparse
import collections
import csv
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from koma_split import ROOT, family, held_out  # noqa: E402

REGISTERED = {"groups": 201, "panels": 412, "multi_family": 3,
              "cross_split": 0, "held_groups": 8, "held_panels": 19}


def digest(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def duplicate_groups(root=ROOT):
    """[[manifest row, ...], ...] for every set of byte-identical (line, rough)."""
    by_content = collections.defaultdict(list)
    with open(Path(root) / "manifest.csv") as fh:
        for r in csv.DictReader(fh):
            key = (digest(Path(root) / "line" / r["name"]),
                   digest(Path(root) / "rough" / r["name"]))
            by_content[key].append(r)
    return [v for v in by_content.values() if len(v) > 1]


def audit(root=ROOT):
    groups = duplicate_groups(root)
    out = {"groups": len(groups), "panels": sum(len(g) for g in groups),
           "multi_family": 0, "cross_split": 0, "held_groups": 0,
           "held_panels": 0, "crossing": []}
    for g in groups:
        fams = {family(r["work_id"]) for r in g}
        sides = {"held" if held_out(f) else "train" for f in fams}
        out["multi_family"] += len(fams) > 1
        if len(sides) > 1:
            out["cross_split"] += 1
            out["crossing"].append(sorted(r["work_id"] for r in g))
        elif sides == {"held"}:
            out["held_groups"] += 1
            out["held_panels"] += len(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    a = audit()
    print(f"byte-identical (line AND rough) duplicate groups: {a['groups']}")
    print(f"  panels inside them                            : {a['panels']}")
    print(f"  groups spanning more than one family          : {a['multi_family']}")
    print(f"  groups entirely inside held                   : {a['held_groups']}"
          f"  ({a['held_panels']} panels)")
    print(f"  groups spanning TRAIN and HELD                : {a['cross_split']}"
          f"   <- must be 0")
    for w in a["crossing"]:
        print(f"    LEAK: {w}")
    if args.self_test:
        bad = [k for k, v in REGISTERED.items() if a[k] != v]
        for k in bad:
            print(f"SELF-TEST FAIL {k}: {a[k]} != registered {v}")
        print("SELF-TEST", "FAIL" if bad else "PASS")
        return 1 if bad else 0
    return 1 if a["cross_split"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
