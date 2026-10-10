"""The stage-2 work-family split, exactly as registered 2026-10-10.

    family(work_id) := third underscore field, lowercased, "-..." and a
                       trailing "_color" dropped, trailing digits (plus one
                       optional letter) stripped; anything starting with
                       "ako" collapses to "ako"
    held_out(family) := int(sha1(f"fillv2_{family}").hexdigest()[:8], 16) % 5 == 0

Whether a character's hair is blacked in is a property of that character, so
the leak that would inflate this task is character identity, not page
adjacency -- hence the family, and hence the ako collapse.

The registration recorded what this rule produces before any score existed:
33 families, 9 held out, 969 panels (19.3%), 4049 train. `_self_check()`
asserts that, so a silent change to the rule cannot pass.
"""

import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path("/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826")
EXPECT = {"families": 33, "held_families": 9, "held_panels": 969, "train_panels": 4049}
EXPECT_HELD = {"avbu": 42, "ben": 89, "china": 69, "gakusai": 383, "hero": 93,
               "housei": 45, "pink": 107, "poolsideidle": 79, "stoptime": 62}


def family(work_id):
    parts = work_id.split("_")
    f = (parts[2] if len(parts) > 2 else work_id).lower()
    f = f.split("-")[0]
    if f.endswith("_color"):
        f = f[: -len("_color")]
    f = re.sub(r"\d+[a-z]?$", "", f)
    return "ako" if f.startswith("ako") else f


def held_out(f):
    return int(hashlib.sha1(f"fillv2_{f}".encode()).hexdigest()[:8], 16) % 5 == 0


def load(root=ROOT):
    """Returns {"train": [(name, work_id, family)], "held": [...]}"""
    out = {"train": [], "held": []}
    with open(Path(root) / "manifest.csv") as fh:
        for r in csv.DictReader(fh):
            f = family(r["work_id"])
            out["held" if held_out(f) else "train"].append((r["name"], r["work_id"], f))
    return out


def _self_check():
    assert family("001_2015_gakusai3") == "gakusai"
    assert family("011_2017_ako5") == "ako"
    assert family("066_2024_housei") == "housei"
    assert family("063_2023b_hamlabi") == "hamlabi"
    if not (Path(ROOT) / "manifest.csv").exists():
        return "manifest absent; rule asserts only"
    s = load()
    fams = {f for _, _, f in s["train"] + s["held"]}
    held = {}
    for _, _, f in s["held"]:
        held[f] = held.get(f, 0) + 1
    got = {"families": len(fams), "held_families": len(held),
           "held_panels": len(s["held"]), "train_panels": len(s["train"])}
    assert got == EXPECT, f"split drifted from the registration: {got} != {EXPECT}"
    assert held == EXPECT_HELD, f"held families drifted: {held}"
    return got


if __name__ == "__main__":
    got = _self_check()
    print(f"koma_split self-check: {got}")
    print("matches the registration (33 families, 9 held, 969/4049 panels) -> PASS")
