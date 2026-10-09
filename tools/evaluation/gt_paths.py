"""Resolve a tile name to its file under dataset/pairs_480, by looking.

Every caller of this used to decide the split with
`"train" if name.startswith("housei") else "test"`. That heuristic is wrong
for 168 of the 192 tiles in `holdout_lineart_family.txt` -- only the 24
`lineart_004_*` tiles actually live in `test` -- and it stayed hidden because
the default sample lists were all `lineart_004_*`. Deciding by whether the file
is there cannot drift as the pools are re-materialised, so that is what this
does.

Recorded in doc/CURRENT.md under Known Tool Traps; fixed 2026-10-09.
"""

from pathlib import Path

PAIRS_ROOT = Path("dataset/pairs_480")
SPLITS = ("train", "test")


def normalize_name(name):
    """Drop a trailing .jpg so lists with and without the suffix both work."""
    return name[:-4] if name.endswith(".jpg") else name


def resolve(name, kind="line", split="auto", root=PAIRS_ROOT):
    """Path to one tile. `split="auto"` searches train then test.

    Raises FileNotFoundError naming the tile rather than returning a path that
    does not exist, so a missing tile cannot be read as an empty image.
    """
    base = normalize_name(name)
    root = Path(root)
    if split != "auto":
        candidate = root / split / kind / f"{base}.jpg"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{base} ({kind}) not in {split}: {candidate}")
    for candidate_split in SPLITS:
        candidate = root / candidate_split / kind / f"{base}.jpg"
        if candidate.exists():
            return candidate
    searched = ", ".join(str(root / s / kind) for s in SPLITS)
    raise FileNotFoundError(f"{base} ({kind}) in none of: {searched}")


def split_of(name, kind="line", root=PAIRS_ROOT):
    """Which split a tile is in, by looking. Used where a caller wants the
    name rather than the path."""
    return resolve(name, kind=kind, root=root).parent.parent.name
