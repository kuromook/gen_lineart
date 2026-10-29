"""Apply the registered normalization to one pool and write it out as PNG.

Both sides arrive here as a quality-95 JPEG and leave as PNG, so the profile
instrument reads each side after exactly one JPEG pass plus one white-point
pass -- no side sees the operations in a different order, and nothing is
re-encoded a second time.

Selection is a measurement: the ink gate is counted here too, even though
the author's pool defines the gate and cannot lose a tile to it.
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image

import normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--glob", default="*.jpg")
    args = parser.parse_args()

    paths = sorted(Path(args.src).glob(args.glob))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dropped = collections.Counter()
    papers = []
    written = 0
    for path in paths:
        gray = np.asarray(Image.open(path).convert("L"))
        if normalize.ink_ratio(gray) < normalize.INK_GATE:
            dropped["below_ink_gate"] += 1
            continue
        equalized, paper = normalize.equalize_white_point(gray)
        papers.append(paper)
        Image.fromarray(equalized).save(out_dir / f"{path.stem}.png")
        written += 1

    print(f"read: {len(paths)}  written: {written}")
    print("dropped, by reason:")
    for reason in sorted(dropped) or []:
        print(f"  {reason}: {dropped[reason]}")
    if not dropped:
        print("  (none)")
    if papers:
        papers = np.array(papers)
        print(f"paper level: min={papers.min()} median={int(np.median(papers))} max={papers.max()}")
    report = out_dir.parent / f"normalize_report_{out_dir.name}.json"
    report.write_text(json.dumps({"read": len(paths), "written": written,
                                  "dropped": dict(dropped),
                                  "ink_gate": normalize.INK_GATE}, indent=1))
    print(f"saved: {report}")


if __name__ == "__main__":
    main()
