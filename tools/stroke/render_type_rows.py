#!/usr/bin/env python
"""Cut panel-composition cluster rows out of montage_k12.png for the naming UI.

Outputs results/panel_composition_20260921/rows/row_{rank:02d}.png and types.json
(display order = size-sorted, same as report_k12.txt and montage rows).
"""
import json
import re
from pathlib import Path

from PIL import Image

D = Path("results/panel_composition_20260921")
CELL, PAD, COLS = 200, 4, 8


def main():
    img = Image.open(D / "montage_k12.png")
    report = (D / "report_k12.txt").read_text()
    blocks = re.split(r"\n== ", report)[1:]
    types = []
    for rank, b in enumerate(blocks):
        m = re.match(r"cluster (\d+) \(rank (\d+)\) n=(\d+) \(([\d.]+)%\)", b)
        tags_m = re.search(r"top member tags: (.+)", b)
        med_m = re.search(r"median: (.+)", b)
        cid, rnk, n, pct = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
        y0 = PAD + rank * (CELL + PAD)
        img.crop((0, y0, img.size[0], y0 + CELL)).save(D / "rows" / f"row_{rank:02d}.png")
        types.append({
            "rank": rank, "cluster": cid, "n": n, "pct": pct,
            "median": med_m.group(1) if med_m else "",
            "tags": tags_m.group(1) if tags_m else "",
            "img": f"rows/row_{rank:02d}.png",
        })
    (D / "types.json").write_text(json.dumps(types, ensure_ascii=False, indent=1))
    print(f"{len(types)} rows -> {D}/rows")


if __name__ == "__main__":
    (D / "rows").mkdir(parents=True, exist_ok=True)
    main()
