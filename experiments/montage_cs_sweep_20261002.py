"""Contact sheet for a cs sweep: conditioning | each cs in numeric order | GT.

Columns are in ascending cs so the two ends of the sweep read left to right.
Tiles are taken in list order from whatever the sweep actually wrote, not
cherry-picked. Every panel is drawn at GT polarity (black ink on white paper),
including the conditioning map, which is stored inverted.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

TRACK = Path(__file__).resolve().parents[1]
GT_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line"
)
COND_DIR = Path(
    "/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning"
)
CELL = 240
LABEL_H = 22


def load_cell(path, invert=False):
    img = Image.open(path).convert("L").resize((CELL, CELL))
    if invert:
        img = Image.eval(img, lambda p: 255 - p)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default="results/cs_sweep_20261002")
    parser.add_argument("--tiles", default="", help="comma-separated tile names; default = all tiles present")
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.sweep_root)
    if not root.is_absolute():
        root = TRACK / root
    arms = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))
    if not arms:
        raise SystemExit(f"no arm directories under {root}")

    if args.tiles:
        stems = [Path(t).stem for t in args.tiles.split(",") if t.strip()]
    else:
        stems = sorted(p.name[: -len("_out.png")] for p in (root / arms[0]).glob("*_out.png"))
    stems = stems[: args.max_rows]

    cols = ["condition"] + arms + ["GT"]
    sheet = Image.new("L", (CELL * len(cols), LABEL_H + len(stems) * (CELL + LABEL_H)), 255)
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        draw.text((c * CELL + 4, 5), name[:34], fill=0)

    for r, stem in enumerate(stems):
        y = LABEL_H + r * (CELL + LABEL_H)
        tile = f"{stem}.jpg"
        draw.text((4, y + CELL + 4), tile, fill=0)
        for c, name in enumerate(cols):
            if name == "condition":
                cell = load_cell(COND_DIR / tile, invert=True)
            elif name == "GT":
                cell = load_cell(GT_DIR / tile)
            else:
                p = root / name / f"{stem}_out.png"
                if not p.exists():
                    continue
                cell = load_cell(p)
            sheet.paste(cell, (c * CELL, y))

    out = Path(args.out) if args.out else root / "montage_cs_sweep.png"
    sheet.save(out)
    print(f"wrote {out}")
    print("columns:", " | ".join(cols))


if __name__ == "__main__":
    main()
