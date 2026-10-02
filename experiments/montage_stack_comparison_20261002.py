"""Side-by-side of the two stacks on the SAME tiles and the SAME conditioning
maps: this track's SD1.5 configuration against the SDXL configuration the
holdout-validation run used.

The point of this sheet is to make the central claim of 2026-10-02 checkable
by eye rather than from ink_ratio alone -- that the hatch wall is a property of
the SD1.5 stack, not of the conditioning scale, because a stack that is in the
line-art regime on these exact 192 tiles already exists in the project.

Not a like-for-like quality comparison: the SDXL run is at 1024 with an anime
base and a different public ControlNet. That difference is the finding.
"""

from pathlib import Path

from PIL import Image, ImageDraw

TRACK = Path(__file__).resolve().parents[1]
GT = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line")
COND = Path("/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning")
SDXL = Path("/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/outputs")
SD15_SMOKE = TRACK / "results/cs_sweep_smoke_20261002"
CELL, LABEL_H = 240, 22

COLS = [
    ("condition (lineart_coarse)", lambda t: (COND / t, True)),
    ("SD1.5 cs1.0 (this track)", lambda t: (SD15_SMOKE / "cs01.00" / f"{Path(t).stem}_out.png", False)),
    ("SD1.5 cs2.5 (this track)", lambda t: (SD15_SMOKE / "cs02.50" / f"{Path(t).stem}_out.png", False)),
    ("SD1.5 cs6.0 (this track)", lambda t: (SD15_SMOKE / "cs06.00" / f"{Path(t).stem}_out.png", False)),
    ("SDXL anime cs2.0", lambda t: (SDXL / "bare_cs2.0" / f"{Path(t).stem}_out.png", False)),
    ("SDXL anime cs3.0", lambda t: (SDXL / "bare_cs3.0" / f"{Path(t).stem}_out.png", False)),
    ("GT", lambda t: (GT / t, False)),
]


def main():
    tiles = [l.strip() for l in open("/home/sh1/deepl/lineart/dataset/pairs_480/holdout_lineart_family.txt") if l.strip()][::24]
    sheet = Image.new("L", (CELL * len(COLS), LABEL_H + len(tiles) * (CELL + LABEL_H)), 255)
    draw = ImageDraw.Draw(sheet)
    for c, (name, _) in enumerate(COLS):
        draw.text((c * CELL + 4, 5), name, fill=0)
    for r, tile in enumerate(tiles):
        y = LABEL_H + r * (CELL + LABEL_H)
        draw.text((4, y + CELL + 4), tile, fill=0)
        for c, (_, resolve) in enumerate(COLS):
            path, invert = resolve(tile)
            if not Path(path).exists():
                continue
            img = Image.open(path).convert("L").resize((CELL, CELL))
            if invert:
                img = Image.eval(img, lambda p: 255 - p)
            sheet.paste(img, (c * CELL, y))
    out = TRACK / "results/regime_diag_20261002/montage_stack_comparison.png"
    sheet.save(out)
    print(f"wrote {out}")
    print("columns:", " | ".join(n for n, _ in COLS))


if __name__ == "__main__":
    main()
