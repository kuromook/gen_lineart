"""One sheet for the tone question: what was handed over as the style exemplar,
and what came back, at both conditioning scales.

The per-cs contact sheets do not show the reference image, which for a
style-transfer arm is the one column that makes the result readable. This puts
it beside the outputs, and puts cs1.0 and cs2.5 on the same row so the
"residuals at 2.5x drown the adapter" reading can be checked by eye rather than
only in near_white_frac.

Rows are the first tile of each of the eight source images, so the sheet spans
the group instead of one picture.
"""

from pathlib import Path

from PIL import Image, ImageDraw

TRACK = Path(__file__).resolve().parents[1]
GT = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line")
COND = Path(
    "/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning"
)
ROOT = TRACK / "results/ipadapter_tone_matched_20261002"
TILES = "/home/sh1/deepl/lineart/dataset/pairs_480/holdout_lineart_family.txt"
CELL, LABEL_H = 230, 20


def source_of(tile):
    return tile.split("_")[1] if "_" in tile else tile


def main():
    tiles = [l.strip() for l in open(TILES) if l.strip()]
    by_source = {}
    for t in tiles:
        by_source.setdefault(source_of(t), []).append(t)
    sources = sorted(by_source)
    # Same pairing the probe used: tile -> first tile of the NEXT source image.
    def ref_of(tile):
        here = source_of(tile)
        other = sources[(sources.index(here) + 1) % len(sources)]
        return tile if other == here else by_source[other][0]

    rows = [by_source[s][0] for s in sources]

    cols = [
        ("condition", lambda t: (COND / t, True)),
        ("REFERENCE handed over", lambda t: (GT / ref_of(t), False)),
        ("cs1.0 baseline", lambda t: (ROOT / "cs1.0/baseline" / f"{Path(t).stem}_out.png", False)),
        ("cs1.0 ip0.4", lambda t: (ROOT / "cs1.0/gt_otherfam_s0.4" / f"{Path(t).stem}_out.png", False)),
        ("cs1.0 ip0.8", lambda t: (ROOT / "cs1.0/gt_otherfam_s0.8" / f"{Path(t).stem}_out.png", False)),
        ("cs1.0 ip1.0", lambda t: (ROOT / "cs1.0/gt_otherfam_s1.0" / f"{Path(t).stem}_out.png", False)),
        ("cs2.5 baseline", lambda t: (ROOT / "cs2.5/baseline" / f"{Path(t).stem}_out.png", False)),
        ("cs2.5 ip1.0", lambda t: (ROOT / "cs2.5/gt_otherfam_s1.0" / f"{Path(t).stem}_out.png", False)),
        ("GT (target)", lambda t: (GT / t, False)),
    ]

    sheet = Image.new("L", (CELL * len(cols), LABEL_H + len(rows) * (CELL + LABEL_H)), 255)
    draw = ImageDraw.Draw(sheet)
    for c, (name, _) in enumerate(cols):
        draw.text((c * CELL + 4, 4), name, fill=0)
    for r, tile in enumerate(rows):
        y = LABEL_H + r * (CELL + LABEL_H)
        draw.text((4, y + CELL + 3), f"{tile}   reference: {ref_of(tile)}", fill=0)
        for c, (_, resolve) in enumerate(cols):
            path, invert = resolve(tile)
            if not Path(path).exists():
                continue
            img = Image.open(path).convert("L").resize((CELL, CELL))
            if invert:
                img = Image.eval(img, lambda p: 255 - p)
            sheet.paste(img, (c * CELL, y))

    out = ROOT / "montage_tone_reference_and_output.png"
    sheet.save(out)
    print(f"wrote {out}")
    print("columns:", " | ".join(n for n, _ in cols))


if __name__ == "__main__":
    main()
