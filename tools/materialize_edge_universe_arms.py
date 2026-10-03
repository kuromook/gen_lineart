"""Materialize the arms Track C's deletion actually lives in.

Track C's classifier is a keep-mask over `edge_map(conditioning)` -- the Canny
contour map of the rough -- so its output universe is that edge map, not the
conditioning's own ink. The signal table of 2026-10-03 compared it against
`CONDITION`, which measure_f1_signal.py renders as the *inverted raw*
conditioning. Those two arms differ in rendering as well as in content, so the
+0.0272 they bracket is not a before/after of the deletion.

This writes the structurally correct baseline (`keep_all` = the whole edge map,
nothing deleted) plus a shuffled-edge degenerate arm for the lesson-9 protocol
check, in the <stem>_out.png black-on-white convention.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
sys.path.insert(0, "/home/sh1/deepl/lineart-stroke-selection/scripts")
from tile_region_manifest_480 import edge_map  # noqa: E402
from train_stroke_selection import load_gray01  # noqa: E402

COND = Path("/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912/conditioning")
TILES = Path("/home/sh1/deepl/lineart/dataset/pairs_480/holdout_lineart_family.txt")
OUT = Path(__file__).resolve().parents[1] / "results/trackc_judge_20261004/arms"

tiles = [l.strip() for l in open(TILES) if l.strip()]
(OUT / "keep_all").mkdir(parents=True, exist_ok=True)
(OUT / "shuffled_edge").mkdir(parents=True, exist_ok=True)

edges = {}
for tile in tiles:
    ce = edge_map((load_gray01(COND / tile) * 255).astype(np.uint8))
    edges[tile] = ce
    Image.fromarray(np.where(ce, 0, 255).astype(np.uint8)).save(
        OUT / "keep_all" / f"{Path(tile).stem}_out.png")

# degenerate arm: each tile gets another source image's edge map, so any signal
# it shows is the floor leaking through rather than correspondence.
rng = np.random.default_rng(20261004)
others = list(tiles)
rng.shuffle(others)
for tile, donor in zip(tiles, others):
    if donor.split("_")[1] == tile.split("_")[1]:
        donor = tiles[(tiles.index(tile) + 97) % len(tiles)]
    Image.fromarray(np.where(edges[donor], 0, 255).astype(np.uint8)).save(
        OUT / "shuffled_edge" / f"{Path(tile).stem}_out.png")

print(f"keep_all: {len(list((OUT/'keep_all').glob('*.png')))}")
print(f"shuffled_edge: {len(list((OUT/'shuffled_edge').glob('*.png')))}")
