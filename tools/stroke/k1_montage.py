#!/usr/bin/env python
"""K1 の目視: 印の付いた線と、印の無い線を、ベタマスクの上に同じ描き方で並べる。

定量(`k1_broken_vs_fill.py`)と同じ `per_stroke.csv` から引くので、絵と数は必ず一致する。
上段 = 印あり(無作為)、下段 = 印なし(無作為、NC-k1 と同じ母集団)。並べ方を変えない。
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, "/home/sh1/deepl/lineart-face-words/tools/face")
sys.path.insert(0, "/home/sh1/deepl/lineart-solid-fill/tools/evaluation")
import group_server as G                                            # noqa: E402
from fill_mask import fill_mask, ink_of, FILL_HALF_WIDTH_PX         # noqa: E402

TILE, COLS, SEED = 300, 6, 20261016
PAD_FRAC, MIN_SIDE = 0.35, 200


def tile_of(pid, poly, note, cache):
    if pid not in cache:
        bgr = G.panel_img(pid)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        cache.clear()
        cache[pid] = (gray, fill_mask(ink_of(gray), radius=FILL_HALF_WIDTH_PX))
    gray, fill = cache[pid]
    H, W = gray.shape
    x0, y0 = poly[:, 0].min(), poly[:, 1].min()
    x1, y1 = poly[:, 0].max(), poly[:, 1].max()
    side = max(max(x1 - x0, y1 - y0) * (1 + 2 * PAD_FRAC), MIN_SIDE)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax, ay = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = int(round(side))
    canvas = np.full((s, s), 236, np.uint8)
    fcrop = np.zeros((s, s), bool)
    sx0, sy0 = max(ax, 0), max(ay, 0)
    sx1, sy1 = min(ax + s, W), min(ay + s, H)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - ay:sy1 - ay, sx0 - ax:sx1 - ax] = gray[sy0:sy1, sx0:sx1]
        fcrop[sy0 - ay:sy1 - ay, sx0 - ax:sx1 - ax] = fill[sy0:sy1, sx0:sx1]
    out = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    out[fcrop] = (0.45 * out[fcrop] + 0.55 * np.array([60, 60, 235])).astype(np.uint8)
    p = np.round((poly - [ax, ay])).astype(np.int32)
    cv2.polylines(out, [p], False, (20, 160, 20), max(int(round(s / 90)), 2), cv2.LINE_AA)
    out = cv2.resize(out, (TILE, TILE), interpolation=cv2.INTER_AREA if s > TILE else cv2.INTER_LINEAR)
    cv2.rectangle(out, (0, 0), (TILE - 1, TILE - 1), (170, 170, 170), 1)
    cv2.putText(out, note, (5, TILE - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def band(rows, title, cache, n):
    tiles = []
    for r in rows[:n]:
        pl = G.face_payload(int(r["order"]))
        s = next(t for t in pl["strokes"] if t["id"] == int(r["sid"]))
        poly = np.asarray(s["p"], np.float64).reshape(16, 2)
        tiles.append(tile_of(int(r["panel"]), poly,
                             f'o{r["order"]} f4={r["f4"]} L={float(r["length"]):.0f}', cache))
    while len(tiles) % COLS:
        tiles.append(np.full((TILE, TILE, 3), 255, np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
    head = np.full((34, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([head, grid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--dir", default="results/k1_broken_vs_fill_20261016")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    d = ROOT / a.dir
    G.load_state(Path(a.labels))
    rows = list(csv.DictReader(open(d / "per_stroke.csv")))
    rng = np.random.default_rng(SEED)
    mk = [r for r in rows if r["marked"] == "1"]
    um = [r for r in rows if r["marked"] == "0"]
    mk = [mk[i] for i in rng.choice(len(mk), min(a.n, len(mk)), replace=False)]
    um = [um[i] for i in rng.choice(len(um), min(a.n, len(um)), replace=False)]
    mk.sort(key=lambda r: int(r["panel"])); um.sort(key=lambda r: int(r["panel"]))
    cache = {}
    top = band(mk, f'marked (broken) n={len(mk)} / random, seed {SEED} / fill mask (radius 4) in red, the stroke in green',
               cache, a.n)
    bot = band(um, f'unmarked n={len(um)} / random from the same faces (NC-k1 population) / same drawing',
               cache, a.n)
    w = max(top.shape[1], bot.shape[1])
    img = np.vstack([np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0)), constant_values=255) for x in (top, bot)])
    out = d / "montage_marked_vs_unmarked.png"
    cv2.imwrite(str(out), img)
    print(out, img.shape)


if __name__ == "__main__":
    main()
