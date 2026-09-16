#!/usr/bin/env python
"""Generate negative stroke candidates for Track F gate 1.

For a GT drawing, hold out one token and ask: does this candidate belong here?
Positive = the token that was really there. Negatives are built from GT only --
the preprocessor's skeleton is not tokenizable (2026-09-17), so it supplies no
negatives until gate 2.

  displaced : the true token shifted 5-12px
  rotated   : the true token turned 15-40 degrees about its centroid
  foreign   : a similar-length token from a different drawing, centred there

A displaced or foreign stroke can land on ANOTHER real stroke, which makes it a
false negative. `on_other` reports that per candidate; filter on it.
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from stroke_churn import dist_to, ink_skeleton, junction_mask, load_gray  # noqa: E402

TOL = 3.0


def comps(sk, min_len):
    jd = cv2.dilate(junction_mask(sk).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    n, lab = cv2.connectedComponents((sk & ~jd).astype(np.uint8), connectivity=8)
    f = lab.ravel()
    order = np.argsort(f, kind="stable")
    fs = f[order]
    s = np.searchsorted(fs, np.arange(1, n), "left")
    e = np.searchsorted(fs, np.arange(1, n), "right")
    return [np.stack(np.unravel_index(order[a:b], sk.shape), 1) for a, b in zip(s, e) if b - a >= min_len]


def place(pts, shape):
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < shape[0]) & (pts[:, 1] >= 0) & (pts[:, 1] < shape[1])
    return pts[ok]


def displace(pts, rng, shape):
    ang = rng.uniform(0, 2 * np.pi)
    d = rng.uniform(5, 12)
    off = np.array([np.sin(ang) * d, np.cos(ang) * d])
    return place(np.round(pts + off).astype(int), shape)


def rotate(pts, rng, shape):
    th = np.deg2rad(rng.uniform(15, 40) * rng.choice([-1, 1]))
    c = pts.mean(0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return place(np.round((pts - c) @ R.T + c).astype(int), shape)


def foreign(pool, pts, rng, shape):
    n = len(pts)
    near = [p for p in pool if 0.6 * n <= len(p) <= 1.6 * n]
    if not near:
        return None
    src = near[rng.integers(len(near))]
    return place(np.round(src - src.mean(0) + pts.mean(0)).astype(int), shape)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = "/home/sh1/deepl/lineart-controlnet-sd15-refine/data"
    p.add_argument("--line-dir", default=f"{data}/line")
    p.add_argument("--list", default=f"{data}/train_list.txt")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--min-token-px", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--output-dir", default="results/negatives_check_20260917")
    p.add_argument("--montage-rows", type=int, default=6)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    names = [l.strip() for l in open(a.list) if l.strip()][: a.limit]
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tiles = {}
    for n in names:
        sk = ink_skeleton(load_gray(Path(a.line_dir) / n))
        if sk.any():
            tiles[n] = (sk, comps(sk, 8))
    pool = [p for _, (_, cs) in tiles.items() for p in cs if len(p) >= a.min_token_px]
    print(f"tiles {len(tiles)}   pool tokens (>= {a.min_token_px}px) {len(pool)}")

    rows, shots = [], []
    for n, (sk, cs) in tiles.items():
        big = [c for c in cs if len(c) >= a.min_token_px]
        for c in big:
            rest = np.zeros_like(sk)
            for o in cs:
                if o is not c:
                    rest[o[:, 0], o[:, 1]] = True
            drest = dist_to(rest)
            cands = {"true": c, "displaced": displace(c, rng, sk.shape),
                     "rotated": rotate(c, rng, sk.shape), "foreign": foreign(pool, c, rng, sk.shape)}
            for kind, pts in cands.items():
                if pts is None or len(pts) < 5:
                    continue
                d = drest[pts[:, 0], pts[:, 1]]
                rows.append({"tile": n, "kind": kind, "n_px": len(pts),
                             "on_other": round(float((d <= TOL).mean()), 4),
                             "med_dist_to_rest": round(float(np.median(d)), 2)})
            if len(shots) < a.montage_rows and len(c) >= 40 and all(cands[k] is not None for k in cands):
                shots.append((n, rest, cands))

    with open(out / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["tile", "kind", "n_px", "on_other", "med_dist_to_rest"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'kind':<12}{'n':>8}{'on_other>0.5':>14}{'on_other mean':>15}{'med dist':>10}")
    for k in ("true", "displaced", "rotated", "foreign"):
        r = [x for x in rows if x["kind"] == k]
        if not r:
            continue
        oo = np.array([x["on_other"] for x in r])
        print(f"{k:<12}{len(r):>8}{(oo > 0.5).mean():>14.3f}{oo.mean():>15.3f}"
              f"{np.mean([x['med_dist_to_rest'] for x in r]):>10.2f}")

    from PIL import Image
    W, K = 300, ["true", "displaced", "rotated", "foreign"]
    sheet = Image.new("RGB", (W * len(K), W * len(shots)), "white")
    for r, (n, rest, cands) in enumerate(shots):
        for ci, k in enumerate(K):
            canvas = np.full((480, 480, 3), 255, np.uint8)
            canvas[rest] = (205, 205, 205)
            pts = cands[k]
            col = (30, 130, 60) if k == "true" else (200, 30, 40)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = place(pts + [dy, dx], (480, 480))
                    canvas[q[:, 0], q[:, 1]] = col
            sheet.paste(Image.fromarray(canvas).resize((W, W), Image.NEAREST), (ci * W, r * W))
    sheet.save(out / "negatives_montage.png")
    print(f"\nmontage: {out/'negatives_montage.png'}")
    print("columns: " + " | ".join(K) + "   (grey = the rest of the drawing, green = real, red = fake)")
    print("rows: " + ", ".join(s[0] for s in shots))


if __name__ == "__main__":
    main()
