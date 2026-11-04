"""Amendment 1 of doc/prereg_fill_windows_20261104.md.

normalize.paper_level calls the image's mode the paper. In a window where
fill is the mode it returns 0, and equalize_white_point then does nothing,
so the paper is left unequalized. For the two new arms only: when the mode is
below 128, the paper is the mode among pixels >= 128. Same formula
v' = clip(round(v * 255 / m)). Rewrites norm_<arm>/ from the raw JPEGs and
prints how many tiles the amendment changed.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import normalize


def paper_amended(gray):
    m = normalize.paper_level(gray)
    if m < 128:
        h = np.bincount(gray.ravel(), minlength=256)
        m = int(h[128:].argmax()) + 128
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/author_fill_windows_20261104")
    args = ap.parse_args()
    for arm in ("f", "l"):
        src, dst = Path(args.root) / f"tiles_{arm}", Path(args.root) / f"norm_{arm}"
        changed, papers = 0, []
        for p in sorted(src.glob("*.jpg")):
            g = np.asarray(Image.open(p).convert("L"))
            m0, m = normalize.paper_level(g), paper_amended(g)
            if m != m0:
                changed += 1
                papers.append(m)
            out = np.clip(np.rint(g.astype(np.float32) * (255.0 / m)), 0, 255).astype(np.uint8)
            Image.fromarray(out).save(dst / f"{p.stem}.png")
        extra = f"  紙(修正後) min {min(papers)} 中央 {int(np.median(papers))} max {max(papers)}" if papers else ""
        print(f"{arm}: 修正で紙が変わったタイル {changed}{extra}")


if __name__ == "__main__":
    main()
