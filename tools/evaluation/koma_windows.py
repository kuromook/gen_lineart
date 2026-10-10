"""Cut a koma panel into 480px windows and put it back.

Registered 2026-10-10 in doc/work_log.md. The window is an implementation
detail of the network; the unit of judgement is the panel, so everything the
model produces has to come back to panel coordinates before it is scored.

  size 480, stride 480 - OVERLAP, OVERLAP = 64
  the last row and column are CLAMPED to the panel edge, not padded, so a
  window always contains real image
  a panel under 480 in a dimension is padded with background white and the
  pad is removed before scoring  (shortcut, see doc/shortcuts.md)

  stitch(reduce="overwrite")  reproduces the input exactly -- IC4 runs on this
  stitch(reduce="mean")       averages overlapping predictions, then 0.5

IC4, the registered stop rule: cut a panel, reassemble, compare against the
panel PNG **as it sits on disk** -- not against our own tiling. Bit-identical
or nothing is trained. An off-by-one in the grid is invisible in every metric.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SIZE = 480
OVERLAP = 64
BG = 255           # background white, the pad value


def grid(h, w, size=SIZE, overlap=OVERLAP):
    """Top-left corners covering (h, w). Clamped at the far edge, so the last
    window overlaps its neighbour by more than `overlap` rather than running
    past the panel."""
    stride = size - overlap
    assert stride > 0, "overlap must be smaller than the window"

    def starts(n):
        if n <= size:
            return [0]
        out = list(range(0, n - size + 1, stride))
        if out[-1] != n - size:
            out.append(n - size)
        return out
    return [(y, x) for y in starts(h) for x in starts(w)]


def pad_to(img, size=SIZE, value=BG):
    """Pad up to at least `size` in both dimensions. Returns (padded, (h, w))
    of the original, so the pad can be removed exactly."""
    h, w = img.shape[:2]
    ph, pw = max(0, size - h), max(0, size - w)
    if not (ph or pw):
        return img, (h, w)
    out = np.full((h + ph, w + pw) + img.shape[2:], value, dtype=img.dtype)
    out[:h, :w] = img
    return out, (h, w)


def cut(img, size=SIZE, overlap=OVERLAP):
    """Returns (windows, coords, original_shape). `img` may be 2-D or 3-D."""
    padded, orig = pad_to(img, size)
    h, w = padded.shape[:2]
    coords = grid(h, w, size, overlap)
    return [padded[y:y + size, x:x + size].copy() for (y, x) in coords], coords, orig


def stitch(windows, coords, orig, size=SIZE, reduce="mean", dtype=None):
    """Put windows back into panel coordinates and drop the pad.

    reduce="overwrite" is exact reassembly and is what IC4 checks.
    reduce="mean" averages the overlaps, then the caller thresholds.
    """
    h0, w0 = orig
    H = max(h0, max((y for y, _ in coords), default=0) + size)
    W = max(w0, max((x for _, x in coords), default=0) + size)
    tail = windows[0].shape[2:]
    if reduce == "overwrite":
        out = np.zeros((H, W) + tail, dtype=dtype or windows[0].dtype)
        for win, (y, x) in zip(windows, coords):
            out[y:y + size, x:x + size] = win
        return out[:h0, :w0]
    acc = np.zeros((H, W) + tail, dtype=np.float64)
    cnt = np.zeros((H, W) + (1,) * len(tail), dtype=np.float64)
    for win, (y, x) in zip(windows, coords):
        acc[y:y + size, x:x + size] += win
        cnt[y:y + size, x:x + size] += 1
    out = acc / np.maximum(cnt, 1)
    return out[:h0, :w0]


def _self_check():
    """One assert cell with a hand-computed value (foundation notice 4)."""
    # 1000x700 at size 480, stride 416: starts(1000) = 0,416,520 and
    # starts(700) = 0,220 -> 3 x 2 = 6 windows, written down before running.
    g = grid(1000, 700)
    assert g == [(0, 0), (0, 220), (416, 0), (416, 220), (520, 0), (520, 220)], g
    assert grid(480, 480) == [(0, 0)]
    assert grid(300, 300) == [(0, 0)]          # smaller than one window
    rng = np.random.default_rng(0)
    for shape in ((1000, 700), (300, 1200), (481, 479), (2000, 1500)):
        a = rng.integers(0, 256, shape, dtype=np.uint8)
        wins, coords, orig = cut(a)
        b = stitch(wins, coords, orig, reduce="overwrite")
        assert b.shape == a.shape and np.array_equal(a, b), shape
    # a constant field must survive the mean reduction exactly
    a = np.full((900, 900), 0.25)
    wins, coords, orig = cut(a)
    b = stitch(wins, coords, orig, reduce="mean")
    assert np.allclose(b, 0.25), b.min()
    return True


_self_check()


def ic4(paths, verbose=True):
    """IC4: reassembly must be bit-identical to the PNG on disk."""
    ok = 0
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  UNREADABLE {p}")
            continue
        wins, coords, orig = cut(img)
        back = stitch(wins, coords, orig, reduce="overwrite")
        same = back.shape == img.shape and np.array_equal(back, img)
        ok += same
        if verbose and not same:
            print(f"  MISMATCH {p.name}  {img.shape} -> {back.shape}")
    print(f"IC4 stitching: {ok}/{len(paths)} panels bit-identical to the PNG "
          f"on disk -> {'PASS' if ok == len(paths) else 'FAIL'}")
    return ok == len(paths)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="panel dir with line/ and rough/")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    paths = []
    for which in ("line", "rough"):
        d = Path(args.root) / which
        if d.is_dir():
            paths += sorted(p for p in d.iterdir()
                            if p.suffix.lower() in (".png", ".jpg"))[:args.limit]
    print(f"self-check: PASS (grid and stitch assert cells)")
    sizes = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED).shape[:2] for p in paths[:args.limit]]
    small = sum(1 for h, w in sizes if min(h, w) < SIZE)
    print(f"{len(paths)} panels; {small} are under {SIZE}px on a side and are padded "
          f"(shortcut, doc/shortcuts.md)")
    sys.exit(0 if ic4(paths) else 1)


if __name__ == "__main__":
    main()
