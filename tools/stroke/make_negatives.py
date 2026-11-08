#!/usr/bin/env python
"""Negative stroke candidates for Track F gate 1.

Hold out one real token; ask whether a candidate belongs where it was.
Negatives come from GT only -- the preprocessor's skeleton does not tokenize
(2026-09-17), so it supplies none until gate 2.

  displaced : the true stroke shifted 5-12px
  rotated   : the true stroke turned 15-40 degrees about its centroid
  foreign   : a similar-length stroke from another drawing, centred there

Three rules exist because the first version leaked (2026-09-17):

1. WIDTH TRAVELS WITH THE STROKE. Measuring a candidate's width from the
   drawing's own ink gives 4px for a real stroke and 0 for every fake, since a
   fake lies on blank paper -- a perfect giveaway with nothing to do with
   placement. Width is an intrinsic attribute, sampled where the stroke really
   is, and carried along when it moves.
2. TRANSFORMED STROKES ARE RE-RASTERIZED. Rotating integer pixels scatters them,
   breaking 8-connectivity, so path-order features (straightness, curvature)
   collapsed for rotated candidates only -- another giveaway. Rotate the ordered
   path, then draw it as a polyline.
3. CANDIDATES THAT LEAVE THE FRAME ARE REJECTED, not clipped, so length stays
   matched across kinds.
"""
import numpy as np
import cv2


def comps(sk, min_len):
    """Junction-split components as (N,2) pixel arrays. One sort, no per-label scan."""
    from stroke_churn import junction_mask
    jd = cv2.dilate(junction_mask(sk).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    n, lab = cv2.connectedComponents((sk & ~jd).astype(np.uint8), connectivity=8)
    f = lab.ravel()
    order = np.argsort(f, kind="stable")
    fs = f[order]
    s = np.searchsorted(fs, np.arange(1, n), "left")
    e = np.searchsorted(fs, np.arange(1, n), "right")
    return [np.stack(np.unravel_index(order[a:b], sk.shape), 1) for a, b in zip(s, e) if b - a >= min_len]


def order_path(pts):
    """Walk from one extreme end, nearest neighbour. Shorter than pts if branchy."""
    d0 = np.abs(pts - pts[0]).max(1)
    start = tuple(int(v) for v in pts[int(d0.argmax())])
    todo = {(int(r), int(c)) for r, c in pts}
    todo.discard(start)
    out, cur = [start], start
    while todo:
        y, x = cur
        nxt = next((p for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and (p := (y + dy, x + dx)) in todo), None)
        if nxt is None:
            break
        todo.discard(nxt)
        out.append(nxt)
        cur = nxt
    return np.array(out)


def inside(pts, shape):
    return bool((pts[:, 0] >= 0).all() and (pts[:, 0] < shape[0]).all()
                and (pts[:, 1] >= 0).all() and (pts[:, 1] < shape[1]).all())


def place(pts, shape):
    """Clip to frame. Kept for callers that render; never used to build candidates."""
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < shape[0]) & (pts[:, 1] >= 0) & (pts[:, 1] < shape[1])
    return pts[ok]


def rasterize(fpts, shape):
    """Draw a float polyline as an 8-connected 1px stroke; None if it leaves the frame."""
    if len(fpts) < 2 or not inside(np.round(fpts).astype(int), shape):
        return None
    m = np.zeros(shape, np.uint8)
    xy = np.round(fpts[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(m, [xy], False, 1, 1)
    p = np.stack(np.nonzero(m), 1)
    return p if len(p) >= 5 else None


def _stats(w):
    return float(np.median(w)), float(np.std(w)), float((w > 8).mean())


def build_candidates(c, pool_pw, rng, shape, ink_dist):
    """{kind: (pixels, width_median, width_std, fill_share)} -- None where impossible.

    `pool_pw` holds (pixels, widths) for strokes from OTHER drawings.
    """
    w_true = ink_dist[c[:, 0], c[:, 1]] * 2.0
    path = order_path(c).astype(float)
    out = {"true": (c,) + _stats(w_true)}

    ang = rng.uniform(0, 2 * np.pi)
    off = np.array([np.sin(ang), np.cos(ang)]) * rng.uniform(5, 12)
    p = np.round(c + off).astype(int)
    out["displaced"] = (p,) + _stats(w_true) if inside(p, shape) else None

    th = np.deg2rad(rng.uniform(15, 40) * rng.choice([-1, 1]))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    cen = path.mean(0)
    p = rasterize((path - cen) @ R.T + cen, shape)
    out["rotated"] = (p,) + _stats(w_true) if p is not None else None

    n = len(c)
    near = [pw for pw in pool_pw if 0.6 * n <= len(pw[0]) <= 1.6 * n]
    if near:
        src, wsrc = near[rng.integers(len(near))]
        sp = order_path(src).astype(float)
        p = rasterize(sp - sp.mean(0) + np.asarray(c, float).mean(0), shape)
        out["foreign"] = (p,) + _stats(wsrc) if p is not None else None
    else:
        out["foreign"] = None
    return out
