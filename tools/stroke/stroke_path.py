#!/usr/bin/env python
"""One ordered polyline per stroke, with a width at every point.

Why this file exists (measured 2026-09-18, before writing it):
`make_negatives.order_path()` walks a stroke's pixels by 8-connected nearest
neighbour and stops at the first gap. A linked stroke is several spans joined
across junction nodes, and **node pixels belong to no stroke**, so the walk
stops at the first junction: it kept only **0.49-0.66** of a stroke's pixels
over four random panels, and 22-47% of strokes lost more than a tenth. Every
polyline in `results/panel_tokens_20260918/` -- and both Transformer runs on
top of it -- was built from roughly half a stroke.

`order_path` is still the right primitive *per span* (a span is one 8-connected
run with no branch point). What was missing is the chaining across the ~4px
node gaps, which is what this module adds, plus a width per point instead of
one median per stroke (worth +0.10 ink F1 / +0.15 IoU in the ceiling
measurement).

Junction note: 86-96% of nodes have three arms, not four, and 38-57% have an
arm shorter than 8px -- most "junctions" are a bulge in the ink sprouting a
short branch of medial axis, not two drawn lines crossing. So chaining through
a node is usually undoing an artifact of the tokenizer, not merging two
different lines.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_negatives import order_path  # noqa: E402


def span_paths(spans, members):
    """Ordered (n, 2) float paths for one stroke's member spans, longest first."""
    out = []
    for i in members:
        rr, cc = spans[i]
        if len(rr) == 0:
            continue
        p = order_path(np.stack([rr, cc], 1))
        if len(p) >= 1:
            out.append(p.astype(np.float32))
    return out


def chain(paths):
    """Join span paths end to end, starting from an extremity.

    Returns (path, gaps): the concatenated polyline and the jump length at each
    join, so a caller can report how far the chaining had to reach (expected
    ~4px, the measured node size).
    """
    if not paths:
        return np.zeros((0, 2), np.float32), []
    if len(paths) == 1:
        return paths[0], []
    ends = [(p[0], p[-1]) for p in paths]
    centre = np.concatenate(paths).mean(0)
    # start at the endpoint furthest from the stroke's centre: that is an end of
    # the whole chain, not the middle of it
    si, se = max(((i, e) for i in range(len(paths)) for e in (0, 1)),
                 key=lambda ie: np.linalg.norm(ends[ie[0]][ie[1]] - centre))
    cur = paths[si] if se == 0 else paths[si][::-1]
    chain_parts, gaps, used = [cur], [], {si}
    tail = cur[-1]
    while len(used) < len(paths):
        best = None
        for i in range(len(paths)):
            if i in used:
                continue
            for e in (0, 1):
                d = float(np.linalg.norm(ends[i][e] - tail))
                if best is None or d < best[0]:
                    best = (d, i, e)
        d, i, e = best
        p = paths[i] if e == 0 else paths[i][::-1]
        chain_parts.append(p)
        gaps.append(d)
        used.add(i)
        tail = p[-1]
    return np.concatenate(chain_parts).astype(np.float32), gaps


def resample(path, per_px=8.0, k_min=8, k_max=64):
    """Arc-length-even points, one per `per_px` of path, clamped to [k_min, k_max].

    16 fixed points is enough at 2px tolerance for 99.6% of strokes (measured),
    but short strokes waste points and strokes over ~300px lose curvature, so
    the count follows the length.
    """
    if len(path) < 2:
        return np.repeat(path[:1], max(k_min, 2), 0).astype(np.float32)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if d[-1] <= 0:
        return np.repeat(path[:1], k_min, 0).astype(np.float32)
    k = int(np.clip(round(d[-1] / per_px) + 1, k_min, k_max))
    t = np.linspace(0.0, d[-1], k)
    return np.stack([np.interp(t, d, path[:, 0]), np.interp(t, d, path[:, 1])], 1).astype(np.float32)


def widths_at(points, dist):
    """Ink width (2 x distance transform) at each point, nearest-pixel sampled."""
    h, w = dist.shape
    r = np.clip(np.round(points[:, 0]).astype(int), 0, h - 1)
    c = np.clip(np.round(points[:, 1]).astype(int), 0, w - 1)
    return (dist[r, c] * 2.0).astype(np.float32)


def stroke_polyline(spans, members, dist, per_px=8.0, k_min=8, k_max=64):
    """spans/members as returned by strokes_from_skeleton -> (points, widths, info).

    `info` carries the chaining diagnostics: how many pixels the ordered path
    kept out of the stroke's pixels (1.0 means nothing was truncated) and the
    largest gap it had to jump.
    """
    paths = span_paths(spans, members)
    raw = sum(len(spans[i][0]) for i in members)
    path, gaps = chain(paths)
    pts = resample(path, per_px, k_min, k_max)
    info = {"kept": len(path) / max(raw, 1), "n_spans": len(members),
            "max_gap": max(gaps) if gaps else 0.0, "arc": float(
                np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0}
    return pts, widths_at(pts, dist), info
