#!/usr/bin/env python
"""Draw stroke tokens back into an image.

The project had no width-aware renderer: `make_negatives.rasterize` draws a 1px
polyline, which is all the negative generator needed. Generation needs the
opposite direction -- tokens back to a drawing -- and every fidelity claim in
this track is measured through this file, so it stays deliberately simple:
a segment between consecutive points at the mean of their two widths, plus a
disc at each point so corners are round rather than mitred. Binary union, so
draw order does not matter.
"""
import cv2
import numpy as np


def render(polys, widths, shape, min_width=1.0):
    """polys: list of (k, 2) float arrays in (row, col); widths: list of (k,) arrays.

    Returns a bool HxW mask of the drawn ink.
    """
    canvas = np.zeros(shape, np.uint8)
    for pts, w in zip(polys, widths):
        if len(pts) == 0:
            continue
        w = np.maximum(np.asarray(w, np.float32), min_width)
        xy = np.round(pts[:, ::-1]).astype(np.int32)
        for i in range(len(xy) - 1):
            t = int(max(1, round(float(w[i] + w[i + 1]) / 2.0)))
            cv2.line(canvas, tuple(xy[i]), tuple(xy[i + 1]), 1, t)
        for i in range(len(xy)):
            r = int(round(float(w[i]) / 2.0))
            if r >= 1:
                cv2.circle(canvas, tuple(xy[i]), r, 1, -1)
        if len(xy) == 1:
            cv2.circle(canvas, tuple(xy[0]), max(1, int(round(float(w[0]) / 2.0))), 1, -1)
    return canvas > 0


def render_discs(points, radii, shape):
    """Stamp discs -- used for leftover skeleton pixels (junction nodes, spurs,
    bridges, sub-min_len strokes) that belong to no stroke token."""
    canvas = np.zeros(shape, np.uint8)
    for (r, c), rad in zip(points, radii):
        cv2.circle(canvas, (int(round(c)), int(round(r))), max(1, int(round(rad))), 1, -1)
    return canvas > 0


def agreement(rendered, ink, tol=2):
    """Recall/precision of the render against the true ink, at `tol` px, plus
    F1 and IoU at 0px. Recall = how much of the drawing came back."""
    out = {}
    if not ink.any() or not rendered.any():
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "iou": 0.0, "extra": 1.0}
    d_ink = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3)
    d_ren = cv2.distanceTransform((~rendered).astype(np.uint8), cv2.DIST_L2, 3)
    out["recall"] = float((d_ren[ink] <= tol).mean())
    out["precision"] = float((d_ink[rendered] <= tol).mean())
    out["extra"] = 1.0 - out["precision"]
    inter = float((rendered & ink).sum())
    out["f1"] = 2 * inter / float(rendered.sum() + ink.sum())
    out["iou"] = inter / float((rendered | ink).sum())
    return out
