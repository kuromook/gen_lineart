#!/usr/bin/env python
"""Score the infill checkpoints on POSITION and SHAPE separately.

Why this exists: the pooled chamfer that `train_infill.py` reports is saturated
by the query's own granularity, not by the model. The query names the 64px cell
holding the missing stroke, so the answer is only known to +-32px; measured on
the same 300 eval instances, a prediction with the TRUE shape placed at the cell
centre already scores chamfer 19.8px, and both trained models sit at 22.4px.
A metric whose floor is 19.8 cannot separate 22.4 from a baseline at 23.3.

That is the same class of error as the 2026-09-18 encoding bug, where the input
could not carry the thing being asked (a displaced candidate moved its token by
0.0057 against a spread of 1.5647) and thirty flat epochs were nearly read as a
null result. There the instrument was the input; here it is the metric.

So this splits the two questions the pooled number confounds:

  POSITION  distance between the predicted polyline's centroid and the true
            one's, against the query cell centre as the reference. This is the
            question the track actually asks -- can the surrounding strokes say
            WHERE the missing stroke goes?
  SHAPE     chamfer after aligning the centroids, against the corpus-median
            stroke as the reference. Can they say WHAT it looks like?

Nothing is retrained; the checkpoints written by `train_infill.py` are loaded
as they are, and the eval instances are reconstructed to match exactly (same
pack, same group split, same per-instance seeds).
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_infill import (CELL, POINTS, Infill, Packed, chamfer,  # noqa: E402
                          encode, query_feat)
from train_stroke_transformer import WAVELENGTHS  # noqa: E402

F_IN = POINTS * 2 + len(WAVELENGTHS) * 4 + 2 + 3


def eval_rows(pack, min_strokes=40, test_frac=0.25, eval_panels=300, seed=20260918):
    """The held-out panels `train_infill.main()` evaluates on, rebuilt."""
    rows = [r for r in csv.DictReader(open(Path(pack) / "panels.csv"))
            if int(r["n"]) >= min_strokes]
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    test_g = set(groups[: int(len(groups) * test_frac)])
    te = [r for r in rows if r["group"] in test_g][:eval_panels]
    tr = [r for r in rows if r["group"] not in test_g]
    return tr, te


def instance(ds, i, max_strokes=600):
    """Replay `Panels.__getitem__`'s choice for eval instance i, keeping the
    raw strokes as well (the montage needs them)."""
    rng = np.random.default_rng(ds.seed + i)
    pts, meta = ds.load(i)
    if len(pts) > max_strokes:
        k = rng.choice(len(pts), max_strokes, replace=False)
        pts, meta = pts[k], meta[k]
    t = int(rng.integers(len(pts)))
    target, tmeta = pts[t], meta[t]
    ctx, cmeta = np.delete(pts, t, 0), np.delete(meta, t, 0)
    cell = (np.floor(target.mean(0) / CELL) + 0.5) * CELL
    return {"target": target, "tmeta": tmeta, "ctx": ctx, "cmeta": cmeta,
            "cell": cell.astype(np.float32)}


def median_stroke(ds, n=200):
    """Corpus-median normalised shape and arc, as `infill_baselines` builds it."""
    shapes, arcs = [], []
    for i in range(min(n, len(ds.rows))):
        pts, meta = ds.load(i)
        c = pts.mean(1, keepdims=True)
        span = np.linalg.norm(pts[:, -1] - pts[:, 0], axis=1) + 1e-6
        shapes.append(((pts - c) / span[:, None, None]).reshape(len(pts), -1))
        arcs.append(meta[:, 3])
    S = np.concatenate(shapes)
    A = np.concatenate(arcs)
    return np.median(S, 0).reshape(POINTS, 2), float(np.median(A))


@torch.no_grad()
def predict(model, inst, dev, context):
    X = encode(inst["ctx"], inst["cmeta"]) if len(inst["ctx"]) else np.zeros((0, F_IN), np.float32)
    q = query_feat(inst["cell"], F_IN)
    n = len(X)
    xb = np.zeros((1, n + 1, F_IN), np.float32)
    mb = np.ones((1, n + 1), bool)
    xb[0, 0] = q[0]; mb[0, 0] = False
    if n:
        xb[0, 1:] = X; mb[0, 1:] = False
    if context == "none":
        mb[:, 1:] = True
    xt = torch.from_numpy(xb).to(dev)
    mt = torch.from_numpy(mb).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        g, at = model(xt, mt)
    pred = g.float().cpu().numpy()[0] * 64.0 + inst["cell"]
    a = at.float().cpu().numpy()[0]
    return pred, {"arc": float(np.expm1(a[0] * 5.0)), "width": float(a[1] * 10.0)}


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Infill(F_IN, ck["args"]["d"], ck["args"]["layers"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def boot_diff(a, b, n=1000, seed=7):
    """Paired bootstrap over panels of median(a) - median(b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    d = [float(np.median(a[k]) - np.median(b[k]))
         for k in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return float(np.median(a) - np.median(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def montage(insts, preds, pos_err, shp, out, win=420, rows_per_bin=8):
    """24 rows, 8 per length band: the panel around the hole, the truth in
    green, the full-context prediction in red, the 64px query cell in blue."""
    arcs = np.array([x["tmeta"][3] for x in insts])
    bands = [("short <40px", np.flatnonzero(arcs < 40)),
             ("mid 40-120px", np.flatnonzero((arcs >= 40) & (arcs <= 120))),
             ("long >120px", np.flatnonzero(arcs > 120))]
    pick = []
    for label, idx in bands:
        take = idx[np.linspace(0, len(idx) - 1, min(rows_per_bin, len(idx))).astype(int)] if len(idx) else []
        pick += [(label, int(i)) for i in take]
    tiles = []
    for label, i in pick:
        inst, pr = insts[i], preds[i]
        c = inst["target"].mean(0)
        y0, x0 = c - win / 2.0
        img = np.full((win, win, 3), 255, np.uint8)

        def draw(poly, colour, thick):
            p = np.round((poly - np.array([y0, x0]))[:, ::-1]).astype(np.int32)
            cv2.polylines(img, [p.reshape(-1, 1, 2)], False, colour, thick, cv2.LINE_AA)

        for s in inst["ctx"]:
            if (np.abs(s.mean(0) - c) < win).all():
                draw(s, (185, 185, 185), 1)
        cy, cx = inst["cell"]
        cv2.rectangle(img, (int(round(cx - CELL / 2 - x0)), int(round(cy - CELL / 2 - y0))),
                      (int(round(cx + CELL / 2 - x0)), int(round(cy + CELL / 2 - y0))),
                      (220, 150, 60), 1)
        draw(inst["target"], (60, 160, 60), 2)
        draw(pr, (60, 60, 220), 2)
        bar = np.full((30, win, 3), 255, np.uint8)
        cv2.putText(bar, f"{label}  arc {inst['tmeta'][3]:.0f}px  pos {pos_err[i]:.1f}px  shape {shp[i]:.1f}px",
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(np.vstack([bar, img]))
    cols = 4
    grid = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.full_like(tiles[0], 255))
        grid.append(np.hstack([np.hstack([t, np.full((t.shape[0], 8, 3), 90, np.uint8)]) for t in row]))
    head = np.full((34, grid[0].shape[1], 3), 255, np.uint8)
    cv2.putText(head, "grey = rest of the panel | green = the missing stroke | red = model (full context) | blue = 64px query cell",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([head] + grid))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="results/panel_pack_20260919")
    p.add_argument("--ckpt-dir", default="results/infill_20260919")
    p.add_argument("--out", default="results/infill_eval_20260919")
    p.add_argument("--eval-panels", type=int, default=300)
    p.add_argument("--max-strokes", type=int, default=600)
    p.add_argument("--seed", type=int, default=20260918)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr, te = eval_rows(a.pack, eval_panels=a.eval_panels, seed=a.seed)
    ds_te = Packed(a.pack, te, a.max_strokes, a.seed + 1, eval_mode=True); ds_te.f_in = F_IN
    ds_tr = Packed(a.pack, tr, a.max_strokes, a.seed); ds_tr.f_in = F_IN
    shape, arc = median_stroke(ds_tr)
    print(f"eval panels {len(te)}  train panels {len(tr)}  corpus median arc {arc:.0f}px", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {c: load_model(Path(a.ckpt_dir) / f"best_{c}.pt", dev) for c in ("full", "none")}

    insts = [instance(ds_te, i, a.max_strokes) for i in range(len(te))]
    res = {c: {"pos": [], "shape": [], "arc_err": [], "w_err": [], "pred": []}
           for c in ("full", "none")}
    ref = {"cell_pos": [], "median_shape": []}
    for inst in insts:
        tgt = inst["target"]; tc = tgt.mean(0)
        ref["cell_pos"].append(float(np.linalg.norm(inst["cell"] - tc)))
        ms = shape * arc
        ref["median_shape"].append(chamfer(ms - ms.mean(0) + tc, tgt))
        for c, m in models.items():
            pr, at = predict(m, inst, dev, c)
            res[c]["pred"].append(pr)
            res[c]["pos"].append(float(np.linalg.norm(pr.mean(0) - tc)))
            res[c]["shape"].append(chamfer(pr - pr.mean(0) + tc, tgt))
            res[c]["arc_err"].append(abs(at["arc"] - float(inst["tmeta"][3])))
            res[c]["w_err"].append(abs(at["width"] - float(inst["tmeta"][1])))

    def q(v):
        v = np.asarray(v)
        return float(np.median(v)), float((v <= 8).mean()), float((v <= 16).mean())

    lines = []
    lines.append(f"{'':<26}{'median':>9}{'<=8px':>9}{'<=16px':>9}")
    lines.append("POSITION (centroid error, px)")
    for name, v in (("model full context", res["full"]["pos"]),
                    ("model no context", res["none"]["pos"]),
                    ("query cell centre", ref["cell_pos"])):
        m, w8, w16 = q(v)
        lines.append(f"  {name:<24}{m:>9.1f}{w8:>9.1%}{w16:>9.1%}")
    lines.append("SHAPE (centroid-aligned chamfer, px)")
    for name, v in (("model full context", res["full"]["shape"]),
                    ("model no context", res["none"]["shape"]),
                    ("corpus median stroke", ref["median_shape"])):
        m, w8, w16 = q(v)
        lines.append(f"  {name:<24}{m:>9.1f}{w8:>9.1%}{w16:>9.1%}")
    lines.append("ATTRIBUTES (median abs error)")
    for c in ("full", "none"):
        lines.append(f"  {'model ' + c:<24}{'arc ' + format(np.median(res[c]['arc_err']), '.1f') + 'px':>18}"
                     f"{'width ' + format(np.median(res[c]['w_err']), '.2f') + 'px':>18}")
    true_arc = np.median([x["tmeta"][3] for x in insts])
    true_w = np.median([x["tmeta"][1] for x in insts])
    lines.append(f"  {'(truth medians)':<24}{'arc ' + format(true_arc, '.0f') + 'px':>18}{'width ' + format(true_w, '.2f') + 'px':>18}")
    d, lo, hi = boot_diff(res["full"]["pos"], res["none"]["pos"])
    lines.append(f"CONTEXT EFFECT on position: median(full) - median(none) = {d:+.2f}px  "
                 f"[95% CI {lo:+.2f}, {hi:+.2f}]")
    table = "\n".join(lines)
    print(table, flush=True)

    with open(out / "per_instance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "arc", "width", "pos_full", "pos_none", "pos_cell",
                    "shape_full", "shape_none", "shape_median"])
        for i, inst in enumerate(insts):
            w.writerow([i, round(float(inst["tmeta"][3]), 1), round(float(inst["tmeta"][1]), 2),
                        round(res["full"]["pos"][i], 2), round(res["none"]["pos"][i], 2),
                        round(ref["cell_pos"][i], 2), round(res["full"]["shape"][i], 2),
                        round(res["none"]["shape"][i], 2), round(ref["median_shape"][i], 2)])
    mpath = montage(insts, res["full"]["pred"], res["full"]["pos"], res["full"]["shape"],
                    out / "montage_predictions.png")
    (out / "summary.txt").write_text(table + f"\n\nmontage: {mpath}\n")
    print(f"\nmontage: {mpath}\nper-instance: {out / 'per_instance.csv'}", flush=True)


if __name__ == "__main__":
    main()
