#!/usr/bin/env python
"""Gate 2: can a STROKE-level measure tell GT line art from broken model output,
and does it beat the single pixel-level axes we already have?

The unit is the linked stroke (strokes.py). Per image we summarize the stroke
set -- how many, how long, how straight, and above all how the strokes sit
relative to each other -- and ask how well each number separates GT from a
model snapshot. Every stroke axis is reported beside `measure_lineart_profile`'s
existing pixel axes (midtone_frac and friends) on the same images: the standing
rule of this project is that a new measure means nothing until it is shown
against the baseline it claims to improve on.

Images are 480px tiles here, because that is what the trained models emit.
"""
import argparse, csv, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from strokes import ink_mask, strokes_from_skeleton  # noqa: E402
from gate1_features import ang_diff, direction, lattice_residual, terminal_tangent  # noqa: E402
from make_negatives import order_path  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402
import measure_lineart_profile as mlp  # noqa: E402

STROKE_KEYS = ["s_n", "s_len_p50", "s_len_p90", "s_straight_p50", "s_width_p50", "s_width_cv",
               "s_fill_frac", "s_spans_mean", "s_captured", "s_junction_frac",
               "r_nn_dist_p50", "r_nn_ang_p50", "r_end_gap_p50", "r_end_gap_p10",
               "r_hatch_res_p50", "r_hatch_n_p50", "r_par_frac", "r_density_cv"]


def profile(path):
    g = cv2.imread(str(path), 0)
    if g is None:
        return None
    if g.shape != (480, 480):
        g = cv2.resize(g, (480, 480), interpolation=cv2.INTER_AREA)
    ink = ink_mask(g)
    out = {"image": str(path)}
    if ink.sum() < 200:
        return {**out, **{k: np.nan for k in STROKE_KEYS}}
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(ink)
    spans, sos, keep, info = strokes_from_skeleton(sk, dist * 2)
    groups = {}
    for i, (r, c) in enumerate(spans):
        groups.setdefault(int(sos[i]), []).append(i)
    pts, lens, widths, straight, fills, nspans = [], [], [], [], [], []
    for k, mem in groups.items():
        if not keep[k]:
            continue
        rr = np.concatenate([spans[i][0] for i in mem]); cc = np.concatenate([spans[i][1] for i in mem])
        p = np.stack([rr, cc], 1)
        w = dist[rr, cc] * 2.0
        path_o = order_path(p).astype(float)
        span = float(np.linalg.norm(path_o[0] - path_o[-1])) if len(path_o) > 1 else 0.0
        pts.append((p, path_o))
        lens.append(len(rr)); widths.append(float(np.median(w)))
        straight.append(span / max(len(rr), 1)); fills.append(float((w > 8).mean()))
        nspans.append(len(mem))
    n = len(pts)
    if n < 3:
        return {**out, **{k: np.nan for k in STROKE_KEYS}}
    cent = np.array([p.mean(0) for p, _ in pts])
    dirs = np.array([direction(p) for p, _ in pts])
    ends, tans, owner = [], [], []
    for i, (_p, q) in enumerate(pts):
        if len(q) < 2:
            continue
        ta, tb = terminal_tangent(q)
        ends += [q[0], q[-1]]; tans += [ta, tb]; owner += [i, i]
    ends = np.array(ends); owner = np.array(owner)
    nn_d, nn_a, e_gap, h_res, h_n, par = [], [], [], [], [], []
    for i in range(n):
        d = np.linalg.norm(cent - cent[i], axis=1); d[i] = np.inf
        j = int(d.argmin())
        nn_d.append(float(d[j])); nn_a.append(float(ang_diff(dirs[j], dirs[i])))
        near = d < 60
        fam = near & (ang_diff(dirs, dirs[i]) < np.deg2rad(12))
        par.append(float(fam.sum()))
        nrm = np.array([np.cos(dirs[i]), -np.sin(dirs[i])])
        res, _cv, _sp = lattice_residual(((cent - cent[i]) @ nrm)[fam])
        if res < 90:
            h_res.append(res); h_n.append(float(fam.sum()))
        if len(ends):
            mine = owner == i
            for e in ends[mine]:
                dd = np.linalg.norm(ends[~mine] - e, axis=1)
                if len(dd):
                    e_gap.append(float(dd.min()))
    cells = cv2.resize(sk.astype(np.float32), (12, 12), interpolation=cv2.INTER_AREA)
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else np.nan
    out.update({
        "s_n": n, "s_len_p50": q(lens, 50), "s_len_p90": q(lens, 90),
        "s_straight_p50": q(straight, 50), "s_width_p50": q(widths, 50),
        "s_width_cv": float(np.std(widths) / (np.mean(widths) + 1e-9)),
        "s_fill_frac": float(np.mean(np.array(fills) > 0.2)),
        "s_spans_mean": float(np.mean(nspans)), "s_captured": float(info["strokes"]) and
        float(sum(lens)) / max(int(sk.sum()), 1),
        "s_junction_frac": float(info["nodes"]) / max(n, 1),
        "r_nn_dist_p50": q(nn_d, 50), "r_nn_ang_p50": q(nn_a, 50),
        "r_end_gap_p50": q(e_gap, 50), "r_end_gap_p10": q(e_gap, 10),
        "r_hatch_res_p50": q(h_res, 50) if h_res else np.nan,
        "r_hatch_n_p50": q(h_n, 50) if h_n else np.nan,
        "r_par_frac": float(np.mean(np.array(par) > 2)),
        "r_density_cv": float(cells.std() / (cells.mean() + 1e-9)),
    })
    prof = mlp.profile_metrics(Path(path))
    for k, v in prof.items():
        if isinstance(v, (int, float)) and k not in out:
            out[f"p_{k}"] = float(v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", default="/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line")
    p.add_argument("--out-roots", nargs="*", default=[
        "/home/sh1/deepl/lineart-pair-signal/results/h34_alignment_probe_20260915/aligned",
        "/home/sh1/deepl/lineart-pair-signal/results/h34_alignment_probe_20260915/control",
        "/home/sh1/deepl/lineart-controlnet-sd15-refine/results/controlnet_lora_manga_consistency_w0.2_snapshot_probe_20260914",
    ])
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default="results/gate2_20260918/profiles.csv")
    a = p.parse_args()
    groups = [("GT", sorted(Path(a.gt_dir).glob("*")))]
    for root in a.out_roots:
        tag = Path(root).parent.name.split("_")[0] + "/" + Path(root).name
        for sd in sorted(Path(root).glob("step_*")):
            groups.append((f"{tag}/{sd.name}", sorted(sd.glob("*.png"))))
    tasks, meta = [], []
    for name, files in groups:
        if a.limit:
            files = files[: a.limit]
        for f in files:
            tasks.append(str(f)); meta.append(name)
    print(f"groups {len(groups)}  images {len(tasks)}", flush=True)
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for i, (m, r) in enumerate(zip(meta, pool.imap(profile, tasks, chunksize=4)), 1):
            if r is None:
                continue
            rows.append({"group": m, **r})
            if i % 200 == 0:
                print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["group"] + [c for c in rows[0] if c != "group"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
    print(f"rows {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
