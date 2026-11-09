#!/usr/bin/env python
"""Per-tile features of v4 rough/lineart pairs (Track D reopen, 2026-11-08).

Registration: doc/track_d_correlation_register_20261108.md. This file is the
measurement half; correlation and the permutation null are a separate script
that reads the CSVs written here, so no number is looked at while measuring.

Per page (v4, ok, primary; sketch_misaligned_by_eye excluded upstream by
pair_quality) the page is cut into a native 480 grid. Every grid cell is
counted by reason (U1 tile rule, register section 4):
  edge        partial cell at the page border (< 480)
  empty       sketch<160 and line<128 both below 1% ink
  sketch_only only the sketch side reaches 1%
  line_only   only the line side reaches 1%
  kept        both reach 1%  -> measured
The kept set is fixed by the 160 arm and reused by the 200 arm (exposure is
not moved per arm).

Per kept tile, three sides are measured with the same code:
  line   (gray < 128), sketch160 (gray < 160), sketch200 (gray < 200).
Axes (register section 5): MLP pixel axes with the threshold passed in, Track
G stroke axes (tools/stroke, copied from lineart-stroke-grammar de593f6) on
the same ink mask, stroke curvature, line direction (edge histogram and
stroke principal axis), edge sharpness, and ink_ratio as exposure only.
U2: the sketch skeleton is split into segments (min_seg 8, stroke_churn
rule); each segment's median distance to the GT skeleton puts it in <=3 /
3-8 / >8 px; the same with the GT skeleton rotated 180 deg as the null arm.
Per class the segments are summarised (count, length, straightness,
curvature, width, darkness, sharpness, direction).

Outputs (under --out): pages.csv (one row per page, tile counts by reason),
tiles.csv (one row per kept tile x side), segs.csv (one row per kept tile x
sketch arm x {real, rot180} x class). Resumable: pages already in pages.csv
are skipped.
"""
import argparse, csv, glob, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "stroke"))
sys.path.insert(0, str(HERE.parent / "pair_extraction"))
from evaluate_stroke_stability import component_stats, skeletonize as mlp_skeletonize  # noqa: E402
from tile_region_manifest_480 import edge_map, line_width_stats  # noqa: E402
from measure_lineart_profile import BACKGROUND_THRESHOLD, FAINT_NEAR_PX, fill_ratio, grid_heterogeneity  # noqa: E402
from stroke_churn import dist_to, ink_skeleton, segments  # noqa: E402
from strokes import ink_mask, strokes_from_skeleton  # noqa: E402
from gate1_features import ang_diff, curvature, direction, lattice_residual, terminal_tangent  # noqa: E402
from make_negatives import order_path  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

T = 480
MIN_TILE_INK = 0.01
LINE_THR, SK_MAIN, SK_ALT = 128, 160, 200
MIN_SEG = 8
SIDES = (("line", LINE_THR), ("sketch160", SK_MAIN), ("sketch200", SK_ALT))


def faint_locality(gray, thr):
    """measure_lineart_profile.faint_locality with the ink threshold passed in."""
    ink = gray < thr
    faint = (gray >= thr) & (gray <= BACKGROUND_THRESHOLD)
    if faint.sum() < 20:
        return 1.0, 0.0
    if ink.sum() < 5:
        return 0.0, float(T)
    d = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3)[faint]
    return float((d <= FAINT_NEAR_PX).mean()), float(d.mean())


def edge_dirs(gray):
    """orientation_entropy's edge histogram, also returned as 4 direction shares.
    The angle is the gradient's; a line's own direction is perpendicular, so the
    bins are rotated by 90 deg to name the line direction (0 = horizontal line)."""
    e = edge_map(gray).astype(np.float32)
    gx = cv2.Sobel(e, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(e, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy); v = mag > 0.25
    out = {"orientation_entropy": 0.0, "edir_0": np.nan, "edir_45": np.nan, "edir_90": np.nan, "edir_135": np.nan}
    if v.sum() < 40:
        return out
    ang = np.mod(np.arctan2(gy[v], gx[v]), np.pi)
    h12, _ = np.histogram(ang, bins=12, range=(0, np.pi), weights=mag[v])
    p = h12 / max(h12.sum(), 1e-9); p = p[p > 0]
    out["orientation_entropy"] = float(-(p * np.log(p)).sum() / np.log(12))
    line_ang = np.mod(ang + np.pi / 2, np.pi)
    b = np.mod(np.round(line_ang / (np.pi / 4)).astype(int), 4)
    w = np.bincount(b, weights=mag[v], minlength=4); w = w / max(w.sum(), 1e-9)
    out.update({"edir_0": w[0], "edir_45": w[1], "edir_90": w[2], "edir_135": w[3]})
    return out


def sharpness_map(gray):
    g = cv2.GaussianBlur(gray, (0, 0), 0.7).astype(np.float32)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def edge_sharpness(gray, ink, sharp):
    """Median gradient magnitude on the ink boundary (ink pixels with a non-ink 4-neighbour)."""
    er = cv2.erode(ink.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8))
    b = ink & (er == 0)
    return float(np.median(sharp[b])) if b.sum() >= 20 else np.nan


def dir_bins(angles, weights):
    """Line direction (radians, mod pi; 0 = horizontal) -> 4 shares."""
    if len(angles) == 0:
        return [np.nan] * 4
    b = np.mod(np.round(np.mod(angles, np.pi) / (np.pi / 4)).astype(int), 4)
    w = np.bincount(b, weights=weights, minlength=4)
    return list(w / max(w.sum(), 1e-9))


def stroke_axes(gray, thr, sharp):
    """Track G gate2 profile() on an array, threshold passed in, plus curvature,
    stroke direction and the mean nearest-neighbour distance."""
    keys = ["s_n", "s_len_p50", "s_len_p90", "s_straight_p50", "s_width_p50", "s_width_cv", "s_spans_mean",
            "s_junction_frac", "r_nn_dist_p50", "r_nn_dist_mean", "r_nn_ang_p50", "r_end_gap_p50", "r_end_gap_p10",
            "r_hatch_n_p50", "r_par_frac", "r_density_cv", "curv_p50", "curv_p90",
            "sdir_0", "sdir_45", "sdir_90", "sdir_135"]
    nan = {k: np.nan for k in keys}
    ink = ink_mask(gray, thr=thr)
    if ink.sum() < 200:
        return nan, "few_ink"
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(ink)
    spans, sos, keep, info = strokes_from_skeleton(sk, dist * 2)
    groups = {}
    for i in range(len(spans)):
        groups.setdefault(int(sos[i]), []).append(i)
    pts, lens, widths, straight, nspans, curv = [], [], [], [], [], []
    for k, mem in groups.items():
        if not keep[k]:
            continue
        rr = np.concatenate([spans[i][0] for i in mem]); cc = np.concatenate([spans[i][1] for i in mem])
        p = np.stack([rr, cc], 1)
        q = order_path(p).astype(float)
        span = float(np.linalg.norm(q[0] - q[-1])) if len(q) > 1 else 0.0
        pts.append((p, q)); lens.append(len(rr)); widths.append(float(np.median(dist[rr, cc] * 2.0)))
        straight.append(span / max(len(rr), 1)); nspans.append(len(mem)); curv.append(curvature(q))
    n = len(pts)
    if n < 3:
        return nan, "few_strokes"
    cent = np.array([p.mean(0) for p, _ in pts])
    dirs = np.array([direction(p) for p, _ in pts])
    ends, owner = [], []
    for i, (_p, q) in enumerate(pts):
        if len(q) >= 2:
            ends += [q[0], q[-1]]; owner += [i, i]
    ends = np.array(ends); owner = np.array(owner)
    nn_d, nn_a, e_gap, h_n, par = [], [], [], [], []
    for i in range(n):
        d = np.linalg.norm(cent - cent[i], axis=1); d[i] = np.inf
        j = int(d.argmin())
        nn_d.append(float(d[j])); nn_a.append(float(ang_diff(dirs[j], dirs[i])))
        fam = (d < 60) & (ang_diff(dirs, dirs[i]) < np.deg2rad(12))
        par.append(float(fam.sum()))
        nrm = np.array([np.cos(dirs[i]), -np.sin(dirs[i])])
        res, _cv, _sp = lattice_residual(((cent - cent[i]) @ nrm)[fam])
        if res < 90:
            h_n.append(float(fam.sum()))
        if len(ends):
            mine = owner == i
            for e in ends[mine]:
                dd = np.linalg.norm(ends[~mine] - e, axis=1)
                if len(dd):
                    e_gap.append(float(dd.min()))
    cells = cv2.resize(sk.astype(np.float32), (12, 12), interpolation=cv2.INTER_AREA)
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else np.nan
    # direction() = atan2(d_row, d_col) mod pi: 0 = horizontal, pi/4 = down-right in image
    # coordinates; edge_dirs() uses the same convention (checked 2026-11-08).
    sd = dir_bins(np.array(dirs), np.array(lens, float))
    return {
        "s_n": n, "s_len_p50": q(lens, 50), "s_len_p90": q(lens, 90), "s_straight_p50": q(straight, 50),
        "s_width_p50": q(widths, 50), "s_width_cv": float(np.std(widths) / (np.mean(widths) + 1e-9)),
        "s_spans_mean": float(np.mean(nspans)), "s_junction_frac": float(info["nodes"]) / max(n, 1),
        "r_nn_dist_p50": q(nn_d, 50), "r_nn_dist_mean": float(np.mean(nn_d)), "r_nn_ang_p50": q(nn_a, 50),
        "r_end_gap_p50": q(e_gap, 50), "r_end_gap_p10": q(e_gap, 10),
        "r_hatch_n_p50": q(h_n, 50) if h_n else np.nan, "r_par_frac": float(np.mean(np.array(par) > 2)),
        "r_density_cv": float(cells.std() / (cells.mean() + 1e-9)),
        "curv_p50": q(curv, 50), "curv_p90": q(curv, 90),
        "sdir_0": sd[0], "sdir_45": sd[1], "sdir_90": sd[2], "sdir_135": sd[3],
    }, "ok"


def pixel_axes(gray, thr):
    ink = gray < thr
    n = int(ink.sum())
    out = {"ink_ratio": n / float(T * T)}
    out["faint_near_ink_ratio"], out["faint_mean_dist_to_ink"] = faint_locality(gray, thr)
    out.update(grid_heterogeneity(ink))
    out.update(edge_dirs(gray))
    out["fill_ratio"] = fill_ratio(ink)
    if ink.all():
        # MLP skeletonize() erodes with cv2's default border (= ink), so an all-ink tile never
        # shrinks and the loop never ends (run 1 hung on 7 pages, 2026-11-09). Only this case is
        # changed, so tiles measured before the fix keep their values; the count is printed.
        out.update({"px_status": "all_ink", "long_component_ratio": np.nan,
                    "components_per_1k_ink_px": np.nan, "line_width_p50": np.nan})
        return out
    skel = mlp_skeletonize(ink)
    comp = component_stats(skel)
    w50, _w95 = line_width_stats(ink)
    out.update({"px_status": "ok", "long_component_ratio": comp["long_component_ratio"],
                "components_per_1k_ink_px": comp["component_count"] / max(n, 1) * 1000.0,
                "line_width_p50": w50})
    return out


def seg_summary(gray, sk_src, to_gt, sharp, dist_ink):
    """Sketch segments by median distance to the GT skeleton -> per-class summary."""
    rows = {c: [] for c in ("le3", "3to8", "gt8")}
    for rr, cc in segments(sk_src, MIN_SEG):
        m = float(np.median(np.minimum(to_gt[rr, cc], 99)))
        c = "le3" if m <= 3 else ("3to8" if m <= 8 else "gt8")
        p = np.stack([rr, cc], 1)
        q = order_path(p).astype(float)
        span = float(np.linalg.norm(q[0] - q[-1])) if len(q) > 1 else 0.0
        rows[c].append((len(rr), span / max(len(rr), 1), curvature(q), float(np.median(dist_ink[rr, cc] * 2.0)),
                        float(np.median(gray[rr, cc])), float(np.median(sharp[rr, cc])), direction(p)))
    out = {}
    for c, r in rows.items():
        if not r:
            out[c] = {"n": 0, "len_sum": 0}
            continue
        a = np.array(r, float)
        L = a[:, 0]
        sd = dir_bins(a[:, 6], L)
        out[c] = {"n": len(r), "len_sum": float(L.sum()), "len_p50": float(np.median(L)),
                  "straight_p50": float(np.median(a[:, 1])), "curv_p50": float(np.median(a[:, 2])),
                  "width_p50": float(np.median(a[:, 3])), "gray_p50": float(np.median(a[:, 4])),
                  "sharp_p50": float(np.median(a[:, 5])),
                  "dir_0": sd[0], "dir_45": sd[1], "dir_90": sd[2], "dir_135": sd[3]}
    return out


def measure_page(job):
    key, sketch_path, line_path = job
    t0 = time.time()
    s = np.asarray(Image.open(sketch_path).convert("L"))
    g = np.asarray(Image.open(line_path).convert("L"))
    page = {"key": key, "h": s.shape[0], "w": s.shape[1], "shape_mismatch": int(s.shape != g.shape),
            "edge": 0, "empty": 0, "sketch_only": 0, "line_only": 0, "kept": 0}
    tiles, segrows = [], []
    if s.shape != g.shape:
        return page, tiles, segrows
    H, W = s.shape
    for r0 in range(0, H, T):
        for c0 in range(0, W, T):
            if r0 + T > H or c0 + T > W:
                page["edge"] += 1
                continue
            st = np.ascontiguousarray(s[r0:r0 + T, c0:c0 + T]); gt = np.ascontiguousarray(g[r0:r0 + T, c0:c0 + T])
            si, li = (st < SK_MAIN).mean() >= MIN_TILE_INK, (gt < LINE_THR).mean() >= MIN_TILE_INK
            if not (si or li):
                page["empty"] += 1; continue
            if not li:
                page["sketch_only"] += 1; continue
            if not si:
                page["line_only"] += 1; continue
            page["kept"] += 1
            tid = f"{r0}_{c0}"
            gt_sk = ink_skeleton(gt)
            to_gt = dist_to(gt_sk)
            to_gt_rot = dist_to(np.ascontiguousarray(np.rot90(gt_sk, 2)))
            for side, thr in SIDES:
                gray = gt if side == "line" else st
                sharp = sharpness_map(gray)
                ink = gray < thr
                row = {"key": key, "tile": tid, "side": side}
                row.update(pixel_axes(gray, thr))
                sa, status = stroke_axes(gray, thr, sharp)
                row.update(sa); row["stroke_status"] = status
                row["edge_sharpness_p50"] = edge_sharpness(gray, ink, sharp)
                tiles.append(row)
                if side != "line":
                    n_ink = int(ink.sum())
                    if n_ink < 20 or ink.size - n_ink < 20:
                        continue
                    sk = skeletonize(ink)
                    dist_ink = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
                    for arm, tg in (("real", to_gt), ("rot180", to_gt_rot)):
                        for c, d in seg_summary(gray, sk, tg, sharp, dist_ink).items():
                            segrows.append({"key": key, "tile": tid, "arm_thr": side, "arm": arm, "cls": c, **d})
    page["sec"] = round(time.time() - t0, 1)
    return page, tiles, segrows


def load_jobs(root, only=None, limit=0):
    rows = list(csv.DictReader(open(Path(root) / "clip_pairs" / "clip_pairs_qc_v4.csv", encoding="utf-8")))
    idx = {os.path.basename(p): p for p in glob.glob(str(Path(root) / "clip_pairs" / "**" / "*.jpg"), recursive=True)}
    drop = {}
    jobs = []
    for x in rows:
        if x["pair_quality"] != "ok":
            drop[x["pair_quality"]] = drop.get(x["pair_quality"], 0) + 1; continue
        if x["is_primary"] != "True":
            drop["not_primary"] = drop.get("not_primary", 0) + 1; continue
        key = f"{x['work_id']}|{x['slug']}|{x['page']}"
        if only and x["work_id"] not in only:
            continue
        jobs.append((key, idx[x["sketch"]], idx[x["line"]]))
    print(f"population: {len(rows)} pairs; dropped by reason {drop}; kept {len(jobs) if not only else '(subset)'}",
          flush=True)
    return jobs[:limit] if limit else jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/clip_pairs_v4")
    ap.add_argument("--out", default="results/pair_feature_correlation_20261108/measure")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(a.root, a.only, a.limit)
    done = set()
    if (out / "pages.csv").exists():
        done = {r["key"] for r in csv.DictReader(open(out / "pages.csv", encoding="utf-8"))}
    jobs = [j for j in jobs if j[0] not in done]
    print(f"to measure: {len(jobs)} pages (already done {len(done)})", flush=True)
    writers = {}

    def write(name, rows):
        if not rows:
            return
        f = out / name
        new = not f.exists()
        with open(f, "a", newline="", encoding="utf-8") as fh:
            if name not in writers:
                writers[name] = list(rows[0].keys()) if new else next(csv.reader(open(f, encoding="utf-8")))
            w = csv.DictWriter(fh, fieldnames=writers[name], extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)

    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (page, tiles, segs) in enumerate(pool.imap_unordered(measure_page, jobs)):
            # all-key header for segs/tiles: fill missing keys
            write("tiles.csv", tiles)
            write("segs.csv", [{**{k: "" for k in SEG_KEYS}, **r} for r in segs])
            write("pages.csv", [page])
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)} pages, {time.time() - t0:.0f}s", flush=True)


SEG_KEYS = ["key", "tile", "arm_thr", "arm", "cls", "n", "len_sum", "len_p50", "straight_p50", "curv_p50",
            "width_p50", "gray_p50", "sharp_p50", "dir_0", "dir_45", "dir_90", "dir_135"]

if __name__ == "__main__":
    main()
