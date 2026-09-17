#!/usr/bin/env python
"""Step 2 prerequisite: per-pixel keep/drop labels for the TRAINING pairs.

The label definition is copied verbatim from
`results/oracle_visual_check_20260913/build_oracle.py` -- one-to-one bipartite
matching between the conditioning image's Canny edge pixels and GT's, at
BSDS_TOLERANCE_PX. A conditioning edge pixel is `keep` iff it is matched. That
mask is the deletion oracle, and the classifier's job is to predict it from the
conditioning image alone. Any other definition would break comparability with
the 0.7425 / measured 0.6765 ceiling.

`maximum_bipartite_matching` stalls for minutes on particular tiles (a
shared-infrastructure finding from Track D). The full pass therefore runs one
subprocess per tile with a hard timeout; this module does a single tile when
given --one.
"""
import argparse, csv, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from tile_region_manifest_480 import edge_map  # noqa: E402

IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0


def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def pixel_oracle(pred_edge, gt_edge, tolerance_px=BSDS_TOLERANCE_PX):
    shape = pred_edge.shape
    pred_pts = np.column_stack(np.nonzero(pred_edge))
    gt_pts = np.column_stack(np.nonzero(gt_edge))
    mask = np.zeros(shape, dtype=bool)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return mask
    tree = cKDTree(gt_pts)
    rows, cols = [], []
    for i, matches in enumerate(tree.query_ball_point(pred_pts, r=tolerance_px)):
        for j in matches:
            rows.append(i)
            cols.append(j)
    if not rows:
        return mask
    graph = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                       shape=(len(pred_pts), len(gt_pts)))
    match = maximum_bipartite_matching(graph, perm_type="column")
    idx = np.nonzero(match >= 0)[0]
    mask[pred_pts[idx, 0], pred_pts[idx, 1]] = True
    return mask


def one(cond_path, gt_path, out_path):
    cond_edge = edge_map(load_gray(cond_path))
    keep = pixel_oracle(cond_edge, edge_map(load_gray(gt_path)))
    # two bit-planes in one PNG: 1 = conditioning edge, 2 = keep
    packed = cond_edge.astype(np.uint8) | (keep.astype(np.uint8) << 1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(packed).save(out_path)
    return int(cond_edge.sum()), int(keep.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C = "/home/sh1/deepl/lineart-stroke-selection"
    p.add_argument("--cond-dir", default=f"{C}/data/rough_lineart_coarse")
    p.add_argument("--gt-dir", default=f"{C}/data/line")
    p.add_argument("--list", default=f"{C}/data/train_list.txt")
    p.add_argument("--out-dir", default=f"{C}/results/keep_labels_20260917")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--one", default="")
    a = p.parse_args()

    if a.one:
        n_edge, n_keep = one(Path(a.cond_dir) / a.one, Path(a.gt_dir) / a.one,
                             Path(a.out_dir) / (Path(a.one).stem + ".png"))
        print(f"{n_edge} {n_keep}")
        return

    names = [l.strip() for l in open(a.list) if l.strip()]
    if a.limit:
        names = names[: a.limit]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, slow = [], 0
    import time
    for i, n in enumerate(names):
        dst = out / (Path(n).stem + ".png")
        if dst.exists():
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, __file__, "--one", n, "--cond-dir", a.cond_dir,
                            "--gt-dir", a.gt_dir, "--out-dir", a.out_dir],
                           capture_output=True, text=True, timeout=None if a.timeout <= 0 else a.timeout + 30)
        dt = time.time() - t0
        if r.returncode != 0 or not r.stdout.strip():
            print(f"  FAILED {n}: {r.stderr.strip()[:120]}", file=sys.stderr)
            continue
        ne, nk = (int(x) for x in r.stdout.split())
        rows.append({"tile": n, "n_edge": ne, "n_keep": nk,
                     "keep_rate": round(nk / ne, 4) if ne else 0.0, "sec": round(dt, 1)})
        if dt > 20:
            slow += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(names)}  slow>{20}s: {slow}", file=sys.stderr, flush=True)
    with open(out / "label_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["tile", "n_edge", "n_keep", "keep_rate", "sec"])
        w.writeheader()
        w.writerows(rows)
    kr = np.array([r["keep_rate"] for r in rows])
    se = np.array([r["sec"] for r in rows])
    print(f"tiles {len(rows)}  keep_rate mean {kr.mean():.4f} median {np.median(kr):.4f}")
    print(f"sec/tile mean {se.mean():.2f} max {se.max():.1f}  (>20s: {slow})")


if __name__ == "__main__":
    main()
