#!/usr/bin/env python
"""How large an image can stroke tokenization handle on this machine?

Tokenizes GT line panels at NATIVE resolution (no 480 resize) with the same
unit as tokenize_corpus (1px skeleton split at crossing-number junctions,
min 8px), and records per image: megapixels, token count, wall time per stage,
peak RSS. `segments()` in stroke_churn loops `labels == k` per component, which
is O(components x pixels); this probe uses an equivalent sort-based grouping
and also times the original on small images to show the difference.
"""
import argparse, csv, random, resource, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from stroke_churn import junction_mask, thin_to_1px, segments as segments_orig  # noqa

def segments_fast(skel, min_len):
    j = cv2.dilate(junction_mask(skel).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    n, lab = cv2.connectedComponents((skel & ~j).astype(np.uint8), connectivity=8)
    flat = lab.ravel(); idx = np.flatnonzero(flat); l = flat[idx]
    o = np.argsort(l, kind="stable"); idx, l = idx[o], l[o]
    cuts = np.flatnonzero(np.diff(l)) + 1
    starts = np.r_[0, cuts]; ends = np.r_[cuts, len(l)]
    keep = (ends - starts) >= min_len
    return int(keep.sum()), (ends - starts)[keep]

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/home/sh1/deepl/lineart/dataset/regions_clip_pairs_v3_koma_panels_20260826/manifest.csv")
    p.add_argument("--root", default="/home/sh1/deepl/lineart")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--image", default="")
    p.add_argument("--compare-orig", action="store_true")
    a = p.parse_args()
    if a.image:
        paths = [(a.image, "")]
    else:
        rows = list(csv.DictReader(open(a.manifest)))
        rows.sort(key=lambda r: int(r["native_width"]) * int(r["native_height"]))
        random.seed(0)
        # stratified over size: evenly spaced ranks, so the tail (max) is included
        ks = sorted({int(i * (len(rows) - 1) / (a.n - 1)) for i in range(a.n)})
        paths = [(str(Path(a.root) / rows[k]["native_line_path"]), rows[k]["name"]) for k in ks]
    w = csv.writer(sys.stdout)
    w.writerow(["name", "w", "h", "mp", "ink_frac", "tokens", "tok_len_med", "t_load", "t_skel", "t_seg", "t_orig", "rss_mb"])
    for path, name in paths:
        t0 = time.time(); g = cv2.imread(path, cv2.IMREAD_GRAYSCALE); t1 = time.time()
        ink = g < 128; sk = thin_to_1px(ink); t2 = time.time()
        n, lens = segments_fast(sk, 8); t3 = time.time()
        to = ""
        if a.compare_orig and g.size <= 2.5e6:
            s = time.time(); segments_orig(sk, 8); to = round(time.time() - s, 2)
        w.writerow([name or Path(path).name, g.shape[1], g.shape[0], round(g.size / 1e6, 2), round(float(ink.mean()), 4),
                    n, int(np.median(lens)) if n else 0, round(t1 - t0, 2), round(t2 - t1, 2), round(t3 - t2, 2), to, int(rss_mb())])
        sys.stdout.flush()

if __name__ == "__main__":
    main()
