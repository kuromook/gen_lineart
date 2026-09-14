"""Track D: do individual strokes settle during training, or drift?

Motivation (user's framing, 2026-09-15): each stroke in line art carries
meaning, and a viewer reads line art through the *relations* between
strokes -- closer to tokens than to pixels. Every loss and metric in this
project sees only per-pixel position, intensity, gradient and width. If
so, training should move steadily along what those losses see (paper,
width, fill -- which the snapshot probe confirmed) while *which* strokes
exist should wander without direction.

This measures the second half directly on the snapshot probe's outputs,
no training. Units are skeleton segments (ink skeleton split at junction
pixels, segments shorter than --min-seg-px dropped) -- deliberately not
connected components: Track C found whole-component keep/drop merges many
strokes at junctions and destroys partial credit
(`../lineart-stroke-selection/results/oracle_visual_check_20260913/build_oracle.py`).

The skeleton is skimage's 1px thinning, NOT the project's shared
`evaluate_stroke_stability.skeletonize()`: that erode/open construction
leaves 2px-thick runs, and on 20 GT tiles it made 64% of skeleton pixels
look like junctions, so only 14% of stroke length survived as segments
and 1.7% as segments >= 30px (found 2026-09-15 by rendering the segments,
before any churn number was used). Its connected-component metrics
(long_component_ratio, components per 1k ink px) do not depend on 1px
width and are unaffected; anything junction-based must not use it.

Per tile and snapshot:
- a GT segment is *present* when >= --coverage of its skeleton pixels lie
  within --tol px of the output's skeleton;
- output skeleton pixels farther than --tol from GT's skeleton are
  *invented*.

Between consecutive snapshots:
- GT stroke churn: length of GT segments gained (absent -> present) and
  lost (present -> absent); net = gained - lost.
- Independence reference: with each snapshot's own presence rate held
  fixed, the churn expected if every segment's presence were redrawn at
  random, r_a(1-r_b) + (1-r_a) r_b per unit length. persistence =
  1 - observed/expected: ~0 means strokes are effectively redrawn at
  random each time (drift); near 1 means the same strokes stay.
- Invented-stroke persistence: fraction of invented skeleton length at the
  earlier snapshot still present (within --tol) at the later one.

Noise floor: `step_10580` and `final` hold identical LoRA tensors and were
inferred with identical seeds, so their churn is inference nondeterminism
only.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize as thin_to_1px

TRACK_A = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine")
PROBE_TAG = "controlnet_lora_manga_consistency_w0.2_snapshot_probe_20260914"
IMAGE_SIZE = 480
INK_THRESHOLD = 128
MIN_PX = 20
# Long GT segments are contours and major strokes; short ones are mostly
# detail and hatching. If only short strokes churn, "drift at the stroke
# level" would be a detail-level effect, not a structural one.
LEN_BINS = {"<30px": (0, 30), "30-100px": (30, 100), ">=100px": (100, 10**9)}


def load_gray(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    if g.shape != (IMAGE_SIZE, IMAGE_SIZE):
        g = cv2.resize(g, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return g


def ink_skeleton(gray):
    ink = gray < INK_THRESHOLD
    n_ink = int(ink.sum())
    if n_ink < MIN_PX or ink.size - n_ink < MIN_PX:
        return np.zeros_like(ink)
    return thin_to_1px(ink)


def junction_mask(skel):
    """Branch points by crossing number: count 0->1 transitions walking the
    8-neighbour ring. >= 3 is a junction. A plain neighbour count misfires on
    1px staircase diagonals."""
    s = np.pad(skel.astype(np.uint8), 1)
    h, w = skel.shape
    offs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    ring = [s[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc] for dr, dc in offs]
    transitions = sum(((ring[k] == 0) & (ring[(k + 1) % 8] == 1)).astype(np.uint8) for k in range(8))
    return skel & (transitions >= 3)


def segments(skel, min_len):
    """Split a 1px skeleton at junctions and return a list of (rows, cols)
    index arrays, one per segment. The junction's 3x3 neighbourhood is
    removed too: dropping only the centre pixel leaves the arms diagonally
    adjacent, so 8-connected labelling would merge them back."""
    junctions = cv2.dilate(junction_mask(skel).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    branches = skel & ~junctions
    n, labels = cv2.connectedComponents(branches.astype(np.uint8), connectivity=8)
    out = []
    for k in range(1, n):
        rr, cc = np.nonzero(labels == k)
        if len(rr) >= min_len:
            out.append((rr, cc))
    return out


def dist_to(skel):
    if not skel.any():
        return np.full(skel.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform((~skel).astype(np.uint8), cv2.DIST_L2, 3)


def discover_snapshots(results_root, include_final):
    snaps = []
    for d in Path(results_root).iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            snaps.append((d.name, int(d.name.split("_")[1])))
    snaps.sort(key=lambda s: s[1])
    if include_final and (Path(results_root) / "final").is_dir():
        snaps.append(("final", snaps[-1][1] if snaps else 0))
    return snaps


def analyze_tile(name, gt_dir, results_root, labels, tol, coverage, min_seg):
    gt_skel = ink_skeleton(load_gray(Path(gt_dir) / name))
    gt_segs = segments(gt_skel, min_seg)
    gt_len = np.array([len(r) for r, _ in gt_segs], dtype=float)
    gt_dist = dist_to(gt_skel)

    present = np.zeros((len(gt_segs), len(labels)), dtype=bool)
    invented = []
    out_len = []
    for j, label in enumerate(labels):
        out_skel = ink_skeleton(load_gray(Path(results_root) / label / f"{name[:-4]}_out.png"))
        od = dist_to(out_skel)
        for i, (rr, cc) in enumerate(gt_segs):
            present[i, j] = float((od[rr, cc] <= tol).mean()) >= coverage
        invented.append(out_skel & (gt_dist > tol))
        out_len.append(int(out_skel.sum()))
    return gt_len, present, invented, out_len


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default=str(TRACK_A / "results" / PROBE_TAG))
    parser.add_argument("--gt-dir", default=str(TRACK_A / "data/holdout_lineart_family_gt_line"))
    parser.add_argument("--sample-list", default=str(TRACK_A / "data/holdout_lineart_family.txt"))
    parser.add_argument("--tol", type=float, default=3.0)
    parser.add_argument("--coverage", type=float, default=0.5)
    parser.add_argument("--min-seg-px", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--no-final", action="store_true", help="skip the step_10580 -> final noise-floor pair")
    parser.add_argument("--output-dir", default="results/stroke_churn_20260915")
    args = parser.parse_args()

    names = [l.strip() for l in open(args.sample_list) if l.strip()]
    if args.limit:
        names = names[: args.limit]
    snaps = discover_snapshots(args.results_root, not args.no_final)
    if args.labels:
        snaps = [s for s in snaps if s[0] in args.labels]
    labels = [s[0] for s in snaps]
    print(f"{len(names)} tiles x snapshots {labels} (tol={args.tol}px, coverage={args.coverage}, min_seg={args.min_seg_px}px)",
          file=sys.stderr, flush=True)

    n_pairs = len(labels) - 1
    tot = {"gt_len": 0.0}
    snap_present_len = np.zeros(len(labels))
    snap_out_len = np.zeros(len(labels))
    snap_inv_len = np.zeros(len(labels))
    pair = {k: np.zeros(n_pairs) for k in ("gained", "lost", "expected", "inv_a", "inv_kept")}
    flips_len = {}
    ever_len = always_len = 0.0
    bin_stat = {b: {"len": 0.0, "present": 0.0, "ever": 0.0, "always": 0.0, "flip2": 0.0, "churn": 0.0} for b in LEN_BINS}

    for t_i, name in enumerate(names):
        gt_len, present, invented, out_len = analyze_tile(
            name, args.gt_dir, args.results_root, labels, args.tol, args.coverage, args.min_seg_px)
        L = gt_len.sum()
        tot["gt_len"] += L
        snap_present_len += (present * gt_len[:, None]).sum(axis=0)
        snap_out_len += np.array(out_len)
        snap_inv_len += np.array([m.sum() for m in invented])
        for k in range(n_pairs):
            a, b = present[:, k], present[:, k + 1]
            pair["gained"][k] += gt_len[~a & b].sum()
            pair["lost"][k] += gt_len[a & ~b].sum()
            if L > 0:
                ra, rb = gt_len[a].sum() / L, gt_len[b].sum() / L
                pair["expected"][k] += L * (ra * (1 - rb) + (1 - ra) * rb)
            inv_a, inv_b = invented[k], invented[k + 1]
            pair["inv_a"][k] += inv_a.sum()
            if inv_a.any():
                pair["inv_kept"][k] += (dist_to(inv_b)[inv_a] <= args.tol).sum()
        core = [j for j, l in enumerate(labels) if l != "final"]
        if len(core) >= 2 and len(gt_len):
            p = present[:, core]
            n_flip = (p[:, 1:] != p[:, :-1]).sum(axis=1)
            for f, ln in zip(n_flip, gt_len):
                flips_len[int(f)] = flips_len.get(int(f), 0.0) + ln
            ever_len += gt_len[p.any(axis=1)].sum()
            always_len += gt_len[p.all(axis=1)].sum()
            for b, (lo, hi) in LEN_BINS.items():
                sel = (gt_len >= lo) & (gt_len < hi)
                if not sel.any():
                    continue
                ln, ps = gt_len[sel], p[sel]
                bs = bin_stat[b]
                bs["len"] += ln.sum()
                bs["present"] += (ln[:, None] * ps).sum()
                bs["ever"] += ln[ps.any(axis=1)].sum()
                bs["always"] += ln[ps.all(axis=1)].sum()
                bs["flip2"] += ln[n_flip[sel] >= 2].sum()
                bs["churn"] += (ln[:, None] * (ps[:, 1:] != ps[:, :-1])).sum()
        if (t_i + 1) % 25 == 0:
            print(f"  {t_i + 1}/{len(names)} tiles", file=sys.stderr, flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    G = tot["gt_len"]

    snap_rows = []
    for j, (label, step) in enumerate(snaps):
        snap_rows.append({
            "label": label, "step": step,
            "gt_len_present_frac": snap_present_len[j] / G,
            "invented_len_frac": snap_inv_len[j] / max(snap_out_len[j], 1),
            "output_skeleton_len": int(snap_out_len[j]),
        })
    pair_rows = []
    for k in range(n_pairs):
        churn = pair["gained"][k] + pair["lost"][k]
        exp = pair["expected"][k]
        pair_rows.append({
            "pair": f"{labels[k]}->{labels[k + 1]}",
            "gained_frac": pair["gained"][k] / G, "lost_frac": pair["lost"][k] / G,
            "net_frac": (pair["gained"][k] - pair["lost"][k]) / G,
            "churn_frac": churn / G, "expected_random_churn_frac": exp / G,
            "persistence": 1 - churn / exp if exp > 0 else float("nan"),
            "invented_kept_frac": pair["inv_kept"][k] / max(pair["inv_a"][k], 1),
        })

    for fname, rows in (("per_snapshot.csv", snap_rows), ("per_pair.csv", pair_rows)):
        with open(out_dir / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"\n{len(names)} tiles, total GT stroke length {G:.0f}px (tol {args.tol}px, coverage {args.coverage})")
    print("\nsnapshot       GT length present   invented share of output")
    for r in snap_rows:
        print(f"{r['label']:<13}{r['gt_len_present_frac']:>14.3f}{r['invented_len_frac']:>22.3f}")
    print("\npair                      gained   lost    net   churn  random-churn  persistence  invented kept")
    for r in pair_rows:
        print(f"{r['pair']:<24}{r['gained_frac']:>8.3f}{r['lost_frac']:>7.3f}{r['net_frac']:>+7.3f}{r['churn_frac']:>8.3f}"
              f"{r['expected_random_churn_frac']:>13.3f}{r['persistence']:>13.3f}{r['invented_kept_frac']:>15.3f}")
    if flips_len:
        core_n = len([l for l in labels if l != "final"])
        print(f"\nacross {core_n} training snapshots: GT length ever present {ever_len / G:.3f}, present at every snapshot {always_len / G:.3f}")
        print("GT length by number of present/absent flips: " +
              ", ".join(f"{f} flips {v / G:.3f}" for f, v in sorted(flips_len.items())))
        bin_rows = []
        print("\nby GT segment length   share of GT  mean present  ever  always  >=2 flips  churn/step")
        for b, bs in bin_stat.items():
            if bs["len"] == 0:
                continue
            row = {
                "bin": b, "share_of_gt_len": bs["len"] / G,
                "mean_present": bs["present"] / (bs["len"] * core_n),
                "ever_present": bs["ever"] / bs["len"], "always_present": bs["always"] / bs["len"],
                "flips_ge2": bs["flip2"] / bs["len"], "churn_per_step": bs["churn"] / (bs["len"] * (core_n - 1)),
            }
            bin_rows.append(row)
            print(f"{b:<22}{row['share_of_gt_len']:>11.3f}{row['mean_present']:>14.3f}{row['ever_present']:>6.3f}"
                  f"{row['always_present']:>8.3f}{row['flips_ge2']:>11.3f}{row['churn_per_step']:>12.3f}")
        with open(out_dir / "per_length_bin.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bin_rows[0]))
            w.writeheader()
            w.writerows(bin_rows)
    print(f"\nsaved: {out_dir / 'per_snapshot.csv'}, {out_dir / 'per_pair.csv'}")


if __name__ == "__main__":
    main()
