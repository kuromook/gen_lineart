"""Score the 192-tile lineart_family holdout validation (companion to
run_holdout_lineart_family_20260913.sh). Checks whether the 5-tile
consistency_weight sweep verdict (w=0.4/cs2.5 best on near_white_frac,
w=0.2/cs2.5 close behind and best on gt_bsds_f1) holds at 192 tiles from
the same source family, and reports the same condition_only baseline /
vs_condition_f1 / oracle-ceiling axes the shared foundation's contamination
check established are mandatory context for any gt_bsds_f1 number.

The 5 diagnostic tiles (data/diag_valid5.txt) are a subset of this 192 --
their own numbers are re-measured here as an anchor against the historical
round1/round2 values, printed separately (filtered from the same cache, no
recomputation).

Rewritten 2026-09-13 after the first version ran silently for >90 minutes
with no visible progress and was killed to investigate (ptrace-based
profiling was blocked in this sandbox -- see doc/work_log.md). Adding
per-tile timing (the instrumentation that survives below) found the real
cause: `profile_metrics(cond_path)` was called directly on the raw
manga_line conditioning image, which is saved in **inverted**
black-background/white-line convention (to match
`control_v11p_sd15s2_lineart_anime`'s zero-conv embedding -- see
tools/preprocess_manga_line_extraction_condition.py's docstring). Read at
face value (ink = gray<128), that convention makes the "background" the
majority-dark pixels, so measure_lineart_profile.py's `ink_ratio` reads
~99% for ordinary condition tiles -- and for a rough that was nearly blank
to begin with, the manga_line output degenerates to ~100% dark, which
sent this project's own skeletonize/component-analysis code down a
pathological near-fully-filled-mask path that could hang indefinitely (one
tile, `lineart_003_011.jpg`, ran past a 30s per-step probe with everything
else -- load, both edge_maps, bipartite_match_f1, profile_metrics(gt) --
finishing in under 0.05s each). This is not new: Track B's own work_log
records hitting the identical inverted-polarity trap on this same
preprocessor family ("最初ink = gray<128のまま測ってベタ率82〜95%という無意
味な値を出した"). Fix: invert the condition image back to
paper/ink convention (matching GT's own convention) before handing it to
profile_metrics -- confirmed both fast (0.014s vs. hanging) and
semantically correct (near_white_frac 0.99 for a rough that really was
nearly blank, instead of a meaningless ink-dominated reading). Per-tile
timing output is kept below since it's what surfaced this in the first
place.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/evaluation")
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from measure_lineart_profile import profile_metrics  # noqa: E402
from tile_region_manifest_480 import edge_map, bipartite_match_f1  # noqa: E402

TRACK = Path(__file__).resolve().parents[1]
TILE_LIST = TRACK / "data/holdout_lineart_family.txt"
SAMPLES = [l.strip() for l in open(TILE_LIST) if l.strip()]
ANCHOR_SAMPLES = {l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()}
COND_DIR = TRACK / "data/holdout_lineart_family_rough_manga_line"
GT_DIR = TRACK / "data/holdout_lineart_family_gt_line"
OUT_ROOT = TRACK / "results/holdout_lineart_family_20260913"
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0
SLOW_TILE_S = 1.0
CHECKPOINTS = ["w0.2_cs2.5", "w0.4_cs2.5"]
EXTRA_MONTAGE = 6  # additional evenly-spaced non-anchor tiles for visual spread


def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def base(name):
    return name[:-4] if name.endswith(".jpg") else name


def condition_profile(cond_path, tmp_dir):
    """profile_metrics() on the manga_line conditioning image, polarity
    corrected. The file is saved black-background/white-line (inverted, to
    match the ControlNet's pretrained zero-conv embedding convention -- see
    module docstring) -- read at face value, ink_ratio/near_white_frac come
    out both meaningless and, for a near-blank rough, pathologically slow
    to compute (skeletonize on a near-fully-filled mask). Invert back to
    paper/ink convention first."""
    im = Image.open(cond_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    inv_path = tmp_dir / cond_path.name
    Image.eval(im, lambda x: 255 - x).save(inv_path)
    return profile_metrics(inv_path)


def build_tile_cache(tmp_dir):
    """One pass over all 192 tiles: load GT/condition once, compute their
    edge maps, profile_metrics, and the condition-vs-GT match once each --
    reused by every checkpoint and by both the full and anchor reports
    below instead of being recomputed per-checkpoint/per-subset."""
    cache = {}
    t0 = time.time()
    for i, name in enumerate(SAMPLES):
        b = base(name)
        t_tile = time.time()
        gt_path, cond_path = GT_DIR / name, COND_DIR / name
        gt_gray, cond_gray = load_gray(gt_path), load_gray(cond_path)
        gt_edge, cond_edge = edge_map(gt_gray), edge_map(cond_gray)
        f1, precision, recall = bipartite_match_f1(cond_edge, gt_edge, BSDS_TOLERANCE_PX)
        cache[b] = {
            "gt_edge": gt_edge,
            "cond_edge": cond_edge,
            "gt_profile": profile_metrics(gt_path),
            "cond_profile": condition_profile(cond_path, tmp_dir),
            "cond_vs_gt_f1": f1,
            "cond_vs_gt_precision": precision,
            "cond_vs_gt_recall": recall,
        }
        dt = time.time() - t_tile
        if dt > SLOW_TILE_S:
            print(f"[cache {i + 1}/{len(SAMPLES)}] {b}: {dt:.2f}s  <-- SLOW", flush=True)
        elif (i + 1) % 40 == 0:
            print(f"[cache {i + 1}/{len(SAMPLES)}] ... ({time.time() - t0:.1f}s elapsed)", flush=True)
    print(f"cache build done: {time.time() - t0:.1f}s total for {len(SAMPLES)} tiles", flush=True)
    return cache


def score_checkpoint(label, cache, subset=None):
    out_dir = OUT_ROOT / "outputs" / label
    acc = {}
    n = 0
    t0 = time.time()
    names = [name for name in SAMPLES if subset is None or base(name) in subset]
    for i, name in enumerate(names):
        b = base(name)
        out_path = out_dir / f"{b}_out.png"
        if not out_path.exists():
            continue
        t_tile = time.time()
        pred_edge = edge_map(load_gray(out_path))
        m = dict(profile_metrics(out_path))
        m["gt_bsds_f1"] = bipartite_match_f1(pred_edge, cache[b]["gt_edge"], BSDS_TOLERANCE_PX)[0]
        m["vs_condition_f1"] = bipartite_match_f1(pred_edge, cache[b]["cond_edge"], BSDS_TOLERANCE_PX)[0]
        for k, v in m.items():
            acc.setdefault(k, []).append(float(v))
        n += 1
        dt = time.time() - t_tile
        if dt > SLOW_TILE_S:
            print(f"  [{label} {i + 1}/{len(names)}] {b}: {dt:.2f}s  <-- SLOW", flush=True)
    print(f"  {label} scoring done ({n} tiles): {time.time() - t0:.1f}s total", flush=True)
    return {k: float(np.mean(v)) for k, v in acc.items()}, n


def condition_baseline(cache, subset=None):
    names = [base(name) for name in SAMPLES if subset is None or base(name) in subset]
    acc = {}
    recalls, precisions = [], []
    for b in names:
        m = dict(cache[b]["cond_profile"])
        m["gt_bsds_f1"] = cache[b]["cond_vs_gt_f1"]
        m["vs_condition_f1"] = 1.0
        for k, v in m.items():
            acc.setdefault(k, []).append(float(v))
        recalls.append(cache[b]["cond_vs_gt_recall"])
        precisions.append(cache[b]["cond_vs_gt_precision"])
    row = {k: float(np.mean(v)) for k, v in acc.items()}
    mean_recall = float(np.mean(recalls))
    row["oracle_f1"] = 2 * mean_recall / (1.0 + mean_recall)
    row["oracle_recall"] = mean_recall
    row["mean_precision"] = float(np.mean(precisions))
    return row, len(names)


def gt_reference(cache, subset=None):
    names = [base(name) for name in SAMPLES if subset is None or base(name) in subset]
    acc = {}
    for b in names:
        for k, v in cache[b]["gt_profile"].items():
            acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}, len(names)


def montage():
    non_anchor = [base(n) for n in SAMPLES if base(n) not in ANCHOR_SAMPLES]
    step = max(1, len(non_anchor) // EXTRA_MONTAGE)
    picks = list(ANCHOR_SAMPLES) + non_anchor[::step][:EXTRA_MONTAGE]
    picks = sorted(set(picks))[: len(ANCHOR_SAMPLES) + EXTRA_MONTAGE]

    cell_px, label_h = 160, 22
    font = ImageFont.load_default()
    cols = ["cond", "GT"] + CHECKPOINTS
    sheet = Image.new("L", (len(cols) * cell_px, len(picks) * (cell_px + label_h) + label_h), 255)
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(cols):
        draw.text((c * cell_px + 4, 6), label, fill=0, font=font)
    for r, b in enumerate(picks):
        y0 = label_h + r * (cell_px + label_h)
        tag = " (anchor/5tile)" if b in ANCHOR_SAMPLES else ""
        draw.text((4, y0 + 6), f"{b}{tag}", fill=0, font=font)
        paths = [COND_DIR / f"{b}.jpg", GT_DIR / f"{b}.jpg"]
        paths += [OUT_ROOT / "outputs" / ckpt / f"{b}_out.png" for ckpt in CHECKPOINTS]
        for c, p in enumerate(paths):
            if Path(p).exists():
                sheet.paste(Image.open(p).convert("L").resize((cell_px, cell_px)),
                            (c * cell_px, y0 + label_h))
    out = OUT_ROOT / "montage_holdout_lineart_family.png"
    sheet.save(out)
    print(f"\nmontage ({len(picks)} of {len(SAMPLES)} tiles, anchor + evenly-spaced spread): {out}")


def report(subset_name, subset, cache):
    gt, n_gt = gt_reference(cache, subset)
    baseline, n_base = condition_baseline(cache, subset)
    print(f"\n########## {subset_name} (n={n_base}) ##########")
    print(f"condition_only: gt_bsds_f1={baseline['gt_bsds_f1']:.4f}  "
          f"oracle_f1={baseline['oracle_f1']:.4f} (recall={baseline['oracle_recall']:.4f})  "
          f"near_white_frac={baseline['near_white_frac']:.3f} (GT {gt['near_white_frac']:.3f})")
    rows = [{"checkpoint": "condition_only", "n": n_base, **baseline}]
    for ckpt in CHECKPOINTS:
        m, n = score_checkpoint(ckpt, cache, subset)
        if n == 0:
            print(f"  {ckpt}: no outputs found yet")
            continue
        beats = "BEATS baseline" if m["gt_bsds_f1"] > baseline["gt_bsds_f1"] else "below baseline"
        print(f"  {ckpt} (n={n}): gt_bsds_f1={m['gt_bsds_f1']:.4f} ({beats}, "
              f"delta {m['gt_bsds_f1'] - baseline['gt_bsds_f1']:+.4f})  "
              f"near_white_frac={m['near_white_frac']:.3f}  "
              f"vs_condition_f1={m['vs_condition_f1']:.4f}  "
              f"ink_ratio={m['ink_ratio']:.4f} (GT {gt['ink_ratio']:.4f})  "
              f"line_width_p50={m['line_width_p50']:.2f} (GT {gt['line_width_p50']:.2f})")
        rows.append({"checkpoint": ckpt, "n": n, **m})
    return rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_ROOT / "_cond_inverted_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== building tile cache ({len(SAMPLES)} tiles) ===", flush=True)
    cache = build_tile_cache(tmp_dir)
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    all_rows = []
    for subset_name, subset in [
        ("FULL 192-tile lineart_family", None),
        ("5-tile anchor (subset of the 192, historical values: w0.2=0.2354, w0.4=0.2350)", ANCHOR_SAMPLES),
    ]:
        rows = report(subset_name, subset, cache)
        for r in rows:
            r["subset"] = subset_name.split()[0]
        all_rows.extend(rows)

    if not any(r["checkpoint"] != "condition_only" and r.get("n", 0) > 0 for r in all_rows):
        print("\nno model outputs found yet -- inference has not completed", file=sys.stderr)
        return

    csv_path = OUT_ROOT / "scores_summary.csv"
    fields = ["subset", "checkpoint", "n"] + [
        k for k in all_rows[-1] if k not in ("subset", "checkpoint", "n")]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nsaved: {csv_path}")
    montage()


if __name__ == "__main__":
    main()
