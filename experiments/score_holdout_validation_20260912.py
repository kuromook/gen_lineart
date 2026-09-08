"""Score the held-out validation, keeping the two groups apart and checking
the five historical tiles as an anchor.

The question this answers: the track's headline -- bare lineart_anime at
1024/cs2.5 beats every trained model, including our own 50h fine-tune by
0.098 gt_bsds_f1 -- was measured on five tiles. Does it survive 192 tiles of
the same source family, and does it survive 100 tiles of a different source?

Three readings are printed, in the order they should be trusted:

1. **anchor** -- the five diag tiles, re-measured here. They sit inside
   group A, so if these do not land near the historical values (bare cs2.5
   0.2582, ft cs2.0 0.1602) then something in the staging or preprocessing
   differs from the original runs and nothing below should be read.
2. **group A (192, same family)** -- the actual scale-up. If the ranking
   here matches the five-tile ranking, the headline holds.
3. **group B (100, housei)** -- a different source. A different ranking here
   is not a contradiction; it is a statement about generalisation, and
   should be reported as such rather than averaged into A.

As everywhere in this track, gt_bsds_f1 is printed with the paper axes: it
cannot tell line art from a grey wash or from a blank page on its own.
"""

import csv
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/evaluation")
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from measure_lineart_profile import profile_metrics  # noqa: E402
from tile_region_manifest_480 import edge_map, bipartite_match_f1  # noqa: E402
from score_resolution_sweep_20260906 import paper_metrics  # noqa: E402

TRACK = Path(__file__).resolve().parents[1]
SHARED = Path("/home/sh1/deepl/lineart/dataset/pairs_480")
ROOT = TRACK / "results/holdout_validation_20260912"
OUT_ROOT = ROOT / "outputs"
IMAGE_SIZE = 480
BSDS_TOLERANCE_PX = 2.0

# Historical five-tile values these runs must reproduce on the same five
# tiles (results/controlnet_lora_sdxl_anime_1024_20260907_eval/scores.csv).
HISTORICAL = {"bare_cs2.0": 0.2568, "bare_cs2.5": 0.2582,
              "bare_cs3.0": 0.2577, "ft_cs2.0": 0.1602}


def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))


def gt_path(tile):
    for split in ("train", "test"):
        p = SHARED / split / "line" / tile
        if p.exists():
            return p
    raise FileNotFoundError(f"no GT line tile for {tile}")


def score_tile(out_png, tile):
    gray = load_gray(out_png)
    m = dict(profile_metrics(out_png), **paper_metrics(gray))
    m["gt_bsds_f1"] = bipartite_match_f1(
        edge_map(gray), edge_map(load_gray(gt_path(tile))), BSDS_TOLERANCE_PX
    )[0]
    return m


AXES = ["gt_bsds_f1", "ink_ratio", "line_width_p50", "near_white_frac",
        "midtone_frac", "orientation_entropy"]


def summarise(rows):
    """Mean plus a spread, because a mean over 192 tiles can hide a bimodal
    outcome that a mean over 5 could not."""
    out = {}
    for a in AXES:
        vals = [r[a] for r in rows]
        out[a] = float(np.mean(vals))
        if a == "gt_bsds_f1":
            out["f1_median"] = float(statistics.median(vals))
            out["f1_stdev"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
    return out


def table(title, per_label, note=""):
    print(f"\n=== {title} ===")
    if note:
        print(f"    {note}")
    print(f"{'config':12}{'n':>5}{'f1_mean':>10}{'f1_med':>9}{'f1_sd':>8}"
          f"{'ink':>9}{'line_w':>9}{'near_wht':>10}{'midtone':>9}")
    for label in sorted(per_label, key=lambda k: -per_label[k]["gt_bsds_f1"]):
        s = per_label[label]
        print(f"{label:12}{s['n']:5d}{s['gt_bsds_f1']:10.4f}{s['f1_median']:9.4f}"
              f"{s['f1_stdev']:8.4f}{s['ink_ratio']:9.4f}{s['line_width_p50']:9.2f}"
              f"{s['near_white_frac']*100:9.1f}%{s['midtone_frac']*100:8.1f}%")


def main():
    groups = {
        "A_lineart_family": [l.strip() for l in
                             open(TRACK / "dataset/pairs_480/holdout_lineart_family.txt") if l.strip()],
        "B_housei": [l.strip() for l in
                     open(TRACK / "dataset/pairs_480/holdout_housei_100.txt") if l.strip()],
    }
    anchor = {l.strip() for l in open(TRACK / "data/diag_valid5.txt") if l.strip()}
    labels = sorted(d.name for d in OUT_ROOT.iterdir()
                    if d.is_dir() and (d / ".complete").exists())
    if not labels:
        print("no completed cells", file=sys.stderr)
        return

    rows = []
    for label in labels:
        for group, tiles in groups.items():
            for tile in tiles:
                png = OUT_ROOT / label / f"{tile[:-4]}_out.png"
                if not png.exists():
                    continue
                rows.append({"config": label, "group": group, "tile": tile,
                             **score_tile(png, tile)})

    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "scores_per_tile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "group", "tile"] + AXES)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in w.fieldnames})

    def by_label(subset):
        out = {}
        for label in labels:
            sel = [r for r in subset if r["config"] == label]
            if sel:
                out[label] = dict(summarise(sel), n=len(sel))
        return out

    anchor_rows = [r for r in rows if r["tile"] in anchor]
    table("1. ANCHOR: the five diag tiles, re-measured", by_label(anchor_rows),
          "must land near the historical five-tile values below")
    print("\n    historical (results/controlnet_lora_sdxl_anime_1024_20260907_eval):")
    for k, v in HISTORICAL.items():
        now = by_label(anchor_rows).get(k, {}).get("gt_bsds_f1")
        if now is not None:
            print(f"      {k:12} now {now:.4f}  historical {v:.4f}  delta {now - v:+.4f}")

    a_rows = [r for r in rows if r["group"] == "A_lineart_family"]
    b_rows = [r for r in rows if r["group"] == "B_housei"]
    table("2. GROUP A: 192 tiles, same source family as the diag five",
          by_label(a_rows), "this is the scale-up -- does the five-tile ranking hold?")
    table("3. GROUP B: 100 housei tiles, a different source",
          by_label(b_rows), "generalisation, not a contradiction if it differs")

    for name, subset in (("GROUP A", a_rows), ("GROUP B", b_rows)):
        best_bare = max((by_label(subset)[l], l) for l in by_label(subset) if l.startswith("bare"))
        ft = by_label(subset).get("ft_cs2.0")
        if ft:
            delta = ft["gt_bsds_f1"] - best_bare[0]["gt_bsds_f1"]
            verdict = ("fine-tuning helps" if delta > 0
                       else "fine-tuning does NOT beat the bare ControlNet")
            print(f"\n{name} verdict: ft_cs2.0 {ft['gt_bsds_f1']:.4f} vs "
                  f"{best_bare[1]} {best_bare[0]['gt_bsds_f1']:.4f} "
                  f"-> delta {delta:+.4f}, {verdict}")

    with open(ROOT / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "config", "n", "f1_mean",
                                          "f1_median", "f1_stdev"] + AXES[1:])
        w.writeheader()
        for gname, subset in (("anchor_diag5", anchor_rows),
                              ("A_lineart_family", a_rows), ("B_housei", b_rows)):
            for label, s in by_label(subset).items():
                w.writerow({"group": gname, "config": label, "n": s["n"],
                            "f1_mean": round(s["gt_bsds_f1"], 4),
                            "f1_median": round(s["f1_median"], 4),
                            "f1_stdev": round(s["f1_stdev"], 4),
                            **{a: round(s[a], 4) for a in AXES[1:]}})
    print(f"\nsaved: {ROOT/'scores.csv'} and {ROOT/'scores_per_tile.csv'}")

    montage(labels, groups)


def montage(labels, groups):
    """Twelve tiles spanning both groups -- a mean over 292 tiles is exactly
    where a grey wash or a blank page would hide, so the sheet is not
    optional."""
    picks = groups["A_lineart_family"][:8] + groups["B_housei"][:4]
    cell, lab_h = 190, 22
    font = ImageFont.load_default()
    cols = ["cond", "GT"] + labels
    sheet = Image.new("L", (len(cols) * cell, len(picks) * (cell + lab_h) + lab_h), 255)
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        d.text((c * cell + 4, 5), name, fill=0, font=font)
    for r, tile in enumerate(picks):
        y = lab_h + r * (cell + lab_h)
        d.text((4, y + 5), tile[:-4], fill=0, font=font)
        paths = [ROOT / "conditioning" / tile, gt_path(tile)]
        paths += [OUT_ROOT / l / f"{tile[:-4]}_out.png" for l in labels]
        for c, p in enumerate(paths):
            if Path(p).exists():
                sheet.paste(Image.open(p).convert("L").resize((cell, cell)),
                            (c * cell, y + lab_h))
    out = ROOT / "montage_holdout.png"
    sheet.save(out)
    print(f"montage: {out}")


if __name__ == "__main__":
    main()
