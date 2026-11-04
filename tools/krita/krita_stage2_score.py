"""Track O stage 2 scorer. Registration: doc/prereg_krita_stage2_20261104.md (+ amendment A1).

    /home/sh1/deepl/lineart/venv/bin/python -I tools/krita/krita_stage2_score.py \
        --outputs results/krita_stage2_20261104 --report results/krita_stage2_20261104/_report

Uses the foundation's instruments as they are (imported from the foundation tree, not copied):
stroke_decomposition.{load_gray, decompose, score_tile, gt_tile_path} and
tile_region_manifest_480.{edge_map, bipartite_match_f1}.

Order is the registration's: IC1-IC4 first, and the Krita arms are not printed if they fail.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FOUNDATION = Path("/home/sh1/deepl/lineart")
sys.path.insert(0, str(FOUNDATION / "tools/evaluation"))
sys.path.insert(0, str(FOUNDATION / "tools/pair_extraction"))
import stroke_decomposition as sd  # noqa: E402
from tile_region_manifest_480 import edge_map, bipartite_match_f1  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAIRS = FOUNDATION / "dataset/pairs_480"
POOLS = {"lineart": PAIRS / "holdout_lineart_family.txt", "housei": PAIRS / "holdout_housei_100.txt"}
COND = Path(sd.COND_DIR_DEFAULT)
HV = Path("/home/sh1/deepl/lineart-controlnet-sdxl-fidelity/results/holdout_validation_20260912")
IC1_DIR = Path("/home/sh1/deepl/lineart-image-prompt/results/ipadapter_variants_20261002/plus_uniform/baseline")
IC1 = {"a_recall": 0.7275228630932893, "b_recall": 0.16005979649798852,
       "c_survival": 0.710625807686322, "neither_ink": 0.19461404797795082,
       "b_px": 4088.2760416666665, "c_px": 13478.390625}
TOL_IC = 0.0005
SEED = 20261104
HEADLINE = ["sketchy_noinv_s1", "smooth_inv_s1"]


def source(tile):
    return tile.rsplit("_", 1)[0] if tile.startswith("lineart") else "_".join(tile.split("_")[:2])


def rough_path(tile):
    for split in ("train", "test"):
        p = PAIRS / split / "rough" / tile
        if p.exists():
            return p
    raise FileNotFoundError(tile)


def dark_px(path, invert=False):
    g = sd.load_gray(path, invert=invert)
    return int((g < 128).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--ic-only", action="store_true", help="run IC1-IC4 and stop (no Krita arm is scored)")
    args = ap.parse_args()
    out_root, rep = ROOT / args.outputs, ROOT / args.report
    rep.mkdir(parents=True, exist_ok=True)

    tiles = {p: [l.strip() for l in open(f) if l.strip()] for p, f in POOLS.items()}
    sd.GT_DIR = Path(sd.GT_DIR_DEFAULT)
    cache = {}
    for pool, ts in tiles.items():
        for t in ts:
            cond = sd.load_gray(COND / t, invert=True)
            gt = sd.load_gray(sd.gt_tile_path(t))
            cache[t] = (cond, gt, sd.decompose(cond, gt), edge_map(gt))

    # matched null partner: one tile j per tile i, same pool, different source image; same j for all arms
    rng = random.Random(SEED)
    partner = {}
    for pool, ts in tiles.items():
        for t in ts:
            partner[t] = rng.choice([u for u in ts if source(u) != source(t)])

    # derived arms on disk: rough itself, rough Otsu, GT
    derived = out_root / "_derived"
    for name in ("rough", "rough_otsu", "gt"):
        (derived / name).mkdir(parents=True, exist_ok=True)
    for ts in tiles.values():
        for t in ts:
            stem = Path(t).stem
            g = np.asarray(Image.open(rough_path(t)).convert("L"))
            Image.fromarray(g).save(derived / "rough" / f"{stem}_out.png")
            _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            Image.fromarray(b).save(derived / "rough_otsu" / f"{stem}_out.png")
            Image.fromarray(cache[t][1]).save(derived / "gt" / f"{stem}_out.png")

    def score_arm(arm_dir, pool, invert=False, path_fn=None, interior=False):
        rows, drops = [], defaultdict(int)
        for t in tiles[pool]:
            path = path_fn(t) if path_fn else arm_dir / f"{Path(t).stem}_out.png"
            if not path.exists():
                drops["missing_output"] += 1
                continue
            cond, gt, parts, gt_edge = cache[t]
            if interior:
                m = np.zeros(cond.shape, bool)
                m[40:-40, 40:-40] = True
                parts = tuple(x & m if x.dtype == bool else x for x in parts)
            r = sd.score_tile(path, cond, gt, parts, invert_pred=invert)
            jp = cache[partner[t]][2]
            if interior:
                jp = tuple(x & m if x.dtype == bool else x for x in jp)
            rn = sd.score_tile(path, *cache[partner[t]][:2], jp, invert_pred=invert)
            out_edge = edge_map(sd.load_gray(path, invert=invert))
            r["f1"] = bipartite_match_f1(out_edge, gt_edge, 2.0)[0]
            r["f1_null"] = bipartite_match_f1(out_edge, cache[partner[t]][3], 2.0)[0]
            r["b_null"] = rn["b_recall"]
            r["dark_px"] = dark_px(path, invert)
            for k, flag in (("out_edge_empty", r["out_px"] == 0), ("A_empty", parts[0].sum() == 0),
                            ("C_empty", parts[2].sum() == 0), ("B_empty", parts[1].sum() == 0)):
                if flag:
                    drops[k] += 1
            ok = parts[0].any() and parts[2].any() and jp[0].any() and jp[2].any()
            r["D"] = r["a_recall"] - r["c_survival"] if ok else np.nan
            r["D_null"] = rn["a_recall"] - rn["c_survival"] if ok else np.nan
            if not ok:
                drops["D_undefined"] += 1
            r.update(tile=t, source=source(t))
            rows.append(r)
        return rows, dict(drops)

    def mean(rows, k):
        v = [r[k] for r in rows if not np.isnan(r[k])]
        return float(np.mean(v)) if v else float("nan")

    def summary(rows):
        s = {k: mean(rows, k) for k in ("D", "D_null", "a_recall", "b_recall", "c_survival", "neither_ink",
                                        "b_null", "f1", "f1_null", "out_px", "dark_px", "b_px", "c_px")}
        s["Dp"] = s["D"] - s["D_null"]
        s["n"] = len(rows)
        s["n_D"] = sum(not np.isnan(r["D"]) for r in rows)
        return s

    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    # ---- instrument checks
    emit("## Instrument checks")
    ic_ok = {"lineart": True, "housei": True}
    rows, _ = score_arm(IC1_DIR, "lineart")
    s = summary(rows)
    for k, v in IC1.items():
        d = abs(s[k] - v)
        good = d <= (TOL_IC if k not in ("b_px", "c_px") else 0.01)
        ic_ok["lineart"] &= good
        emit(f"IC1 lineart {k}: got {s[k]:.6f} want {v:.6f} |d|={d:.6f} {'OK' if good else 'FAIL'}")
    rows, _ = score_arm(HV / "outputs/bare_cs2.0", "housei")
    f1h = mean(rows, "f1")
    good = abs(f1h - 0.1564) <= TOL_IC
    ic_ok["housei"] &= good
    emit(f"IC2 housei f1 bare_cs2.0: got {f1h:.6f} want 0.1564 |d|={abs(f1h - 0.1564):.6f} {'OK' if good else 'FAIL'}")
    rows, _ = score_arm(None, "lineart", invert=True, path_fn=lambda t: COND / t)
    f1c = mean(rows, "f1")
    ic3 = abs(f1c - 0.3231) <= TOL_IC
    emit(f"IC3 lineart f1 condition: got {f1c:.6f} want 0.3231 |d|={abs(f1c - 0.3231):.6f} "
         f"{'OK' if ic3 else 'FAIL -> f1 columns not read'}")
    for pool in tiles:
        rg, _ = score_arm(derived / "gt", pool)
        rc, _ = score_arm(None, pool, invert=True, path_fn=lambda t: COND / t)
        sg, sc = summary(rg), summary(rc)
        good = (abs(sg["D"] - 1) < 1e-9 and sg["c_survival"] == 0 and sg["neither_ink"] == 0
                and abs(sc["D"]) < 1e-9 and sc["neither_ink"] == 0)
        ic_ok[pool] &= good
        emit(f"IC4 {pool}: gt D={sg['D']:.6f} c={sg['c_survival']:.6f} neither={sg['neither_ink']:.6f} | "
             f"condition D={sc['D']:.6f} neither={sc['neither_ink']:.6f} {'OK' if good else 'FAIL'}")
    emit(f"IC verdict: {ic_ok}  (f1 columns readable: {ic3})")
    if args.ic_only:
        (rep / "ic.txt").write_text("\n".join(lines) + "\n")
        return

    # ---- arms
    arm_dirs = [("condition (lineart_coarse)", None, True, lambda t: COND / t),
                ("gt", derived / "gt", False, None),
                ("rough", derived / "rough", False, None),
                ("rough_otsu", derived / "rough_otsu", False, None)]
    krita = sorted(d.name for d in out_root.iterdir() if d.is_dir() and not d.name.startswith("_"))
    krita = HEADLINE + [k for k in krita if k not in HEADLINE]
    arm_dirs += [(k, out_root / k, False, None) for k in krita]

    per_tile, table = [], []
    for pool in tiles:
        emit(f"\n## Pool {pool} ({len(tiles[pool])} tiles) -- {'READ' if ic_ok[pool] else 'NOT READ (IC failed)'}")
        hdr = (f"{'arm':30s} {'n':>4s} {'n_D':>4s} {'D':>7s} {'D_null':>7s} {'Dprime':>7s} {'a':>6s} {'c':>6s} "
               f"{'b':>6s} {'b_null':>6s} {'neither':>7s} {'f1':>6s} {'f1sig':>7s} {'dark_px':>8s} drops")
        emit(hdr)
        for name, d, inv, fn in arm_dirs:
            rows, drops = score_arm(d, pool, invert=inv, path_fn=fn)
            if not rows:
                emit(f"{name:30s} no rows {drops}")
                continue
            s = summary(rows)
            s.update(arm=name, pool=pool, drops=drops)
            table.append(s)
            for r in rows:
                per_tile.append({"arm": name, "pool": pool, **{k: r[k] for k in (
                    "tile", "source", "D", "D_null", "a_recall", "b_recall", "c_survival", "neither_ink",
                    "b_null", "f1", "f1_null", "out_px", "dark_px")}})
            if not ic_ok[pool] and name not in ("condition (lineart_coarse)", "gt"):
                continue
            emit(f"{name:30s} {s['n']:4d} {s['n_D']:4d} {s['D']:7.4f} {s['D_null']:7.4f} {s['Dp']:7.4f} "
                 f"{s['a_recall']:6.4f} {s['c_survival']:6.4f} {s['b_recall']:6.4f} {s['b_null']:6.4f} "
                 f"{s['neither_ink']:7.4f} {s['f1']:6.4f} {s['f1'] - s['f1_null']:7.4f} {s['dark_px']:8.0f} {drops}")
            # per source D'
            bys = defaultdict(list)
            for r in rows:
                bys[r["source"]].append(r)
            parts = []
            pos = 0
            for src in sorted(bys):
                ss = summary(bys[src])
                parts.append(f"{src.split('_')[-1]}:{ss['Dp']:+.3f}")
                pos += ss["Dp"] > 0
            s["sources_pos"] = pos
            s["sources_n"] = len(bys)
            emit(f"{'':30s}   per-source D': {' '.join(parts)}  (positive {pos}/{len(bys)})")

    # NC4: interior-only D for sketchy_noinv_s1
    emit("\n## NC4 (outer 40 px excluded), sketchy_noinv_s1")
    for pool in tiles:
        a, _ = score_arm(out_root / "sketchy_noinv_s1", pool)
        b, _ = score_arm(out_root / "sketchy_noinv_s1", pool, interior=True)
        sa, sb = summary(a), summary(b)
        emit(f"{pool}: D full={sa['D']:.4f} interior={sb['D']:.4f} |diff|={abs(sa['D'] - sb['D']):.4f} (expected <= 0.02)")

    with open(rep / "summary.csv", "w", newline="") as f:
        cols = ["pool", "arm", "n", "n_D", "D", "D_null", "Dp", "a_recall", "b_recall", "c_survival", "neither_ink",
                "b_null", "f1", "f1_null", "out_px", "dark_px", "b_px", "c_px", "sources_pos", "sources_n", "drops"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    with open(rep / "per_tile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_tile[0]))
        w.writeheader()
        w.writerows(per_tile)
    (rep / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {rep}")


if __name__ == "__main__":
    main()
