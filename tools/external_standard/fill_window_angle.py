"""The fill-window angle, as registered: doc/prereg_fill_windows_20261104.md.

Per axis and per external pool k in {E6, I6}:

    delta_k = AUC13(A1F, E_k) - AUC13(A1L, E_k)     (author = 1, external = 0)

A1F = the author's windows that carry fill, A1L = the author's line windows
from other sources. Both are new; E6 and I6 are the third pool's tiles.

Printed before any bar:
  N1  -- within each work, the A1F/A1L label is permuted over SOURCES
         (a source's windows move together), keeping each work's number of
         F-sources; delta_k again. 1000 draws, band 2.5-97.5%.
Checks: IC1 the IC5 fill split reproduces on the A1 tiles; IC2 A1's AUC13
reproduces the third pool's table; IC3 self AUC; IC4 the new JPEGs carry the
same quantization table as an A2 tile; PC fill_ratio must move; NC1 the
permutation is centred.
"""

import argparse
import collections
import csv
import glob
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_auc import AXES, DISCRETE_AXES, NULL_AXIS, auc, tie_structure  # noqa: E402

MOVE, STILL, PC_MIN, NC_MED = 0.10, 0.05, 0.20, 0.02   # registered
DRAWS = 1000
IC2_TOL = 0.0001
PC_AXIS = "fill_ratio"
# result_third_pool_20261031 / auc_three_arms_*.txt, AUC13 column. Registered as IC2.
KNOWN_AUC13 = {
    "e6": {"ink_ratio": 0.1799, "long_component_ratio": 0.2397, "components_per_1k_ink_px": 0.1816,
           "line_width_p50": 0.9788, "width_consistency": 0.5409, "faint_of_drawn_ratio": 0.7718,
           "faint_near_ink_ratio": 0.2087, "faint_mean_dist_to_ink": 0.8635, "long_line_ratio": 0.7415,
           "orientation_entropy": 0.3989, "deep_black_ratio": 0.0451, "background_ratio": 0.7644},
    "i6": {"ink_ratio": 0.0654, "long_component_ratio": 0.5990, "components_per_1k_ink_px": 0.1184,
           "line_width_p50": 0.8845, "width_consistency": 0.4931, "faint_of_drawn_ratio": 0.9130,
           "faint_near_ink_ratio": 0.1867, "faint_mean_dist_to_ink": 0.9301, "long_line_ratio": 0.6049,
           "orientation_entropy": 0.2945, "deep_black_ratio": 0.0129, "background_ratio": 0.8927},
}
EXT = {"e6": "a3_animagine", "i6": "a3_illustrious"}


def ic1_fill_split():
    """IC5's own sampling: 200 A1 tiles, seed 20261030, per-tile fill share median."""
    files = sorted(glob.glob("data/external_standard/norm_author/*.png"))
    random.Random(20261030).shuffle(files)
    s = []
    for p in files[:200]:
        g = np.asarray(Image.open(p).convert("L"))
        ink = (g < 128).astype(np.uint8)
        d = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        s.append(((ink > 0) & (d >= 8)).sum() / max(int(ink.sum()), 1))
    return float(np.median(s)), float(np.mean(s))


def qtable(path):
    return Image.open(path).quantization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-new", default="results/fill_windows_20261104/profile_fill.csv")
    ap.add_argument("--profile-third", default="results/third_pool_20261031/profile_three_arms.csv")
    ap.add_argument("--manifest", default="data/author_fill_windows_20261104/fill_windows_manifest.csv")
    ap.add_argument("--out", default="results/fill_windows_20261104")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    axes = list(AXES) + [PC_AXIS]
    rows = {k: [] for k in ("a1f", "a1l", "a1", "e6", "i6")}
    for r in csv.DictReader(open(args.profile_new)):
        rows[r["label"]].append(r)
    third = {"a1_author_analogue": "a1", EXT["e6"]: "e6", EXT["i6"]: "i6"}
    for r in csv.DictReader(open(args.profile_third)):
        if r["label"] in third:
            rows[third[r["label"]]].append(r)
    src = {}
    for m in csv.DictReader(open(args.manifest)):
        src[Path(m["tile"]).stem] = (m["work"], m["work"] + "/" + m["source"], m["arm"])

    def vals(k, a):
        return np.array([float(r[a]) for r in rows[k]])

    p("# 塗りを含む窓 —— 登録 doc/prereg_fill_windows_20261104.md")
    p("## 母集団")
    for k in rows:
        p(f"  {k}: タイル {len(rows[k])}")
    lost = [r["sample"] for k in ("a1f", "a1l") for r in rows[k] if Path(r["sample"]).stem not in src]
    p(f"  台帳に無い新しいタイル: {len(lost)}")
    for k in ("a1f", "a1l"):
        fs = [float(m["fill_share"]) for m in csv.DictReader(open(args.manifest)) if m["arm"] == k[-1]]
        p(f"  {k}: 塗りの割合 中央 {np.median(fs):.4f} / 平均 {np.mean(fs):.4f}")

    p("## IC1 IC5 の分け方の再現(A1 の 200枚、中央値 0.2%)")
    med, mean = ic1_fill_split()
    ic1 = abs(med - 0.002) <= 0.0005
    p(f"  中央 {med:.4f} / 平均 {mean:.4f} → {'通った' if ic1 else '通らない'}")

    p("## IC2 A1 の AUC13 が第3のプールの表を再現")
    ic2 = True
    for k in ("e6", "i6"):
        bad = [a for a in AXES if abs(auc(vals("a1", a), vals(k, a)) - KNOWN_AUC13[k][a]) > IC2_TOL + 5e-5]
        ic2 &= not bad
        p(f"  {k}: 外れた軸 {bad if bad else 'なし'}")
    ic3 = all(abs(auc(vals("a1f", a), vals("a1f", a)) - 0.5) < 1e-12 for a in axes)
    p(f"## IC3 自己 AUC: {'通った' if ic3 else '通らない'}")
    new_jpg = sorted(glob.glob("data/author_fill_windows_20261104/tiles_f/*.jpg"))[0]
    a2_jpg = sorted(glob.glob("data/author_digital_a2/tiles/*.jpg"))[0]
    ic4 = qtable(new_jpg) == qtable(a2_jpg)
    p(f"## IC4 量子化表が A2 のタイルと一致: {'通った' if ic4 else '通らない'}")

    # N1: permute arms over sources within each work.
    by_src = collections.defaultdict(lambda: {"a1f": [], "a1l": []})
    for k in ("a1f", "a1l"):
        for i, r in enumerate(rows[k]):
            work, s, _ = src[Path(r["sample"]).stem]
            by_src[(work, s)][k].append(i)
    work_srcs = collections.defaultdict(list)
    for (work, s), d in by_src.items():
        work_srcs[work].append((s, "a1f" if d["a1f"] else "a1l"))
    mats = {k: {a: vals(k, a) for a in axes} for k in rows}
    rng = np.random.default_rng(20261104)
    null = {(k, a): np.empty(args.draws) for k in ("e6", "i6") for a in axes}
    for t in range(args.draws):
        fi, li = [], []
        for work, lst in work_srcs.items():
            n_f = sum(1 for _, arm in lst if arm == "a1f")
            perm = rng.permutation(len(lst))
            for j, idx in enumerate(perm):
                s, arm = lst[idx]
                tiles = [("a1f", i) for i in by_src[(work, s)]["a1f"]] + [("a1l", i) for i in by_src[(work, s)]["a1l"]]
                (fi if j < n_f else li).extend(tiles)
        for a in axes:
            f_v = np.array([mats[k][a][i] for k, i in fi])
            l_v = np.array([mats[k][a][i] for k, i in li])
            for k in ("e6", "i6"):
                null[(k, a)][t] = auc(f_v, mats[k][a]) - auc(l_v, mats[k][a])
    band = {key: (float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for key, v in null.items()}
    p("## 帰無値 N1(作品の中で、コマ単位に腕を入れ替える)")
    p(f"  作品ごとの元コマ(F / L): " + " / ".join(
        f"{w} {sum(1 for _, x in l if x == 'a1f')}/{sum(1 for _, x in l if x == 'a1l')}" for w, l in work_srcs.items()))
    nc1 = True
    for a in axes:
        s = "  ".join(f"{k} 中央 {band[(k, a)][0]:+.4f} 帯 [{band[(k, a)][1]:+.4f}, {band[(k, a)][2]:+.4f}]"
                      for k in ("e6", "i6"))
        nc1 &= all(abs(band[(k, a)][0]) < NC_MED for k in ("e6", "i6"))
        p(f"  {a:26s} {s}")
    p(f"  NC1 中央 |·| < {NC_MED}: {'通った' if nc1 else '通らない'}")

    p("## 主値")
    res = {}
    for a in axes:
        res[a] = {}
        for k in ("e6", "i6"):
            af, al, a1 = (auc(vals(x, a), vals(k, a)) for x in ("a1f", "a1l", "a1"))
            res[a][k] = {"a1f": af, "a1l": al, "a1": a1, "delta": af - al, "gate": al - a1}
    pc = all(abs(res[PC_AXIS][k]["delta"]) >= PC_MIN and res[PC_AXIS][k]["delta"] > 0 for k in ("e6", "i6"))
    p(f"  PC {PC_AXIS}: Δ E6 {res[PC_AXIS]['e6']['delta']:+.4f} / I6 {res[PC_AXIS]['i6']['delta']:+.4f} "
      f"→ {'通った' if pc else '通らない —— 選び分けが効いていない。何も読まない'}")
    ok = ic1 and ic2 and ic3 and ic4 and nc1 and pc and not lost
    p("")
    p("axis                        | E6: A1F    A1L    A1  |  Δ     帯の外 | I6: A1F    A1L    A1  |  Δ     帯の外 | 判定")
    verdict = {}
    for a in AXES:
        cell = []
        outside = {}
        for k in ("e6", "i6"):
            r = res[a][k]
            _, lo, hi = band[(k, a)]
            outside[k] = not (lo <= r["delta"] <= hi)
            cell.append(f"{r['a1f']:.4f} {r['a1l']:.4f} {r['a1']:.4f} | {r['delta']:+.4f} {'外' if outside[k] else '中':>4s}")
        d = [res[a][k]["delta"] for k in ("e6", "i6")]
        if not ok:
            v = "C5 器具 —— 何も読まない"
        elif a == NULL_AXIS:
            v = "帰無値の腕(読まない)"
        elif a in DISCRETE_AXES:
            v = "判定保留(離散。条例4)"
        elif all(abs(x) >= MOVE for x in d) and d[0] * d[1] > 0 and all(outside.values()):
            v = "C2 塗りが入ると動く"
        elif all(abs(x) <= STILL for x in d) and not any(outside.values()):
            v = "C1 塗りが入っても動かない"
        else:
            v = "判定保留"
        verdict[a] = v
        p(f"{a:27s} | {cell[0]} | {cell[1]} | {v}")
    p("")
    p("## 副(判定に使わない)")
    for a in AXES:
        flips = [k for k in ("e6", "i6") if (res[a][k]["a1f"] - 0.5) * (res[a][k]["a1"] - 0.5) < 0]
        p(f"  {a:26s} 門の差 A1L−A1: E6 {res[a]['e6']['gate']:+.4f} / I6 {res[a]['i6']['gate']:+.4f}"
          f"   A1F と A1 で 0.5 の側が違う: {flips if flips else 'なし'}")
    for a in DISCRETE_AXES:
        for k in ("a1f", "a1l"):
            p(f"  {a} {k}: {tie_structure(vals(k, a))}")
    (out / "fill_windows.txt").write_text("\n".join(lines) + "\n")
    json.dump({"ic": {"ic1": [ic1, med, mean], "ic2": ic2, "ic3": ic3, "ic4": ic4, "nc1": nc1, "pc": pc},
               "band": {f"{k}|{a}": v for (k, a), v in band.items()}, "res": res, "verdict": verdict},
              open(out / "fill_windows.json", "w"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
