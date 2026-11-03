"""The long-line width angle, as registered: doc/prereg_long_line_width_20261104.md.

long_line_ratio_skel: the same Hough rule as long_line_ratio, applied to the
1px skeleton of the ink, as a share of skeleton pixels. The original axis is
not touched; the new one is a separate column.

Steps (each prints before the next):
  IC1  synthetic lines: straight >= 0.95 at every width; wavy spread <= 0.10
       and max <= 0.30 over widths 1..9. If this fails nothing else runs.
  measure the new axis on the third pool's four pools (A1, A2, E6, I6).
  IC2  the original axis reproduces AUC23 E6 0.7532 / I6 0.6027 (+-0.0001).
  IC3  self AUC 0.5.
  N1   per prompt, E6 and I6 tiles swap sides with p = 1/2; G_skel again.
  NC1  Spearman with line_width_p50, original vs skeleton.
  NC2  N1 median |.| < 0.02.
  C1/C2 and the side note.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pair_extraction"))
sys.path.insert(0, str(ROOT / "tools" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tile_region_manifest_480 import long_line_ratio  # noqa: E402
from measure_lineart_profile import THRESHOLD, IMAGE_SIZE  # noqa: E402
from compute_auc import auc  # noqa: E402

MIN_FRACTION = 0.08
G_ORIG_KNOWN = 0.7532 - 0.6027
KNOWN = {"e6": 0.7532, "i6": 0.6027}
IC2_TOL = 0.0001 + 5e-5   # the table is printed to 4 places
DRAWS = 1000

POOLS = {
    "a1": ("a1_author_analogue", "data/external_standard/norm_author"),
    "a2": ("a2_author_digital", "data/author_digital_a2/norm"),
    "e6": ("a3_animagine", "data/external_standard/norm_animagine_xl_31_seed2"),
    "i6": ("a3_illustrious", "data/external_standard/norm_illustrious_xl_v01_seed2"),
}


def long_line_ratio_skel(ink, tile=IMAGE_SIZE, min_length_fraction=MIN_FRACTION):
    skel = skeletonize(ink)
    if skel.sum() < 20:
        return 0.0, True
    segments = cv2.HoughLinesP(skel.astype(np.uint8) * 255, 1, np.pi / 180, threshold=80,
                               minLineLength=int(tile * min_length_fraction), maxLineGap=6)
    if segments is None:
        return 0.0, False
    canvas = np.zeros(ink.shape, np.uint8)
    for x0, y0, x1, y1 in segments[:, 0]:
        cv2.line(canvas, (x0, y0), (x1, y1), 1, 3)
    return float(((canvas > 0) & skel).sum() / max(int(skel.sum()), 1)), False


def synthetic(kind, w):
    m = np.zeros((480, 480), np.uint8)
    if kind == "straight":
        for y in (100, 240, 380):
            cv2.line(m, (40, y), (440, y), 1, w)
    else:
        xs = np.arange(40, 440)
        ys = (240 + 80 * np.sin(xs / 25)).astype(int)
        cv2.polylines(m, [np.stack([xs, ys], 1).astype(np.int32)], False, 1, w)
    return m.astype(bool)


def prompt_of(sample):
    return sample.split("_seed")[0].rsplit("_", 1)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-third", default="results/third_pool_20261031/profile_three_arms.csv")
    ap.add_argument("--out", default="results/long_line_width_20261104")
    ap.add_argument("--ic1-only", action="store_true")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    p("# long_line_ratio の線幅 —— 登録 doc/prereg_long_line_width_20261104.md")
    p("## IC1 人工画像")
    widths = (1, 2, 3, 5, 7, 9)
    st = [long_line_ratio_skel(synthetic("straight", w))[0] for w in widths]
    wv = [long_line_ratio_skel(synthetic("wavy", w))[0] for w in widths]
    wo = [long_line_ratio(synthetic("wavy", w), 480, MIN_FRACTION) for w in widths]
    p("幅        " + " ".join(f"{w:>6d}" for w in widths))
    p("直線 骨格 " + " ".join(f"{v:6.3f}" for v in st))
    p("波線 骨格 " + " ".join(f"{v:6.3f}" for v in wv))
    p("波線 元   " + " ".join(f"{v:6.3f}" for v in wo))
    ic1 = min(st) >= 0.95 and (max(wv) - min(wv)) <= 0.10 and max(wv) <= 0.30
    p(f"IC1: {'通った' if ic1 else '通らない —— 何も読まない'}")
    if not ic1 or args.ic1_only:
        (out / "long_line_width.txt").write_text("\n".join(lines) + "\n")
        return

    rows = [r for r in csv.DictReader(open(args.profile_third))]
    data = {k: [] for k in POOLS}
    label_to_key = {v[0]: k for k, v in POOLS.items()}
    tiny = {k: 0 for k in POOLS}
    missing = {k: 0 for k in POOLS}
    for r in rows:
        k = label_to_key.get(r["label"])
        if k is None:
            continue
        path = ROOT / POOLS[k][1] / r["sample"]
        if not path.exists():
            missing[k] += 1
            continue
        gray = np.asarray(Image.open(path).convert("L"))
        v, small = long_line_ratio_skel(gray < THRESHOLD)
        tiny[k] += small
        data[k].append({"sample": r["sample"], "orig": float(r["long_line_ratio"]),
                        "skel": v, "width": float(r["line_width_p50"])})
    with open(out / "long_line_skel.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "sample", "long_line_ratio", "long_line_ratio_skel", "line_width_p50"])
        for k, items in data.items():
            for d in items:
                w.writerow([k, d["sample"], d["orig"], d["skel"], d["width"]])
    p("## 測った母集団")
    for k in POOLS:
        p(f"  {k}: タイル {len(data[k])} / 画像が無い {missing[k]} / 骨格 20画素未満で 0.0 {tiny[k]}")

    def arr(k, col):
        return np.array([d[col] for d in data[k]])

    p("## IC2 / IC3")
    ic2 = True
    for k in ("e6", "i6"):
        v = auc(arr("a2", "orig"), arr(k, "orig"))
        ok = abs(v - KNOWN[k]) <= IC2_TOL
        ic2 &= ok
        p(f"  元の軸 AUC23({k}) {v:.4f} 対 既知 {KNOWN[k]}: {'一致' if ok else '不一致'}")
    ic3 = abs(auc(arr("e6", "skel"), arr("e6", "skel")) - 0.5) < 1e-12
    p(f"  IC2 {'通った' if ic2 else '通らない'} / IC3 {'通った' if ic3 else '通らない'}")

    # N1: swap E6 and I6 per prompt
    by = {k: {} for k in ("e6", "i6")}
    for k in by:
        for d in data[k]:
            by[k].setdefault(prompt_of(d["sample"]), []).append(d["skel"])
    shared = sorted(set(by["e6"]) & set(by["i6"]))
    only_e = [x for k in sorted(set(by["e6"]) - set(by["i6"])) for x in by["e6"][k]]
    only_i = [x for k in sorted(set(by["i6"]) - set(by["e6"])) for x in by["i6"][k]]
    a2s = arr("a2", "skel")
    rng = np.random.default_rng(20261104)
    null = np.empty(args.draws)
    for t in range(args.draws):
        flip = rng.random(len(shared)) < 0.5
        e = [x for i, k in enumerate(shared) for x in (by["i6"][k] if flip[i] else by["e6"][k])] + only_e
        i_ = [x for i, k in enumerate(shared) for x in (by["e6"][k] if flip[i] else by["i6"][k])] + only_i
        null[t] = auc(a2s, np.array(e)) - auc(a2s, np.array(i_))
    med, lo, hi = np.median(null), np.percentile(null, 2.5), np.percentile(null, 97.5)
    p("## 帰無値 N1(モデルの入れ替え)")
    p(f"  対になるプロンプト {len(shared)} / E6 にだけ {len(only_e)} タイル / I6 にだけ {len(only_i)} タイル")
    p(f"  G_skel の帰無: 中央 {med:+.4f}  帯 [{lo:+.4f}, {hi:+.4f}]")
    nc2 = abs(med) < 0.02
    p(f"  NC2 中央 |·|<0.02: {'通った' if nc2 else '通らない'}")

    p("## NC1 線幅との Spearman(外部2プールを合わせる)")
    ow = np.concatenate([arr("e6", "width"), arr("i6", "width")])
    oo = np.concatenate([arr("e6", "orig"), arr("i6", "orig")])
    osk = np.concatenate([arr("e6", "skel"), arr("i6", "skel")])
    r_o = float(spearmanr(oo, ow).statistic)
    r_s = float(spearmanr(osk, ow).statistic)
    p(f"  元の軸 ρ {r_o:+.3f} / 骨格の軸 ρ {r_s:+.3f} —— "
      f"{'予想どおり(元の方が大きい)' if abs(r_o) > abs(r_s) else '予想と違う: 幅で読む理由が弱い'}")

    p("## 主値")
    res = {}
    for col in ("orig", "skel"):
        a23 = {k: auc(arr("a2", col), arr(k, col)) for k in ("e6", "i6")}
        a13 = {k: auc(arr("a1", col), arr(k, col)) for k in ("e6", "i6")}
        res[col] = {"auc23": a23, "auc13": a13, "G": a23["e6"] - a23["i6"], "G13": a13["e6"] - a13["i6"]}
        p(f"  {col:4s}: AUC23 E6 {a23['e6']:.4f} / I6 {a23['i6']:.4f}  G {res[col]['G']:+.4f}   "
          f"(副 AUC13 E6 {a13['e6']:.4f} / I6 {a13['i6']:.4f}  G13 {res[col]['G13']:+.4f})")
    p("  プールの平均: " + " / ".join(
        f"{k} 元 {arr(k, 'orig').mean():.4f} 骨格 {arr(k, 'skel').mean():.4f}" for k in POOLS))

    p("## 判定")
    g_o, g_s = res["orig"]["G"], res["skel"]["G"]
    half = 0.5 * abs(G_ORIG_KNOWN)
    inside = lo <= g_s <= hi
    if not (ic1 and ic2 and ic3 and nc2):
        verdict = "C5 器具 —— 何も読まない"
    elif abs(g_s) <= half and inside:
        verdict = "C1 線幅で説明できる"
    elif abs(g_s) > half and not inside:
        verdict = "C2 線幅を消しても残る"
    else:
        verdict = "判定保留"
    p(f"  G_orig {g_o:+.4f}(既知 {G_ORIG_KNOWN:+.4f})/ G_skel {g_s:+.4f} / 半分 {half:.4f} / 帯の中 {inside}")
    p(f"  → {verdict}")
    side = abs(res["skel"]["auc23"]["e6"] - 0.5) <= 0.05
    p(f"  副: 骨格の AUC23(E6) {res['skel']['auc23']['e6']:.4f} —— "
      + ("|·−0.5| ≤ 0.05: 作者側と外部側の差そのものも線幅だった、と添える" if side else "0.5 から 0.05 より離れている"))

    (out / "long_line_width.txt").write_text("\n".join(lines) + "\n")
    json.dump({"ic1": ic1, "ic2": ic2, "ic3": ic3, "nc2": nc2, "null": [med, lo, hi],
               "rho": {"orig": r_o, "skel": r_s}, "res": res, "verdict": verdict,
               "tiny": tiny, "missing": missing},
              open(out / "long_line_width.json", "w"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
