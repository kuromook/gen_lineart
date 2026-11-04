"""Width-free straightness, as registered: doc/prereg_straight_branch_20261104.md.

straight_branch_share: skeleton -> drop branch points (>= 3 skeleton
neighbours) -> 8-connected branches -> keep branches whose principal-axis
extent is >= 38 px -> share of those branch pixels whose branch has RMS
distance <= 1.0 px from its principal axis.
"""

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_auc import auc  # noqa: E402
from long_line_width import KNOWN, POOLS, prompt_of, synthetic  # noqa: E402

THRESHOLD = 128
L_MIN = 38          # 480 * 0.08, the original axis's shortest segment
RMS_MAX = 1.0
MIN_BRANCH_PX = 5
G_ORIG = 0.7532 - 0.6027
HALF = 0.5 * abs(G_ORIG)
IC2_TOL = 0.0001 + 5e-5
TIE_BAR = 0.50
DRAWS = 1000
NEIGH = np.ones((3, 3), int)
NEIGH[1, 1] = 0


def straight_branch_share(ink):
    """Returns (value, status): status in {'ok', 'tiny_skeleton', 'no_long_branch'}."""
    sk = skeletonize(ink)
    if sk.sum() < 20:
        return 0.0, "tiny_skeleton"
    nb = ndimage.convolve(sk.astype(int), NEIGH, mode="constant") * sk
    branches = sk & (nb < 3)
    lab, _ = ndimage.label(branches, structure=np.ones((3, 3)))
    num = den = 0
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        ys, xs = np.nonzero(lab[sl] == i)
        if len(ys) < MIN_BRANCH_PX:
            continue
        pts = np.stack([xs, ys], 1).astype(float)
        pts -= pts.mean(0)
        w, v = np.linalg.eigh(pts.T @ pts)
        if np.ptp(pts @ v[:, 1]) < L_MIN:
            continue
        den += len(ys)
        num += len(ys) * (np.sqrt(max(w[0], 0.0) / len(ys)) <= RMS_MAX)
    if den == 0:
        return 0.0, "no_long_branch"
    return num / den, "ok"


def synth(kind, w, rng):
    if kind in ("straight", "wavy"):
        return synthetic(kind, w)
    m = np.zeros((480, 480), np.uint8)
    if kind == "diag":
        cv2.line(m, (40, 60), (420, 440), 1, w)
        cv2.line(m, (40, 440), (300, 40), 1, w)
    elif kind == "arc":
        cv2.ellipse(m, (240, 480), (300, 300), 0, 200, 340, 1, w)
    elif kind == "cross":
        cv2.line(m, (40, 240), (440, 240), 1, w)
        cv2.line(m, (240, 40), (240, 440), 1, w)
        cv2.circle(m, (240, 240), 120, 1, w)
    elif kind.startswith("jit"):
        amp = float(kind[3:])
        xs = np.arange(40, 441, 8)
        for y0 in (100, 240, 380):
            ys = y0 + rng.uniform(-amp, amp, len(xs))
            cv2.polylines(m, [np.stack([xs, ys], 1).astype(np.int32)], False, 1, w)
    return m.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-third", default="results/third_pool_20261031/profile_three_arms.csv")
    ap.add_argument("--out", default="results/straight_branch_20261104")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--ic1-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    def save():
        (out / "straight_branch.txt").write_text("\n".join(lines) + "\n")

    p("# 幅に依らない直線らしさ —— 登録 doc/prereg_straight_branch_20261104.md")
    p("## IC1 人工画像")
    widths = (1, 2, 3, 5, 7, 9)
    rng = np.random.default_rng(0)
    want = {"straight": "hi", "diag": "hi", "jit0.5": "hi", "jit1": "hi", "jit2": "hi",
            "wavy": "lo", "arc": "lo", "cross": "any", "jit3": "any"}
    ic1 = True
    for kind, w_ in want.items():
        v = [straight_branch_share(synth(kind, w, rng))[0] for w in widths]
        ok = (max(v) - min(v)) <= 0.10
        ok &= min(v) >= 0.95 if w_ == "hi" else True
        ok &= max(v) <= 0.05 if w_ == "lo" else True
        ic1 &= ok
        p(f"  {kind:8s} " + " ".join(f"{x:.3f}" for x in v) + f"  差 {max(v) - min(v):.3f}  {'○' if ok else '×'}")
    p(f"IC1: {'通った' if ic1 else '通らない —— 何も読まない'}")
    if not ic1 or args.ic1_only:
        save()
        return

    label_to_key = {v[0]: k for k, v in POOLS.items()}
    data = {k: [] for k in POOLS}
    status = {k: collections.Counter() for k in POOLS}
    for r in csv.DictReader(open(args.profile_third)):
        k = label_to_key.get(r["label"])
        if k is None:
            continue
        path = ROOT / POOLS[k][1] / r["sample"]
        if not path.exists():
            status[k]["image_missing"] += 1
            continue
        v, st = straight_branch_share(np.asarray(Image.open(path).convert("L")) < THRESHOLD)
        status[k][st] += 1
        data[k].append({"sample": r["sample"], "orig": float(r["long_line_ratio"]),
                        "new": v, "width": float(r["line_width_p50"])})
    with open(out / "straight_branch.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pool", "sample", "long_line_ratio", "straight_branch_share", "line_width_p50"])
        for k, items in data.items():
            for d in items:
                wr.writerow([k, d["sample"], d["orig"], d["new"], d["width"]])

    def arr(k, c):
        return np.array([d[c] for d in data[k]])

    p("## 母集団と IC4(離散)")
    tie_fire = False
    for k in POOLS:
        v = arr(k, "new")
        c = collections.Counter(np.round(v, 12))
        big = c.most_common(1)[0][1] / len(v)
        tie_fire |= big >= TIE_BAR
        p(f"  {k}: タイル {len(v)}  {dict(status[k])}  値0 {np.mean(v == 0):.1%} / 値1 {np.mean(v == 1):.1%} / "
          f"最大の同値塊 {big:.1%}  平均 {v.mean():.4f}")
    p(f"  IC4: {'発火 —— 判定保留' if tie_fire else '発火しない'}")

    p("## IC2 / IC3")
    ic2 = True
    for k in ("e6", "i6"):
        v = auc(arr("a2", "orig"), arr(k, "orig"))
        ok = abs(v - KNOWN[k]) <= IC2_TOL
        ic2 &= ok
        p(f"  元の軸 AUC23({k}) {v:.4f} 対 既知 {KNOWN[k]}: {'一致' if ok else '不一致'}")
    ic3 = abs(auc(arr("e6", "new"), arr("e6", "new")) - 0.5) < 1e-12
    p(f"  IC2 {'通った' if ic2 else '通らない'} / IC3 {'通った' if ic3 else '通らない'}")

    by = {k: collections.defaultdict(list) for k in ("e6", "i6")}
    for k in by:
        for d in data[k]:
            by[k][prompt_of(d["sample"])].append(d["new"])
    shared = sorted(set(by["e6"]) & set(by["i6"]))
    only_e = [x for q in sorted(set(by["e6"]) - set(by["i6"])) for x in by["e6"][q]]
    only_i = [x for q in sorted(set(by["i6"]) - set(by["e6"])) for x in by["i6"][q]]
    a2n = arr("a2", "new")
    rng = np.random.default_rng(20261104)
    null = np.empty(args.draws)
    for t in range(args.draws):
        flip = rng.random(len(shared)) < 0.5
        e = [x for i, q in enumerate(shared) for x in (by["i6"][q] if flip[i] else by["e6"][q])] + only_e
        s = [x for i, q in enumerate(shared) for x in (by["e6"][q] if flip[i] else by["i6"][q])] + only_i
        null[t] = auc(a2n, np.array(e)) - auc(a2n, np.array(s))
    med, lo, hi = float(np.median(null)), float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))
    nc2 = abs(med) < 0.02
    p("## 帰無値 N1(モデルの入れ替え)")
    p(f"  対になるプロンプト {len(shared)} / E6 にだけ {len(only_e)} タイル / I6 にだけ {len(only_i)} タイル")
    p(f"  G_new の帰無: 中央 {med:+.4f}  帯 [{lo:+.4f}, {hi:+.4f}]   NC2 {'通った' if nc2 else '通らない'}")

    ow = np.concatenate([arr("e6", "width"), arr("i6", "width")])
    r_o = float(spearmanr(np.concatenate([arr("e6", "orig"), arr("i6", "orig")]), ow).statistic)
    r_n = float(spearmanr(np.concatenate([arr("e6", "new"), arr("i6", "new")]), ow).statistic)
    p("## NC1 線幅との Spearman(外部2プール)")
    p(f"  元の軸 {r_o:+.3f} / 新しい軸 {r_n:+.3f} —— "
      f"{'予想どおり(新しい軸の方が小さい)' if abs(r_n) < abs(r_o) else '予想と違う: 幅を消せたとは書かない'}")

    p("## 主値")
    res = {}
    for c in ("orig", "new"):
        a23 = {k: auc(arr("a2", c), arr(k, c)) for k in ("e6", "i6")}
        a13 = {k: auc(arr("a1", c), arr(k, c)) for k in ("e6", "i6")}
        a12 = auc(arr("a1", c), arr("a2", c))
        res[c] = {"auc23": a23, "auc13": a13, "auc12": a12, "G": a23["e6"] - a23["i6"]}
        p(f"  {c:4s}: AUC23 E6 {a23['e6']:.4f} / I6 {a23['i6']:.4f}  G {res[c]['G']:+.4f}   "
          f"(副 AUC13 E6 {a13['e6']:.4f} / I6 {a13['i6']:.4f} / AUC12 {a12:.4f})")

    p("## 判定")
    g = res["new"]["G"]
    inside = lo <= g <= hi
    if not (ic1 and ic2 and ic3 and nc2):
        verdict = "C5 器具 —— 何も読まない"
    elif tie_fire:
        verdict = "判定保留(IC4 離散)"
    elif abs(g) <= HALF and inside:
        verdict = "C1 モデル差は線幅だった"
    elif abs(g) > HALF and not inside:
        verdict = "C2 幅を消してもモデル差が残る"
    else:
        verdict = "判定保留"
    p(f"  G_orig {res['orig']['G']:+.4f}(既知 {G_ORIG:+.4f})/ G_new {g:+.4f} / 半分 {HALF:.4f} / 帯の中 {inside}")
    p(f"  → {verdict}")
    side = abs(res["new"]["auc23"]["e6"] - 0.5) <= 0.05
    p(f"  副: AUC23(E6) {res['new']['auc23']['e6']:.4f} —— "
      + ("|·−0.5| ≤ 0.05: 作者側と外部側の差そのものも線幅だった、と添える" if side else "0.5 から 0.05 より離れている"))
    save()
    json.dump({"ic1": ic1, "ic2": ic2, "ic3": ic3, "ic4_fire": tie_fire, "nc2": nc2,
               "null": [med, lo, hi], "rho": {"orig": r_o, "new": r_n}, "res": res,
               "status": {k: dict(v) for k, v in status.items()}, "verdict": verdict},
              open(out / "straight_branch.json", "w"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
