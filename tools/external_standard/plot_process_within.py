"""The figure for doc/result_process_within_extraction_20261031.md.

Three panels, all reading the run's own JSON: the five cells as dots, the
matched subsets as dots, and every own pair against its era distance with
NC4's regression line. Colours are this track's two validated hues.
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a8a"
SRC = "results/process_within_20261031/process_within.json"
OUT = "results/process_within_20261031/process_figure.png"
MATCH = 3
NAMES = {"S1": "S1 工程同じ 編同じ", "S2": "S2 工程同じ 編ちがう",
         "S3": "S3 工程ちがう 編同じ", "S4": "S4 工程も編もちがう"}


def main():
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAGothic", "DejaVu Sans"]
    d = json.loads(pathlib.Path(SRC).read_text())
    own = {k: v for k, v in d["pairs"].items()
           if " 対 X" not in k and "fitnesskoma" not in k}
    ext = [v["multi"] for k, v in d["pairs"].items() if " 対 X" in k]

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # --- panel 1: all pairs, by cell
    keys = ["S1", "S2", "S3", "S4"]
    rows = [[v["multi"] for v in own.values() if v["cell"] == k] for k in keys]
    rows.append(ext)
    labels = [f"{NAMES[k]}\n({len(r)}対)" for k, r in zip(keys, rows[:4])]
    labels.append(f"S5 外部との対\n({len(ext)}対)")
    cols = [BLUE, BLUE, ORANGE, ORANGE, GREY]
    rng = np.random.default_rng(7)
    for i, (vals, c) in enumerate(zip(rows, cols)):
        y = len(rows) - 1 - i + rng.uniform(-0.13, 0.13, len(vals))
        ax[0].scatter(vals, y, s=46, color=c, alpha=0.75, edgecolor="white", linewidth=0.7)
        ax[0].plot([np.median(vals)] * 2, [len(rows) - 1 - i - 0.28, len(rows) - 1 - i + 0.28],
                   color=c, linewidth=2.6)
    ax[0].set_yticks(range(len(rows) - 1, -1, -1))
    ax[0].set_yticklabels(labels, fontsize=9)
    ax[0].set_xlabel("対の多変量 AUC (縦棒は中央)", fontsize=10)
    ax[0].set_title("全55対 + 外部22対 —— セルごと", fontsize=11)
    ax[0].set_xlim(0.45, 1.02)
    ax[0].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # --- panel 2: matched only
    m = {k: [v["multi"] for v in own.values()
             if v["cell"] == k and v["era"] <= MATCH] for k in keys}
    order = ["S1", "S2", "S3"]
    for i, k in enumerate(order):
        c = BLUE if k in ("S1", "S2") else ORANGE
        y = len(order) - 1 - i + rng.uniform(-0.1, 0.1, len(m[k]))
        ax[1].scatter(m[k], y, s=70, color=c, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax[1].plot([np.median(m[k])] * 2, [len(order) - 1 - i - 0.25, len(order) - 1 - i + 0.25],
                   color=c, linewidth=2.8)
    ax[1].set_yticks(range(len(order) - 1, -1, -1))
    ax[1].set_yticklabels([f"{NAMES[k]}\n({len(m[k])}対)" for k in order], fontsize=9)
    ax[1].set_xlabel("対の多変量 AUC", fontsize=10)
    ax[1].set_title("時期距離 ≤ 3 の対だけ(S3m は3対。判定保留)", fontsize=11)
    ax[1].set_xlim(0.45, 1.02)
    ax[1].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # --- panel 3: era distance
    for k in keys:
        x = [v["era"] for v in own.values() if v["cell"] == k]
        y = [v["multi"] for v in own.values() if v["cell"] == k]
        c = BLUE if k in ("S1", "S2") else ORANGE
        mk = "o" if k in ("S1", "S3") else "^"
        ax[2].scatter(x, y, s=52, color=c, marker=mk, alpha=0.8,
                      edgecolor="white", linewidth=0.7, label=NAMES[k])
    xs = np.array([v["era"] for v in own.values()], dtype=float)
    ys = np.array([v["multi"] for v in own.values()], dtype=float)
    sl, ic = np.polyfit(xs, ys, 1)
    gx = np.linspace(0, xs.max(), 10)
    gy = np.clip(sl * gx + ic, None, 1.0)
    ax[2].plot(gx, gy, color=GREY, linewidth=2, linestyle="--",
               label=f"NC4 傾き {sl:+.4f} / 札  R² {d['summary']['NC4']['r2']:.3f}")
    ax[2].axvline(MATCH + 0.5, color="#c8c8c8", linewidth=1)
    ax[2].text(MATCH + 0.7, 0.47, "← matched", fontsize=8.5, color="#777777")
    ax[2].set_xlabel("時期距離(年代札の索引の差)", fontsize=10)
    ax[2].set_ylabel("対の多変量 AUC", fontsize=10)
    ax[2].set_title("時期だけでも距離は上がる(NC4)", fontsize=11)
    ax[2].set_ylim(0.45, 1.03)
    ax[2].legend(fontsize=8, loc="lower right", framealpha=0.95)

    for a in ax:
        a.grid(axis="x", color="#eeeeee", linewidth=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("同じ抽出の中の工程と編 —— 11作 2,947枚 / 判定の単位はタイル1枚",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT, dpi=140)
    print("saved", OUT)


if __name__ == "__main__":
    main()
