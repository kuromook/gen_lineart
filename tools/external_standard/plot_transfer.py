"""One figure for the transfer run: the ladder, the tag rows, and the arms.

Reads the JSON the measurement wrote; nothing is retyped by hand.
Palette: slots 1 and 2 of the reference categorical theme, validated with the
skill's checker (worst adjacent CVD dE 24.7, all checks pass).
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

R = pathlib.Path("results/multivariate_transfer_20261031")
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd6"
MODELS = [("animagine", "animagine-xl-3.1(本命)", BLUE),
          ("illustrious", "illustriousXL_v01(NC3)", ORANGE)]

for f in font_manager.findSystemFonts():
    if "NotoSansCJKjp" in f or "NotoSansCJK-Regular" in f:
        font_manager.fontManager.addfont(f)
        break
plt.rcParams.update({
    "font.family": "Noto Sans CJK JP", "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
})

data = {m: json.loads((R / f"transfer_{m}.json").read_text()) for m, _, _ in MODELS}

fig = plt.figure(figsize=(14.5, 13.6))
gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 0.75], hspace=0.46,
                      left=0.235, right=0.985, top=0.885, bottom=0.055)

# ---- Panel A: the ladder ------------------------------------------------
ax = fig.add_subplot(gs[0])
rows = [("E4", "E4 作者の中\nA1 対 A2(群分割5折)"),
        ("E3", "E3 逆向きの移植\nA2 対 A3 で学習 → A1"),
        ("E2", "E2 移植(主)\nA1 対 A3 で学習 → A2"),
        ("E1", "E1 内部\nA1 の作品を1つ残す")]
for i, (key, _) in enumerate(rows):
    n = data["animagine"]["F9"][f"null_{key}"]
    ax.add_patch(plt.Rectangle((n["lo"], i - 0.34), n["hi"] - n["lo"], 0.68,
                               color="#e9e8e3", zorder=0))
for m, label, color in MODELS:
    f9 = data[m]["F9"]
    for i, (key, _) in enumerate(rows):
        y = i + (0.14 if m == "animagine" else -0.14)
        v = f9[key]["mean"] if key in ("E1",) else f9[key]
        band = f9.get(f"{key}_band")
        if band:
            ax.plot(band, [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
        elif key in ("E1", "E4"):
            folds = list(f9["E1"]["folds"].values()) if key == "E1" else None
            if folds:
                ax.plot([min(folds), max(folds)], [y, y], color=color, lw=2,
                        alpha=0.45, solid_capstyle="round", zorder=2)
        ax.plot([v], [y], "o", ms=9, color=color, mec=SURFACE, mew=2, zorder=3,
                label=label if i == 3 else None)
        ax.annotate(f"{v:.3f}", (v, y), textcoords="offset points", xytext=(11, -4),
                    fontsize=10, color=INK)
for x, txt, st in [(0.5, "0.5", "--"), (0.60, "棒 0.60(C1)", ":"), (0.75, "棒 0.75(C2)", ":")]:
    ax.axvline(x, color=INK2 if st == "--" else "#8a8984", lw=1, ls=st, zorder=1)
    ax.annotate(txt, (x, -0.52), ha="center", fontsize=9, color=INK2)
ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=10)
ax.set_xlim(0, 1.14)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylim(-0.75, 3.6)
ax.set_xlabel("test AUC(タイル1枚を単位、tie は 0.5)")
ax.set_title("移植の梯子 —— 灰色の帯は null_label(学習側の札を群単位で置換、F9)の 95%。"
             "E2・E3 の棒は帯の中に在る = この設計では読めない\n"
             "細い線: E2・E3 は群単位 bootstrap の 95%、E1 は5つの折の幅",
             loc="left", fontsize=11.5, color=INK, pad=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---- Panel B: E2 per tag row -------------------------------------------
ax2 = fig.add_subplot(gs[1])
tags = list(data["animagine"]["F9"]["E2_rows"].keys())[::-1]
for m, label, color in MODELS:
    rows2 = data[m]["F9"]["E2_rows"]
    for i, t in enumerate(tags):
        y = i + (0.16 if m == "animagine" else -0.16)
        held = t == "2022b/ako10"
        ax2.plot([0.5, rows2[t]], [y, y], color=color, lw=2, alpha=0.3, zorder=1)
        ax2.plot([rows2[t]], [y], "o", ms=9, color=SURFACE if held else color,
                 mec=color, mew=2, zorder=3, label=label if i == 0 else None)
        ax2.annotate(f"{rows2[t]:.3f}" + ("  (群2個。判定しない)" if held else ""),
                     (rows2[t], y), textcoords="offset points", xytext=(9, -4),
                     fontsize=10, color=INK)
ax2.axvline(0.5, color=INK2, lw=1, ls="--")
ax2.set_yticks(range(len(tags)), tags, fontsize=10)
ax2.set_xlim(0.4, 1.24)
ax2.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.set_xlabel("E2 の test AUC(その札のタイル 対 A3 の試し半分)")
ax2.set_title("A2 の札ごと(畳まない)—— どの札も移植の側に出る。ako7 の2行が最も低い",
              loc="left", fontsize=11.5, color=INK, pad=10)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# ---- Panel C: the arms --------------------------------------------------
ax3 = fig.add_subplot(gs[2])
arms = [("M3", "M3 F9 の回帰(9軸)"), ("M2", "M2 単軸の最良(モデルなし)"),
        ("M1", "M1 頻度表 ink_ratio(学習なし)"), ("M0", "M0 定数(恒等)")]
h = 0.34
for m, label, color in MODELS:
    f9 = data[m]["F9"]
    for i, (key, _) in enumerate(arms):
        v = f9["E2"] if key == "M3" else f9[key]
        y = i + (h / 2 + 0.01 if m == "animagine" else -h / 2 - 0.01)
        ax3.barh(y, v, height=h, color=color, zorder=2,
                 label=label if i == 0 else None)
        note = f"  ({f9['M2_axis']})" if key == "M2" else ""
        ax3.annotate(f"{v:.3f}{note}", (v, y), textcoords="offset points",
                     xytext=(7, -4), fontsize=10, color=INK)
ax3.axvline(0.5, color=INK2, lw=1, ls="--")
ax3.set_yticks(range(len(arms)), [a[1] for a in arms], fontsize=10)
ax3.set_xlim(0, 1.32)
ax3.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax3.set_xlabel("同じ試し側(A2 対 A3 の試し半分)での test AUC")
ax3.set_title("腕の比較 —— 1本ずつでは移植できない(M2)。9本を同時に入れると移植の側に出る(M3)",
              loc="left", fontsize=11.5, color=INK, pad=10)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)

fig.text(0.015, 0.972, "多変量の移植 —— F9(9軸。門に当たる2軸と ink_ratio を抜いた集合)",
         ha="left", fontsize=15, color=INK)
fig.text(0.015, 0.947, "登録 doc/prereg_multivariate_transfer_20261031.md / "
         "判定の単位はタイル1枚 / この track は絵を判定しない",
         ha="left", fontsize=10.5, color=INK2)
handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=c, mec=SURFACE, mew=2,
                      label=l) for _, l, c in MODELS]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985),
           frameon=False, fontsize=11, ncol=2)
out = R / "transfer_figure.png"
fig.savefig(out, dpi=130)
print("wrote", out)
