"""One figure for the ako-lineage distances: every pair on one axis.

Reads the JSON the measurement wrote. Palette: slots 1-3 of the reference
categorical theme (validated: worst adjacent CVD dE 9.2; the aqua slot carries
a contrast WARN, so every mark is directly labelled).
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

R = pathlib.Path("results/ako_series_20261031")
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
SURFACE, INK, INK2, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"

for f in font_manager.findSystemFonts():
    if "NotoSansCJKjp" in f or "NotoSansCJK-Regular" in f:
        font_manager.fontManager.addfont(f)
        break
plt.rcParams.update({
    "font.family": "Noto Sans CJK JP", "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": "#dcdbd6",
    "xtick.color": INK2, "ytick.color": INK2,
})

data = json.loads((R / "ako_series.json").read_text())
pairs = {k: v for k, v in data.items() if k != "verdict"}

def group_of(k):
    a, b = k.split(" 対 ")
    if "参考" in b:
        return "参考"
    if b.startswith("X"):
        return "外部"
    return "編またぎ" if a.startswith("K5") else "編同じ"

GROUPS = [("編またぎ", "編をまたぐ ―― FANBOX編 `ako5`(2025) 対 学祭編 `ako7`〜`ako10`", BLUE),
          ("編同じ", "編が同じ ―― 学祭編どうし(`ako7`〜`ako10`)", AQUA),
          ("外部", "外部との対 ―― ako の各行 対 SD の2モデル", ORANGE),
          ("参考", "(参考)`ako5`(2025) 対 `ako` 系外のアナログ4作", GREY)]

rows = []
for key, title, color in GROUPS:
    sel = sorted([(k, v) for k, v in pairs.items() if group_of(k) == key],
                 key=lambda kv: kv[1]["multi"])
    rows.append((None, title, color, None))
    rows.extend((k, k, color, v) for k, v in sel)

fig, ax = plt.subplots(figsize=(14.6, 12.2))
fig.subplots_adjust(left=0.335, right=0.975, top=0.90, bottom=0.062)

ext_min = min(v["multi"] for k, v in pairs.items() if group_of(k) == "外部")
ax.axvline(0.5, color=INK2, lw=1, ls="--")
ax.axvline(ext_min, color=ORANGE, lw=1.4, ls=":")
ax.annotate(f"`ako` 対外部の最小 {ext_min:.4f}", (ext_min, len(rows) - 0.1),
            ha="right", va="bottom", fontsize=10, color=ORANGE)

labels = []
for i, (key, title, color, v) in enumerate(rows):
    y = len(rows) - 1 - i
    if v is None:
        labels.append((y, title, True))
        continue
    labels.append((y, title.replace(" 対 ", "  対  "), False))
    lo, hi = v["band"]
    ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
    judged = v["judged"]
    ax.plot([v["multi"]], [y], "o", ms=9, zorder=3,
            color=color if judged else SURFACE, mec=color, mew=2)
    note = "" if judged else "  (置換の帯が足りない)"
    ax.annotate(f"{v['multi']:.4f}{note}", (max(hi, v['multi']), y),
                textcoords="offset points", xytext=(9, -4), fontsize=10, color=INK)

ax.set_yticks([y for y, _t, _h in labels],
              [t for _y, t, _h in labels], fontsize=10)
for tick, (_y, _t, head) in zip(ax.get_yticklabels(), labels):
    tick.set_color(INK if head else INK2)
    tick.set_fontsize(10.5 if head else 9.5)
ax.set_xlim(0.45, 1.155)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_ylim(-0.8, len(rows) - 0.2)
ax.set_xlabel("多変量(F9 = 9軸)の群分割5折の平均 test AUC。細い線は群単位 bootstrap の 95%")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.text(0.015, 0.975, "`ako` 系の中の距離 —— 作者の中の対が、作者と外部の対を超える",
         ha="left", fontsize=15.5, color=INK)
fig.text(0.015, 0.952,
         "登録 doc/prereg_ako_within_series_20261031.md / 判定の単位はタイル1枚 / "
         "この track は絵を判定しない", ha="left", fontsize=10.5, color=INK2)
fig.text(0.015, 0.928,
         "白抜きの点 = 置換の帯が ±0.15 を超えた対(判定しない)。"
         "編・工程・時期・抽出経路は、この対では一緒に動く",
         ha="left", fontsize=10.5, color=INK2)
out = R / "ako_series_figure.png"
fig.savefig(out, dpi=130)
print("wrote", out)
