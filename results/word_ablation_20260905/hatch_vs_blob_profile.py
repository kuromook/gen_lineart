"""User question (2026-09-05): visually, text_focus (highest orientation_entropy
in the word-ablation sweep, +0.0098 vs baseline) looks like it has LESS
cross-hatch than other variants, not more -- it seems to escape into a flat
solid/gray blob instead (see montage_extremes.png row lineart_008_023: a
smoothly-curved solid dark blob with a couple of text-like squiggles, not a
hatch mesh). orientation_entropy alone can't tell these two failure modes
apart: a single smoothly-curved blob boundary can scatter edge angles just
as broadly as a genuine cross-hatch mesh.

This reuses measure_lineart_profile.py's full multi-axis profile (already
the project's standard tool for exactly this "which axis is being silently
sacrificed" question, see its docstring) to add two axes that DO distinguish
them:
- components_per_1k_ink_px: hatch = many separate short strokes -> HIGH.
  solid blob = one giant connected region -> LOW.
- width_consistency (stroke-width p95/p50): hatch = thin, fairly uniform
  width -> LOW. solid blob = interior far from any boundary -> HIGH.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/sh1/deepl/lineart/tools/evaluation")
sys.path.insert(0, "/home/sh1/deepl/lineart/tools/pair_extraction")
from measure_lineart_profile import profile_metrics  # noqa: E402

TRACK = Path("/home/sh1/deepl/lineart-controlnet-realpairs")
SAMPLES = [l.strip()[:-4] for l in open(TRACK / "data/diag_valid5.txt") if l.strip()]

SOURCES = {
    "GT": TRACK / "data" / "diag_gt_line_{s}.jpg",
    "baseline(no word)": TRACK / "results/controlnet_lora_manga_nomangaword_20260905_eval" / "{s}_out.png",
    "comic(+.002 entropy)": TRACK / "results/word_ablation_20260905/outputs/comic" / "{s}_out.png",
    "text_focus(+.010 entropy, MAX)": TRACK / "results/word_ablation_20260905/outputs/text_focus" / "{s}_out.png",
    "blood(-.029 entropy, MIN)": TRACK / "results/word_ablation_20260905/outputs/blood" / "{s}_out.png",
    "black_border(-.022 entropy)": TRACK / "results/word_ablation_20260905/outputs/black_border" / "{s}_out.png",
}

AXES = ["orientation_entropy", "components_per_1k_ink_px", "width_consistency", "line_width_p50", "ink_ratio"]


def main():
    import numpy as np

    print(f"{'source':32}" + "".join(f"{a:>26}" for a in AXES))
    for label, tpl in SOURCES.items():
        vals = {a: [] for a in AXES}
        for s in SAMPLES:
            p = Path(str(tpl).format(s=s))
            m = profile_metrics(p)
            for a in AXES:
                vals[a].append(m[a])
        means = {a: np.mean(vals[a]) for a in AXES}
        print(f"{label:32}" + "".join(f"{means[a]:26.4f}" for a in AXES))


if __name__ == "__main__":
    main()
