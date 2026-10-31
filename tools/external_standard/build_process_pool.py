"""Build the eleven ruled works as one pool, normalized exactly as A2 was.

doc/prereg_process_within_extraction_20261031.md. The registration is closed;
this file only implements it.

Order is the registered one and is not re-chosen here: each tile is read back
from the quality-95 JPEG the tiler wrote, the ink gate is applied, then the
white point is equalized, then it is saved as PNG. Those are normalize.py's
own functions -- nothing is reimplemented.

Selection is a measurement, so the gate is counted per work (IC4), and the
paper level is printed per work (IC5).
"""

import argparse
import collections
import csv
import json
import pathlib
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import normalize  # noqa: E402

MANIFEST = "/home/sh1/deepl/lineart/results/clip_pairs_v3_koma_tiles_480_20260826.csv"
TILES = "/home/sh1/deepl/lineart/dataset/pairs_480/train/line_clip_pairs_v3_koma_20260826"
NAME = re.compile(r"^masked_\d+_sub_koma_\d+_\d+_(\d{4}[a-z]?)_(.+?)__")

# The author's rulings, 2026-10-31 (doc/author_process_rulings_20261031.md).
# work: (arc, line-art stage, device label)
RULED = {
    "2017/ako5":      ("学祭編", "デジタル", "未記載"),
    "2018/ako6":      ("学祭編", "デジタル", "未記載"),
    "2021/akogoods":  ("fanbox編", "デジタル", "iPadと明記"),
    "2021/akokate":   ("fanbox編", "デジタル", "iPadと明記"),
    "2022b/akocult":  ("fanbox編", "デジタル", "iPadと明記"),
    "2023b/akocultB": ("fanbox編", "デジタル", "iPadと明記"),
    "2024c/ako":      ("fanbox編", "アナログ", "—"),
    "2024/ako3B":     ("fanbox編", "アナログ", "—"),
    "2019/succor":    ("系外", "デジタル", "未記載"),
    "2017/running":   ("系外", "デジタル", "未記載"),
    "2024b/fitness":  ("系外", "アナログ", "—"),
}
# Printed as "excluded", with its reason, because selection is a measurement.
EXCLUDED_UNRULED_COLOUR = "2022b/hyoui_color"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/author_process_20261031")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "norm").mkdir(parents=True, exist_ok=True)
    tiles = pathlib.Path(TILES)

    want = collections.defaultdict(list)
    seen = collections.Counter()
    for rec in csv.DictReader(open(MANIFEST)):
        m = NAME.match(rec["source_name"])
        if not m:
            seen["name_unparsed"] += 1
            continue
        work = f"{m.group(1)}/{m.group(2)}"
        seen[work] += 1
        if work in RULED:
            want[work].append(rec)

    rows, gate, missing, papers = [], collections.Counter(), collections.Counter(), {}
    for work in RULED:
        kept, lv = 0, []
        for rec in want[work]:
            src = tiles / rec["name"]
            if not src.exists():
                missing[work] += 1
                continue
            gray = np.asarray(Image.open(src).convert("L"))
            if normalize.ink_ratio(gray) < normalize.INK_GATE:
                gate[work] += 1
                continue
            eq, paper = normalize.equalize_white_point(gray)
            lv.append(paper)
            Image.fromarray(eq).save(out / "norm" / f"{src.stem}.png")
            arc, proc, dev = RULED[work]
            rows.append({"tile": src.stem, "tag": work.split("/")[0],
                         "work": work, "arc": arc, "line_process": proc,
                         "device": dev, "group": rec["source_name"],
                         "page": rec["source_page"]})
            kept += 1
        papers[work] = (int(min(lv)), int(np.median(lv)), int(max(lv))) if lv else None
        print(f"{work:16s} 台帳 {len(want[work]):4d}  実体なし {missing[work]:4d}"
              f"  門で落ち {gate[work]:3d}  採用 {kept:4d}"
              f"  紙 {papers[work]}", flush=True)

    man = out / "pool_manifest.csv"
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    g = collections.Counter(r["work"] for r in rows)
    grp = collections.defaultdict(set)
    pg = collections.defaultdict(set)
    for r in rows:
        grp[r["work"]].add(r["group"])
        pg[r["work"]].add(r["page"])
    print("\n## IC3 採用したタイル / 群 / 頁")
    for work in RULED:
        print(f"  {work:16s} タイル {g[work]:4d}  群 {len(grp[work]):4d}  頁 {len(pg[work]):3d}")
    print(f"  計 タイル {len(rows)}  群 {sum(len(grp[w]) for w in RULED)}")
    print(f"\n## 外した母集団(理由つき)")
    print(f"  {EXCLUDED_UNRULED_COLOUR}: {seen[EXCLUDED_UNRULED_COLOUR]} 行"
          "(色原稿か線画層かの裁定が無い。census へ照会中)")
    print(f"  裁定の無い他の作品タグ: {len(seen) - len(RULED) - 1} 種")
    print(f"  門で落ちた計: {sum(gate.values())} / 実体の無い台帳の行: {sum(missing.values())}")
    (out / "build_report.json").write_text(json.dumps(
        {"kept": len(rows), "gate": dict(gate), "missing": dict(missing),
         "papers": {k: v for k, v in papers.items()},
         "ink_gate": normalize.INK_GATE,
         "excluded_color_rows": seen[EXCLUDED_UNRULED_COLOUR]},
        indent=1, ensure_ascii=False))
    print(f"\n書き出し: {man}")


if __name__ == "__main__":
    main()
