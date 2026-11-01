"""Build the rough side of the same tiles, twice: equalized and as it is.

doc/prereg_rough_side_mixture_20261101.md. The registration is closed; this
file only implements it.

The rough tile sits beside the line-art tile under the same name in the
extraction, so the pool is the stage-2 manifest read again with a different
directory. Two things are deliberately NOT borrowed from the line-art side:

  * the gate. On the rough side being faint is the thing we want to see, so
    only a tile with no ink pixel at all is dropped (IC3, 規律9).
  * the white point. It is equalized into norm/ and left alone in raw/, and
    both are measured, so that a separation living only in raw/ gets named as
    paper rather than as drawing (IC4, 条例2 を役割ごとに引き直したもの).

Selection is a measurement, so every drop is counted per work and printed.
"""

import argparse
import collections
import csv
import json
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import normalize  # noqa: E402

ROUGH = pathlib.Path("/home/sh1/deepl/lineart/dataset/pairs_480/train/rough")
INK = 128           # a pixel darker than this is ink, for the emptiness test only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/author_process_20261101/pool_manifest.csv")
    ap.add_argument("--out", default="data/author_rough_20261101")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    for sub in ("norm", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    man = list(csv.DictReader(open(args.manifest)))
    rows, missing, empty, papers, inks = [], collections.Counter(), collections.Counter(), {}, {}
    by_work = collections.defaultdict(list)
    for r in man:
        by_work[r["work"]].append(r)

    for work, items in by_work.items():
        lv, ink = [], []
        kept = 0
        for r in items:
            src = ROUGH / f"{r['tile']}.jpg"
            if not src.exists():
                missing[work] += 1
                continue
            gray = np.asarray(Image.open(src).convert("L"))
            ratio = float((gray < INK).mean())
            if ratio <= 0.0:
                empty[work] += 1
                continue
            eq, paper = normalize.equalize_white_point(gray)
            Image.fromarray(gray).save(out / "raw" / f"{r['tile']}.png")
            Image.fromarray(eq).save(out / "norm" / f"{r['tile']}.png")
            lv.append(paper)
            ink.append(ratio)
            rows.append(dict(r))
            kept += 1
        papers[work] = (int(min(lv)), int(np.median(lv)), int(max(lv))) if lv else None
        inks[work] = (float(np.min(ink)), float(np.median(ink)), float(np.max(ink))) if ink else None
        print(f"{work:20s} 線画 {len(items):4d}  下絵なし {missing[work]:3d}"
              f"  墨0で落ち {empty[work]:3d}  採用 {kept:4d}"
              f"  紙 {papers[work]}  墨率 {None if not ink else tuple(round(v,5) for v in inks[work])}",
              flush=True)

    with open(out / "pool_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n## IC2 線画と下絵が1対1")
    print(f"  線画 {len(man)} / 下絵 {len(rows)}"
          f"  → {'一致' if len(man) == len(rows) else '一致しない'}"
          f"(下絵なし {sum(missing.values())} / 墨0 {sum(empty.values())})")
    (out / "build_report.json").write_text(json.dumps(
        {"line_tiles": len(man), "kept": len(rows), "missing": dict(missing),
         "empty": dict(empty), "papers": papers, "ink": inks,
         "ink_threshold_for_emptiness": INK}, indent=1, ensure_ascii=False))
    print(f"\n書き出し: {out / 'pool_manifest.csv'}")


if __name__ == "__main__":
    main()
