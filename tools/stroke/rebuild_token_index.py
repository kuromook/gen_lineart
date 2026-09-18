#!/usr/bin/env python
"""Rebuild a token directory's index.csv from the npz files actually on disk.

The exporter writes index.csv as it goes, so a run that is interrupted and
resumed would otherwise lose the rows of every panel it skips as already done.
This reads the truth off disk instead, and reports any file that fails to open
(the one way an interrupted write can still hurt).
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/sh1/deepl/lineart")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="results/panel_tokens_v2_20260918")
    p.add_argument("--out", default="")
    a = p.parse_args()
    tok = Path(a.tokens)
    groups = {}
    for m in sorted(ROOT.glob("dataset/regions_*_koma_panels_2026*/manifest.csv")):
        src = m.parent.name
        for r in csv.DictReader(open(m)):
            groups[(src, r["name"])] = r.get("content_fingerprint", "") or f"{src}/{r['name']}"
    rows, bad = [], []
    files = sorted(tok.glob("*/*.npz"))
    for i, f in enumerate(files, 1):
        src, name = f.parent.name, f.name[:-4]
        try:
            d = np.load(f)
            rows.append({"source": src, "name": name,
                         "group": groups.get((src, name), f"{src}/{name}"),
                         "strokes": int(d["kpts"].shape[0]), "joints": int(d["joints"].shape[0])})
        except Exception as e:
            bad.append((str(f), repr(e)[:80]))
        if i % 500 == 0:
            print(f"{i}/{len(files)}", flush=True)
    out = Path(a.out) if a.out else tok / "index.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["source", "name", "group", "strokes", "joints"])
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} panels -> {out}")
    if bad:
        print(f"壊れたファイル {len(bad)} 件(消してから再開すれば作り直される):")
        for f, e in bad[:10]:
            print("  ", f, e)
    else:
        print("壊れたファイルなし")


if __name__ == "__main__":
    main()
