#!/usr/bin/env python
"""Summarize tokenize_panels output per source and against the tile-unit corpus."""
import argparse, csv, collections
from pathlib import Path
import numpy as np


def q(a, ps=(10, 50, 90)):
    return " / ".join(f"{np.percentile(a, p):.0f}" for p in ps) if len(a) else "-"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="results/panel_skeleton_20260917")
    p.add_argument("--tiles", default="results/tokenize_corpus_20260917/tokens.csv")
    a = p.parse_args()
    panels = list(csv.DictReader(open(Path(a.cache) / "panel_strokes.csv")))
    by = collections.defaultdict(lambda: {"len": [], "spans": [], "fill": []})
    for r in csv.DictReader(open(Path(a.cache) / "strokes.csv")):
        d = by[r["source"]]
        d["len"].append(int(r["n_px"])); d["spans"].append(int(r["n_spans"])); d["fill"].append(float(r["fill_share"]))
        d = by["ALL"]
        d["len"].append(int(r["n_px"])); d["spans"].append(int(r["n_spans"])); d["fill"].append(float(r["fill_share"]))
    pp = collections.defaultdict(list)
    for r in panels:
        pp[r["source"]].append(r); pp["ALL"].append(r)
    print("| source | panels | strokes/panel p10/p50/p90 | strokes/MP p50 | len px p10/p50/p90 | spans/stroke mean | strokes ≥2 spans | len share ≥2 spans | captured p50 | fill_share≤0.2 |")
    print("|---|---:|---|---:|---|---:|---:|---:|---:|---:|")
    for s in sorted(pp, key=lambda k: (k == "ALL", k)):
        P = pp[s]; d = by[s]
        L = np.array(d["len"]); S = np.array(d["spans"]); F = np.array(d["fill"])
        spp = np.array([int(r["strokes"]) for r in P])
        mp = np.array([int(r["w"]) * int(r["h"]) / 1e6 for r in P])
        cap = np.array([float(r["captured_strokes"]) for r in P])
        name = s.replace("regions_", "").split("_koma")[0]
        print(f"| {name} | {len(P)} | {q(spp)} | {np.median(spp/mp):.0f} | {q(L)} | {S.mean():.2f} | {(S>=2).mean():.1%} | "
              f"{L[S>=2].sum()/L.sum():.1%} | {np.median(cap):.3f} | {(F<=0.2).mean():.1%} |")
    if Path(a.tiles).exists():
        T = np.array([int(r["n_px"]) for r in csv.DictReader(open(a.tiles))])
        print(f"\ntile unit (480px, junction spans): len p10/p50/p90 {q(T)}, n={len(T)}")


if __name__ == "__main__":
    main()
