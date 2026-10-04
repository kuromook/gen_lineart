#!/usr/bin/env python
"""目印付けの候補の順番と、検証用の取り置きを固定する(2026-10-04 事前登録)。

候補 = 顔の枠(H3 faces.csv)に中心が入るまとまりのうち r(直径/枠の一辺)< 0.5。順番は seed 固定の無作為。
取り置き: シリーズ群を無作為に並べ候補の 15% に達するまで → test_series。残りはページ単位の sha1 mod 100 < 20 → test_panel。
出力: labels/queue_v1.csv、labels/holdout_plan.json。既にあれば上書きしない(--force なし)。
"""
import csv, hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TRACKF = Path("/home/sh1/deepl/lineart-stroke-grammar")
CLUSTERS = TRACKF / "results/cluster_set_20260919"
PACK = TRACKF / "results/panel_pack_20260919"
FACES = ROOT / "results/h3_face_parts_20261004/faces.csv"
H3_CHECK = ROOT / "results/h3_face_parts_20261004/check.json"
GROUPS = ROOT / "tools/gen/series_groups.json"
OUT = ROOT / "labels"
SEED, R_MAX, SERIES_FRAC, PANEL_MOD = 20261004, 0.5, 0.15, 20
A_SIZE = 28.0


def page_key(name):
    m = re.match(r"^(.*)_p\d+\.(png|jpg)$", name)
    return m.group(1) if m else name


def main():
    if (OUT / "queue_v1.csv").exists():
        sys.exit("labels/queue_v1.csv は既にある(順番と取り置きは固定。作り直さない)")
    OUT.mkdir(exist_ok=True)
    meta = list(csv.DictReader(open(CLUSTERS / "meta.csv")))
    prow = list(csv.DictReader(open(PACK / "panels.csv")))
    faces = defaultdict(list)
    for t in csv.DictReader(open(FACES)):
        if int(t["box"]) > 0:
            faces[int(t["panel"])].append((int(t["box"]), float(t["x0"]), float(t["y0"]), float(t["x1"]), float(t["y1"])))
    sg = json.load(open(GROUPS))
    w2g = {w: g for g, ws in sg["groups"].items() for w in ws}

    cand = []
    for i, m in enumerate(meta):
        pid = int(m["panel"])
        if pid not in faces:
            continue
        x, y = float(m["cx"]), float(m["cy"])                     # meta cx = x、cy = y(画像の座標)
        best = None
        for (k, x0, y0, x1, y1) in faces[pid]:
            area = (x1 - x0) * (y1 - y0)
            if x0 <= x <= x1 and y0 <= y <= y1 and (best is None or area < best[0]):
                best = (area, k, x0, y0, x1, y1)
        if best is None:
            continue
        area, k, x0, y0, x1, y1 = best
        r = 2 * A_SIZE / float(m["scale"]) / np.sqrt(area)
        if r >= R_MAX:
            continue
        cand.append(dict(row=i, panel=pid, box=k, x0=x0, y0=y0, x1=x1, y1=y1, r=round(float(r), 4),
                         u=round((x - x0) / (x1 - x0), 4), v=round((y - y0) / (y1 - y0), 4),
                         work=m["work"], group=w2g[m["work"]], page=page_key(prow[pid]["name"])))
    ref = json.load(open(H3_CHECK))["r_bin_counts"]
    n_ref = ref["r<0.1"] + ref["0.1-0.25"] + ref["0.25-0.5"]
    assert len(cand) == n_ref, f"候補数 {len(cand)} が H3 の {n_ref} と不一致"

    rng = np.random.default_rng(SEED)
    gcount = Counter(c["group"] for c in cand)
    groups = sorted(gcount)
    order_g = [groups[j] for j in rng.permutation(len(groups))]
    test_groups, acc = [], 0
    for g in order_g:
        if acc >= SERIES_FRAC * len(cand):
            break
        test_groups.append(g); acc += gcount[g]
    for c in cand:
        if c["group"] in test_groups:
            c["usage"] = "test_series"
        else:
            h = int(hashlib.sha1(c["page"].encode()).hexdigest(), 16) % 100
            c["usage"] = "test_panel" if h < PANEL_MOD else "train"
    perm = rng.permutation(len(cand))
    queue = [cand[j] for j in perm]
    for o, c in enumerate(queue):
        c["order"] = o
    cols = ["order", "row", "panel", "box", "x0", "y0", "x1", "y1", "r", "u", "v", "usage", "work", "group", "page"]
    with open(OUT / "queue_v1.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader(); wr.writerows(queue)
    use = Counter(c["usage"] for c in cand)
    plan = dict(seed=SEED, r_max=R_MAX, series_frac_target=SERIES_FRAC, panel_mod=PANEL_MOD, n_candidates=len(cand),
                n_groups=len(groups), test_series_groups=test_groups,
                usage_counts=dict(use), usage_frac={k: round(v / len(cand), 4) for k, v in use.items()},
                first300=dict(Counter(c["usage"] for c in queue[:300])))
    json.dump(plan, open(OUT / "holdout_plan.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(plan, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
