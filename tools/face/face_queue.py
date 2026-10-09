#!/usr/bin/env python
"""線単位のグループ付けの対象(顔)の順番と、検証用の取り置きを固定する(2026-10-09 事前登録)。

対象 = H3 faces.csv の顔の枠すべて。順番は seed 20261009 の無作為。
取り置きは labels/holdout_plan.json と同じ: 試験側のシリーズ群 → test_series、残りはページ名の sha1 mod 100 < 20 → test_panel。
出力: labels/face_queue_v1.csv。既にあれば上書きしない。
"""
import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_queue import FACES, GROUPS, OUT, PACK, page_key            # noqa: E402

SEED = 20261009


def main():
    if (OUT / "face_queue_v1.csv").exists():
        sys.exit("labels/face_queue_v1.csv は既にある(順番と取り置きは固定。作り直さない)")
    plan = json.load(open(OUT / "holdout_plan.json"))
    prow = list(csv.DictReader(open(PACK / "panels.csv")))
    works = {(r["source"], r["name"]): r["work"] for r in csv.DictReader(open(PACK / "panel_works.csv"))}
    w2g = {w: g for g, ws in json.load(open(GROUPS))["groups"].items() for w in ws}
    faces = []
    for t in csv.DictReader(open(FACES)):
        if int(t["box"]) <= 0:
            continue
        pid = int(t["panel"]); r = prow[pid]
        work = works[(r["source"], r["name"])]
        g = w2g[work]; page = page_key(r["name"])
        if g in plan["test_series_groups"]:
            usage = "test_series"
        else:
            usage = "test_panel" if int(hashlib.sha1(page.encode()).hexdigest(), 16) % 100 < plan["panel_mod"] else "train"
        faces.append(dict(panel=pid, box=int(t["box"]), conf=t["conf"], x0=t["x0"], y0=t["y0"], x1=t["x1"], y1=t["y1"],
                          usage=usage, work=work, group=g, page=page))
    perm = np.random.default_rng(SEED).permutation(len(faces))
    queue = [faces[j] for j in perm]
    for o, f in enumerate(queue):
        f["order"] = o
    cols = ["order", "panel", "box", "conf", "x0", "y0", "x1", "y1", "usage", "work", "group", "page"]
    with open(OUT / "face_queue_v1.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader(); wr.writerows(queue)
    use = Counter(f["usage"] for f in faces)
    print(json.dumps(dict(n_faces=len(faces), usage=dict(use), frac={k: round(v / len(faces), 3) for k, v in use.items()},
                          first100=dict(Counter(f["usage"] for f in queue[:100]))), ensure_ascii=False))

    # まとまり単位の queue と、同じコマが同じ側になっていることの確認
    pu = {f["panel"]: f["usage"] for f in faces}
    bad = sum(1 for t in csv.DictReader(open(OUT / "queue_v1.csv")) if pu[int(t["panel"])] != t["usage"])
    print("まとまり単位の queue と用途が食い違うコマ:", bad)
    assert bad == 0


if __name__ == "__main__":
    main()
