#!/usr/bin/env python
"""IC-k6(2026-10-16 事前登録、修正 B2): 橋の画素は JOINT の画素を超えない。

畳まれた橋のうち `keep` に落ちたもの(損失 1)は `assigned` に入らないので、定義上
`export_panel_tokens_v2.py` の JOINT 残渣に含まれる。コマごとに
`損失 1 の画素数 <= JOINT の n_px の総和` が成り立つはず。超えたら器具の誤りなので止める。

JOINT は節点・ヒゲ・橋・min_len 未満の残りを連結成分ごとに 1 つにまとめているので、
橋だけを取り出すことはできない。**置き換えではなく照合にだけ使う。**
"""
import csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TOKENS = Path("/home/sh1/deepl/lineart-stroke-grammar/results/panel_tokens_v2_20260918")
D = ROOT / "results/k4a_bridge_loss_20261016"

rows = list(csv.DictReader(open(D / "per_panel.csv")))
out, bad, miss = [], [], 0
for r in rows:
    f = TOKENS / r["source"] / (r["name"] + ".npz")
    if not f.exists():
        miss += 1; continue
    j = np.load(f)["joints"]
    jpx = float(j[:, 3].sum()) if len(j) else 0.0
    l1 = float(r["loss1_dropped_px"])
    ok = l1 <= jpx
    out.append(dict(pid=int(r["pid"]), joint_n=int(len(j)), joint_px=int(jpx),
                    loss1_px=int(l1), share=round(l1 / jpx, 4) if jpx else None,
                    joint_radius_median=round(float(np.median(j[:, 2])), 3) if len(j) else None,
                    ok=int(ok)))
    if not ok:
        bad.append(int(r["pid"]))

share = np.array([o["share"] for o in out if o["share"] is not None], float)
rad = np.array([o["joint_radius_median"] for o in out if o["joint_radius_median"] is not None], float)
chk = dict(n_panels=len(out), missing_npz=miss, violations=len(bad), violating_pids=bad[:20],
           passed=bool(out and not bad and miss == 0),
           loss1_over_joint_px=dict(median=round(float(np.median(share)), 4),
                                    p10=round(float(np.percentile(share, 10)), 4),
                                    p90=round(float(np.percentile(share, 90)), 4)),
           joint_radius_px=dict(median=round(float(np.median(rad)), 3)))
with open(D / "ic_k6_joints.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
json.dump(chk, open(D / "ic_k6.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(chk, ensure_ascii=False, indent=1))
print("IC-k6:", "合格" if chk["passed"] else "不合格 —— 止めて相談する")
