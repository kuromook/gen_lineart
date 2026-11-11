"""Continuation of the author's sketch eye check on the v5 pairs (2026-11-12). User 11-12: D keeps (2) what is still
mixed into v5 sketches and (3) the out-of-sample check of the dark-region rule; 'partial sketch' and 'line-side pattern'
go to Track Q, so the choices are NOT extended here (same six as 11-10).

Batch: N pages drawn with a fixed seed from the v5 measured population (1,281), minus pages already answered under any
version. Random rather than work order so one batch does not come from one or two works. Before the batch is shown,
the rule (gray<60, 9x9 opening, largest region >= 0.002 — fixed on the first 100, in-sample) is applied to the v5
sketches and written to predictions_*.csv, so the comparison afterwards is out-of-sample.
Item ids keep the 11-10 numbering (p#### = position in the v4 population) with version clip_pairs_v5."""
import argparse, csv, json, os, random, sys
import cv2, numpy as np
sys.path.insert(0, "tools/evaluation")
from pair_feature_tiles import load_jobs

D = "data/sketch_eye_check_20261110"
ROOT = "data/clip_pairs_v5/clip_pairs"
THR = 0.002


def largest_dark(path):
    s = cv2.imread(path, 0)
    m = cv2.morphologyEx((s < 60).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return 0.0 if n < 2 else float(st[1:, 4].max()) / s.size


ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--seed", type=int, default=20261112)
a = ap.parse_args()

v4 = [k for k, _, _ in load_jobs("data/clip_pairs_v4")]
pid = {k: f"p{i:04d}" for i, k in enumerate(v4)}
v5 = load_jobs("data/clip_pairs_v5", qc="clip_pairs_qc_v5.csv")
seen = {json.loads(l)["item_id"] for l in open(f"{D}/answers.jsonl", encoding="utf-8")}
pool = [j for j in v5 if pid[j[0]] not in seen]
random.Random(a.seed).shuffle(pool)          # one fixed order; batch b takes slice b
pick = pool[(a.batch - 1) * a.n: a.batch * a.n]
print(f"v5 population {len(v5)}; already answered {len(seen)}; pool {len(pool)}; batch {a.batch}: {len(pick)} pages")

spec = json.load(open(f"{D}/questions.json", encoding="utf-8"))
spec["page_title"] = f"下絵の混ざりもの v5 第{a.batch}組 2026-11-12"
spec["decides"] = "この答えで、v5 の下絵にまだ何が混ざっているかを数え、黒い面の目安が見ていない頁でも当たるかを確かめる"
spec["items"] = []
with open(f"{D}/predictions_v5_batch{a.batch}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["item_id", "largest_dark_frac_v5", "rule_fires"])
    for key, sp, lp in pick:
        frac = largest_dark(sp)
        w.writerow([pid[key], f"{frac:.6f}", int(frac >= THR)])
        wk, slug, page = key.split("|")
        spec["items"].append({
            "id": pid[key], "version": "clip_pairs_v5", "note": f"{wk} / {page}", "key": key,
            "images": [{"src": os.path.relpath(sp, ROOT), "caption": "下絵の層(問う対象)", "target": True},
                       {"src": os.path.relpath(lp, ROOT), "caption": "線画(見比べ用)"}]})
json.dump(spec, open(f"{D}/questions_v5_batch{a.batch}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("rule fires on", sum(1 for l in open(f"{D}/predictions_v5_batch{a.batch}.csv") if l.rstrip().endswith(",1")), "of", len(pick))
