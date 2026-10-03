"""Read back the judgements and answer the three questions, in order.

Input is the artifact's `judgments` collection dumped to JSON (a list of
documents, or a {id: doc} map). Nothing is fitted here -- this is the stage-1
readout that decides whether stage 2 is allowed to start.

Order matters and is pre-registered:

  0. intra-rater consistency on the repeated pairs. It is the ceiling on any
     scorer, and the stage-2 gate is defined as 0.85 x this number.
  1. INSTRUMENT CHECK -- oracle vs placebo_oracle. If the judge does not prefer
     the ideal deletion over a random one of the same size, the eye has no
     resolution on this material and nothing below it may be read.
  2. classifier vs placebo -- did Track C delete the RIGHT lines?
  3. classifier vs rough -- was deleting an improvement at all? Reported
     separately and never pooled: this is the one pairing that cannot be
     ink-matched.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results/trackc_judge_20261004"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(path):
    raw = json.loads(Path(path).read_text())
    docs = raw if isinstance(raw, list) else list(raw.values())
    out = {}
    for d in docs:
        d = d.get("data", d)
        if d.get("pairId"):
            out[d["pairId"]] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgements", required=True, help="dump of the judgments collection")
    ap.add_argument("--pairs", default=str(ROOT / "pairs.csv"))
    args = ap.parse_args()

    j = load(args.judgements)
    pairs = {r["id"]: r for r in csv.DictReader(open(args.pairs))}
    print(f"judgements: {len(j)} of {len(pairs)} pairs\n")

    def winner(d):
        if d["choice"] == "tie":
            return "tie"
        return d["left"] if d["choice"] == "left" else d["right"]

    # 0. intra-rater consistency ------------------------------------------
    same = tot = 0
    for pid, d in j.items():
        rep = pairs.get(pid, {}).get("repeat_of") or d.get("repeatOf")
        if not rep or rep not in j:
            continue
        tot += 1
        if winner(d) == winner(j[rep]):
            same += 1
    consistency = same / tot if tot else float("nan")
    lo, hi = wilson(same, tot)
    print(f"0. intra-rater consistency: {same}/{tot} = {consistency:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"   stage-2 gate = 0.85 x {consistency:.3f} = {0.85*consistency:.3f} hold-out agreement\n")

    # 1-3. per pairing ------------------------------------------------------
    by = defaultdict(lambda: defaultdict(int))
    per_tile = defaultdict(dict)
    for pid, d in j.items():
        row = pairs.get(pid)
        if row is None or row.get("repeat_of"):
            continue
        by[row["pairing"]][winner(d)] += 1
        per_tile[row["pairing"]][row["tile"]] = winner(d)

    order = [("1. INSTRUMENT CHECK  oracle vs placebo_oracle", "oracle_vs_placebo_oracle", "oracle"),
             ("2. DECISIVE          classifier vs placebo", "classifier_vs_placebo", "classifier"),
             ("3. SEPARATE          classifier vs rough", "classifier_vs_rough", "classifier")]
    summary = []
    for title, key, target in order:
        c = by.get(key, {})
        n_dec = sum(v for k, v in c.items() if k != "tie")
        k_t = c.get(target, 0)
        ties = c.get("tie", 0)
        rate = k_t / n_dec if n_dec else float("nan")
        lo, hi = wilson(k_t, n_dec)
        verdict = "above chance" if lo > 0.5 else ("below chance" if hi < 0.5 else "NOT separable")
        print(f"{title}")
        print(f"   {target} preferred {k_t}/{n_dec} = {rate:.3f}  (95% CI {lo:.3f}-{hi:.3f})  ties {ties}  -> {verdict}")
        summary.append({"pairing": key, "target": target, "n_decided": n_dec, "n_ties": ties,
                        "preferred": k_t, "rate": round(rate, 4) if n_dec else None,
                        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "verdict": verdict})
        if key == "oracle_vs_placebo_oracle" and n_dec and lo <= 0.5:
            print("\n   The instrument check did not pass. Nothing below this line may be read:")
            print("   if the ideal deletion is not visibly better than a random one of the same")
            print("   size, this material cannot answer whether Track C deleted the right lines.\n")

    json.dump({"n_judgements": len(j), "intra_rater": {"same": same, "n": tot,
               "consistency": round(consistency, 4) if tot else None,
               "stage2_gate": round(0.85 * consistency, 4) if tot else None},
               "pairings": summary},
              open(ROOT / "judgement_readout.json", "w"), indent=1)
    print(f"\nwritten: {ROOT / 'judgement_readout.json'}")


if __name__ == "__main__":
    main()
