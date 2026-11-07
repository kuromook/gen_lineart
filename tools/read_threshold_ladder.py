#!/usr/bin/env python3
"""梯子の聞き取りを**登録した読む順で**読む(`config/judge_protocol_ladder_20261107.json` の `read_order`)。

① 脚(`gt` / `rough`)—— **脚がずれていたら、段の読みに進まない**
② 段ごとの通過率を**段の順に**。反転が挟めたか
③ 再掲8件の一致(健全性の確認。一貫性の測定ではない)
④ 軸

**通過率は2通り印字する**(登録 `bar`): 「使える ÷ 全件」が主、「使える ÷(使える+使えない)」が副。
**保留(分からない / どれでもない)は段ごとに名前を付けて別に印字し、分母に畳まない。**
**選択肢が anchor-20261004 と違う**(保留を足した)ので、脚を並べるときはそう印字する。
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/judge_ladder_20261107"
RUNGS = ["gt", "level_10", "level_20", "level_35", "level_50", "level_70"]
MEASURED = {"gt": 0.000, "level_10": 0.094, "level_20": 0.188,
            "level_35": 0.313, "level_50": 0.454, "level_70": 0.587}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DATA))
    a = ap.parse_args()
    d = Path(a.dir)
    man = {r["id"]: r for r in csv.DictReader(open(d / "items_manifest.csv", encoding="utf-8"))}
    rows = [json.loads(l) for l in open(d / "answers.jsonl", encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["ts"])
    latest = {}
    for r in rows:
        latest[r["item_id"]] = r

    print("══ 梯子の読み出し(登録した読む順)")
    print(f"   行 {len(rows)} / 問い {len(man)} / 答えのある問い {len(latest)}")
    scales = Counter(round(r.get("shown_scale") or -1, 3) for r in latest.values())
    print(f"   見せた倍率: {dict(sorted(scales.items()))} —— **等倍で見ていない項目はここに出る**")
    sess = Counter(r["session"] for r in latest.values())
    print(f"   画面の通し: {dict(sess)}(1回で取り切ったか)")

    # ---- ① 脚 ----
    print("\n── ① 脚(ここがずれていたら段の読みに進まない)")
    print("   **選択肢が anchor-20261004 と違う**(「どれでもない」「分からない」を足した)。"
          "並べるときは必ずそう書く")
    for leg in ("gt", "rough"):
        ids = [i for i, m in man.items() if m["arm"] == leg and not m["repeat_of"]]
        vals = Counter(latest[i]["value"] for i in ids if i in latest)
        k, n = vals.get("usable", 0), len(ids)
        lo, hi = wilson(k, n)
        old = {"gt": "14/15", "rough": "0/15"}[leg]
        print(f"   {leg:6s} 使える {k}/{n}  95%CI {lo:.3f}-{hi:.3f}   "
              f"(2026-10-04 の同じ脚: {old})  内訳 {dict(vals)}")

    # ---- ② 段 ----
    print("\n── ② 段ごとの通過率(段の順)")
    print(f"   {'段':9s} {'落ちた墨(実測中央値)':>12s}  {'使える/全件':>12s} {'主':>6s}  "
          f"{'使える/(使える+使えない)':>14s}  保留")
    rates = {}
    for arm in RUNGS:
        ids = [i for i, m in man.items() if m["arm"] == arm and not m["repeat_of"]]
        vals = Counter(latest[i]["value"] for i in ids if i in latest)
        n = len(ids)
        k = vals.get("usable", 0)
        held = vals.get("__unknown__", 0) + vals.get("__none__", 0)
        dec = k + vals.get("not_usable", 0)
        rates[arm] = k / n if n else float("nan")
        sub = f"{k}/{dec}" + (f" = {k/dec:.3f}" if dec else "")
        print(f"   {arm:9s} {MEASURED[arm]:>12.3f}  {k:>5d}/{n:<6d} {rates[arm]:>6.3f}  "
              f"{sub:>14s}  分からない {vals.get('__unknown__',0)} / どれでもない {vals.get('__none__',0)}"
              + ("  **保留が多い = 境目の在処の証拠**" if held >= 3 else ""))

    crossed = [(a, b) for a, b in zip(RUNGS, RUNGS[1:])
               if (rates[a] - 0.5) * (rates[b] - 0.5) < 0]
    print()
    if crossed:
        for a, b in crossed:
            print(f"   **反転を挟んだ**: {a}({rates[a]:.3f}) と {b}({rates[b]:.3f}) の間で 0.5 を跨ぐ")
    else:
        side = "上" if all(r > 0.5 for r in rates.values()) else ("下" if all(r < 0.5 for r in rates.values()) else "まちまち")
        print(f"   **反転を挟めなかった** —— どの段も通過率が 0.5 の同じ側({side})に留まった。")
        print("   登録の反証条件: **結論ではなく梯子を伸ばす**(0.70 より先、または 0.10 より手前)。")
        print("   **この収集からは閾値を言えない。** 段は 15枚なので分解能は 1/15。")

    # ---- ③ 再掲 ----
    print("\n── ③ 再掲(健全性の確認。一貫性の測定ではない —— 飽和した段では易しさを測る)")
    agree = 0
    reps = [(i, m) for i, m in man.items() if m["repeat_of"]]
    for i, m in sorted(reps, key=lambda x: int(x[1]["pos"])):
        o = m["repeat_of"]
        a_, b_ = latest.get(o, {}).get("label"), latest.get(i, {}).get("label")
        same = a_ == b_
        agree += same
        print(f"   {o} → {i}  {m['arm']:9s} 位置 {man[o]['pos']:>3s}→{m['pos']:>3s}  "
              f"{a_} / {b_}  {'一致' if same else '**食い違い**'}")
    print(f"   一致 {agree}/{len(reps)}" + (f" = {agree/len(reps):.3f}" if reps else ""))

    # ---- ④ 軸 ----
    print("\n── ④ 軸")
    ax = ROOT / "results/judge_damage_ladder_20261107/ladder_axes.csv"
    print(f"   {ax}(両端 mean_ink 0.836 / near_white_frac 0.887、隣り合う段は 0.53〜0.72)")
    print("   **段が濃さで分かれるのは作った結果**であって、判定を再現するという話ではない(規律13)")

    # ---- 再計算器が読める形でも書く(登録簿の数を手で打たない) ----
    vd = [{"arm": man[i]["arm"], "tile": man[i]["tile"], "item_id": i,
           "repeat_of": man[i]["repeat_of"],
           "verdict": "usable" if latest[i]["value"] == "usable" else latest[i]["value"]}
          for i in sorted(latest, key=lambda x: int(man[x]["pos"])) if not man[i]["repeat_of"]]
    out = d / "verdicts_by_arm.json"
    out.write_text(json.dumps(vd, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n   再計算器が読む形: {out}(**登録簿の数はここから引き直される**。手で打たない)")

    # ---- 一言 ----
    cm = [(i, latest[i]["label"], latest[i]["comment"], man[i]["arm"]) for i in latest
          if latest[i].get("comment")]
    if cm:
        print("\n── 一言(作者の言葉。器具の言葉に翻訳しない)")
        for i, lab, c, arm in cm:
            print(f"   {i} {arm:9s} {lab}: 「{c}」")


if __name__ == "__main__":
    main()
