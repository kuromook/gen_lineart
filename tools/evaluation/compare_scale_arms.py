#!/usr/bin/env python
"""Compare the scale-curve arms at MATCHED steps on the deciding column.

Hypothesis 5 asks whether more distinct training pairs make the model draw GT
strokes its conditioning map lacks. The column that answers it is
"GT drawn: cond lacks it" from output_vs_condition_proximity; f1 cannot, since
it largely measures how faithfully the output copied its condition.

Read the trajectory, not the endpoint: the 460-pair arm swings 0.180 -> 0.090 ->
0.249 across snapshots while "cond has it" swings with it, so a single snapshot
difference between arms says little on its own.
"""
import argparse, re
from pathlib import Path

ROW = re.compile(r"^(step_\d+|final)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\|\s*([\d.]+)\s+([\d.]+)")


def read(path):
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        m = ROW.match(line.strip())
        if m:
            out[m.group(1)] = dict(zip(
                ("gt_only", "cond_only", "both", "neither", "cond_has", "cond_lacks"),
                (float(m.group(i)) for i in range(2, 8))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=[
        "460=results/scale_curve_20260916/scale_460/output_vs_condition_proximity.txt",
        "1837=results/h34_alignment_probe_20260915/control/output_vs_condition_proximity.txt",
        "8467=results/scale_curve_20260916/scale_8467/output_vs_condition_proximity.txt"])
    a = ap.parse_args()

    arms = {}
    for spec in a.arms:
        name, path = spec.split("=", 1)
        d = read(path)
        if d:
            arms[name] = d
        else:
            print(f"(まだ無い: {name} <- {path})")
    if not arms:
        return
    steps = sorted({s for d in arms.values() for s in d}, key=lambda s: (s != "final", int(s.split("_")[1]) if "_" in s else 0))

    # The raw column confounds "draws the missing strokes" with "draws more of
    # everything": within each arm cond_lacks and cond_has rise and fall together
    # (the 460 arm dips at 1500 and jumps at 2290 on both). The ratio asks what
    # share of the model's GT coverage goes to strokes its condition does NOT
    # supply -- the quantity hypothesis 5 is actually about.
    print("\n=== 被覆量との連動(正規化が要る理由) ===")
    print(f"{'arm':<10}{'r(cond_has, cond_lacks)':>26}")
    for n, d in arms.items():
        xs = [d[s2]["cond_has"] for s2 in steps if s2 in d]
        ys = [d[s2]["cond_lacks"] for s2 in steps if s2 in d]
        if len(xs) > 2:
            mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
            cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
            vx = sum((x-mx)**2 for x in xs) ** 0.5
            vy = sum((y-my)**2 for y in ys) ** 0.5
            print(f"{n+'ペア':<10}{(cov/(vx*vy) if vx*vy else float('nan')):>26.3f}")

    print("\n=== 正規化: 条件に無い線 ÷ 条件にある線 (被覆量を除いた選択性) ===")
    print(f"{'step':<10}" + "".join(f"{n + 'ペア':>14}" for n in arms))
    for s2 in steps:
        row = f"{s2:<10}"
        for n, d in arms.items():
            if s2 in d and d[s2]["cond_has"] > 0:
                row += f"{d[s2]['cond_lacks'] / d[s2]['cond_has']:>14.3f}"
            else:
                row += f"{'-':>14}"
        print(row)

    for col, label in (("cond_lacks", "条件画像に無いGTの線を描いた割合 (生値・決め手)"),
                       ("cond_has", "条件画像にあるGTの線を描いた割合 (参考: 模写の忠実さ)"),
                       ("gt_only", "出力の骨格のうち GTだけの近く")):
        print(f"\n=== {label} ===")
        print(f"{'step':<10}" + "".join(f"{n + 'ペア':>14}" for n in arms))
        for s in steps:
            row = f"{s:<10}"
            for n, d in arms.items():
                row += f"{d[s][col]:>14.3f}" if s in d else f"{'-':>14}"
            print(row)


if __name__ == "__main__":
    main()
