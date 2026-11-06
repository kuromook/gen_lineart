#!/usr/bin/env python3
"""**軸が作者の分類を再現するかを測る**(規律13、2026-10-31)。

なぜ要るか: 条例1 の門は**腕の対ごとの分離 AUC** を見る。分離が大きければ「計算で答が出る
ので人に聞かない」と読むが、**それは「その軸が作者の判定を再現する」とは別のこと**である
—— 分離は腕どうしの違いを言うだけで、作者がどちらを使えると言うかは言わない。
規律13 は代用の前に**的中率と取りこぼしの印字**を要求する。これはその器具。

出すもの: 腕ごとの行 / 2×2 / 的中率 / 取りこぼし / 誤って通した率。
**閾値はこの記録に合わせて選ぶので、出る数は上限**(hold-out ではない)。それも印字する。
**落とした記録は理由別に数えて印字する**(規律9)。

使い方:
  axis_vs_verdicts.py --records <判定の記録.json> \
      --arm-path rough=results/.../stroke_projected/rough/{tile}_out.png \
      --arm-path gt=/abs/dir/{tile}.jpg [--jpeg-roundtrip 95] [--positive usable]

終了コード: 0 = 測れた / 2 = 走れない。**代用を許すかどうかは判定しない** —— それは門と人。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_axis_separation import AXES, axis_values  # noqa: E402


def best_threshold(vals, labels):
    """精度を最大にする閾値と向きを返す。**この記録に合わせているので上限**である。"""
    cands = sorted(set(vals))
    mids = [(a + b) / 2 for a, b in zip(cands, cands[1:])] or [cands[0]]
    best = None
    for t in mids:
        for sign in (+1, -1):
            pred = [(v >= t) if sign > 0 else (v <= t) for v in vals]
            acc = float(np.mean([p == l for p, l in zip(pred, labels)]))
            if best is None or acc > best[0]:
                best = (acc, t, sign)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--arm-path", action="append", required=True,
                    help="<腕>=<{tile} を含むパスの型>。腕ごとに1つ")
    ap.add_argument("--positive", default="usable")
    ap.add_argument("--jpeg-roundtrip", type=int, default=None, metavar="Q")
    a = ap.parse_args()

    tmpl = {}
    for s in a.arm_path:
        if "=" not in s:
            print(f"**型が読めない**: {s}", file=sys.stderr)
            return 2
        k, v = s.split("=", 1)
        tmpl[k.strip()] = v.strip()

    rec = json.loads(Path(a.records).read_text())
    items = list(rec.items()) if isinstance(rec, dict) else list(enumerate(rec))

    rows, dropped = [], {}
    for _id, r in items:
        arm, tile, verdict = r.get("arm"), r.get("tile"), r.get("verdict")
        if arm not in tmpl:
            dropped[f"腕の型が無い({arm})"] = dropped.get(f"腕の型が無い({arm})", 0) + 1
            continue
        p = Path(tmpl[arm].format(tile=tile))
        if not p.exists():
            dropped["画像が無い"] = dropped.get("画像が無い", 0) + 1
            continue
        if verdict is None:
            dropped["判定が無い"] = dropped.get("判定が無い", 0) + 1
            continue
        rows.append((arm, tile, verdict == a.positive, axis_values(p, a.jpeg_roundtrip)))

    print(f"══ 軸は作者の分類を再現するか  記録 {Path(a.records).name}")
    print(f"   正規化: " + (f"全画像を JPEG quality {a.jpeg_roundtrip} の経路に通した"
                            if a.jpeg_roundtrip is not None else "**無し**"))
    for reason, n in sorted(dropped.items()):
        print(f"   落とした: {reason} {n} 件")
    print(f"   使った記録 {len(rows)} 件 / 全 {len(items)} 件  "
          f"—— 正 `{a.positive}` {sum(1 for r in rows if r[2])} 件")
    if not rows:
        print("**使える記録が無い。**")
        return 2

    for axis in AXES:
        vals = [r[3][axis] for r in rows]
        labels = [r[2] for r in rows]
        acc, t, sign = best_threshold(vals, labels)
        pred = [(v >= t) if sign > 0 else (v <= t) for v in vals]
        tp = sum(1 for p, l in zip(pred, labels) if p and l)
        fn = sum(1 for p, l in zip(pred, labels) if not p and l)
        fp = sum(1 for p, l in zip(pred, labels) if p and not l)
        tn = sum(1 for p, l in zip(pred, labels) if not p and not l)
        hit = tp / (tp + fn) if tp + fn else None
        miss = fn / (tp + fn) if tp + fn else None
        fpr = fp / (fp + tn) if fp + tn else None
        print()
        print(f"   軸 {axis}  —— 閾値 {t:.4f} で「{'≧' if sign > 0 else '≦'} なら使える」"
              f"(**この記録に合わせて選んだ閾値。出る数は上限**)")
        print(f"     的中率 {hit:.3f} ({tp}/{tp+fn})  取りこぼし {miss:.3f} ({fn}/{tp+fn})  "
              f"誤って通した率 {fpr:.3f} ({fp}/{fp+tn})  精度 {acc:.3f}")
        print(f"     腕                 n   作者「使える」   軸「使える」   食い違い")
        for arm in sorted({r[0] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r[0] == arm]
            na = sum(1 for i in idx if labels[i])
            np_ = sum(1 for i in idx if pred[i])
            dis = sum(1 for i in idx if pred[i] != labels[i])
            print(f"     {arm:16s} {len(idx):3d}   {na:12d}   {np_:12d}   {dis:8d}")
    print()
    print("   **この器具は代用を許さない。** 規律13 が要求するのは印字で、許可ではない ——"
          "再現しないなら軸は作者の判定の代わりにならず、再現しても閾値を合わせた分は上限である。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
