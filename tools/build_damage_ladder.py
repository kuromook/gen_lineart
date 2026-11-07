#!/usr/bin/env python3
"""**判定者の閾値を測るための梯子を作る**(2026-11-07、ユーザーの決定で着手)。

なぜ要るか: いまの脚は**床と天井に張り付いている**(`rough` 0/15、`gt` 14/15)。飽和した
項目では **(a) 棒が動いたかどうか**(規律10)も **(b) 判定者の一貫性**も測れない ——
2回見せればほぼ確実に一致するが、それは項目が易しいことの反映で、分解能ではない。
**だから判定が反転する場所を先に作る。**

**これは手法についての判定ではない。** 梯子は GT から墨を段階的に落としただけのもので、
どのモデルの出力も入れない。**lesson 10(削除方向の天井は0)は動かない** —— 梯子が返すのは
「作者がどこで『使えない』と言い始めるか」で、削除の手法が使えるかどうかではない。
条例5 が禁じているのは**手法の判定を頼み直すこと**で、判定者そのものの校正は別物である。

**落とし方は宣言する。** ここでは**墨の連結成分を丸ごと**消す(無作為の順、seed 固定)——
画素を点々と抜くと「線が壊れた」側に早く振れるので、梯子が潰れる。**閾値は(判定者 × 落とし方)の
性質で、落とし方を変えれば動く。** だから落とし方を名前で登録に書く。

**段ごとに「実際に落ちた割合」を測って印字する**(狙いの割合ではなく実測。成分を丸ごと消すので
狙いを超える)。**落としたタイルは理由別に数えて印字する**(規律9)。

使い方: build_damage_ladder.py [--per-level 15] [--seed 20261107] [--montage]
終了コード: 0 = 作れた / 2 = 走れない。**判定は作らない。** 集めるかどうかは門と人。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
GT = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line")
TILES = Path("/home/sh1/deepl/lineart/dataset/pairs_480/holdout_lineart_family.txt")
OUT = ROOT / "results/judge_damage_ladder_20261107"

# **段は先に書く。** 0 は GT そのもの(= 天井の脚)。等間隔にしないのは、反転が
# どちら寄りでも挟めるようにするため
LEVELS = (0.00, 0.10, 0.20, 0.35, 0.50, 0.70)
# **成分を取る閾値は緩く、量は濃さで測る。** 最初は `g < 128`(芯だけ)で取っていて、
# **反アリアスの裾が残り「消す梯子」ではなく「薄くする梯子」になった**(2026-11-07 の目視で露見)。
# 薄くする向きは Track O の材料を落とした軸そのものなので、ここで作ってはいけない。
INK = 242          # これより暗い画素を線の一部とみなす(裾を含めて1つの成分にする)
CORE = 128         # 「芯」の閾値。gt_ink(層化に使う)は従来どおり芯で測る
MIN_COMPONENTS = 4  # これ未満だと成分単位で段を作れない


def components(mask, weight):
    """成分は**裾を含む**マスクで取り、大きさは**濃さの合計**で測る(画素数ではない)。"""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return lab, np.array([])
    sizes = ndimage.sum(weight, lab, index=np.arange(1, n + 1))
    return lab, sizes


def make_level(g, weight, lab, sizes, order, target):
    """**狙いを超えない**ように、無作為の順で成分を丸ごと選んで消す。

    最初の版は「超えたら止める」にしていて、**梯子が潰れた** —— 巨大な成分が1つ在ると、
    それを消した瞬間に 0.35 の段が実測 1.00 まで跳ぶ(2026-11-07 の実測: 狙い 0.35 で
    中央値 0.63・最大 1.00)。**段は独立変数なので、跳ねたら段ではない。**
    だから**入るものだけ入れる**(超える成分は飛ばす)。結果は狙い以下になり、
    届かない絵は呼び出し側が落とす。
    """
    total = float(weight.sum())
    if not total:
        return g.copy(), 0.0, 0
    budget = target * total
    removed, drop = 0.0, []
    for idx in order:
        sz = float(sizes[idx - 1])
        if removed + sz <= budget:
            drop.append(idx)
            removed += sz
    out = g.copy()
    if drop:
        out[np.isin(lab, drop)] = 255
    return out, removed / total, len(drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-level", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--montage", action="store_true", help="原寸の目視用の帯を1枚作る(results/ 配下)")
    a = ap.parse_args()

    if not GT.is_dir() or not TILES.exists():
        print(f"**材料が無い**: {GT} / {TILES}", file=sys.stderr)
        return 2
    rng = np.random.default_rng(a.seed)

    stems = [Path(l.strip()).stem for l in open(TILES) if l.strip()]
    dropped, usable = {}, []
    for s in stems:
        f = GT / f"{s}.jpg"
        if not f.exists():
            dropped["GT が無い"] = dropped.get("GT が無い", 0) + 1
            continue
        g = np.asarray(Image.open(f).convert("L"))
        mask = g < INK
        weight = np.where(mask, (255.0 - g) / 255.0, 0.0)
        if not mask.any():
            dropped["墨が無い(白紙)"] = dropped.get("墨が無い(白紙)", 0) + 1
            continue
        lab, sizes = components(mask, weight)
        if len(sizes) < MIN_COMPONENTS:
            dropped[f"連結成分が {MIN_COMPONENTS} 未満"] = dropped.get(f"連結成分が {MIN_COMPONENTS} 未満", 0) + 1
            continue
        usable.append((s, g, weight, lab, sizes, float((g < CORE).mean())))

    need = a.per_level * len(LEVELS)
    for reason, n in sorted(dropped.items()):
        print(f"落とした: {reason} {n} 枚")
    print(f"使えるタイル {len(usable)} 枚 / 要る {need} 枚("
          f"{len(LEVELS)} 段 × {a.per_level} 枚、**段の間でタイルを重ねない**)")
    if len(usable) < need:
        print("**足りない。** 段を減らすか枚数を減らすか、母集団を広げる —— "
              "**重ねて埋めない**(同じ絵を2つの段で見ると対比が入る)", file=sys.stderr)
        return 2

    # 墨の量の五分位で層化し、**段ごとに各五分位から同数**を取る。狙いに届かない絵は
    # その場で捨てて次の候補を引く —— **段の枚数を揃えるため**(捨てた数は理由別に印字する)
    usable.sort(key=lambda r: r[5])
    buckets = [list(rng.permutation(b)) for b in np.array_split(np.arange(len(usable)), 5)]
    per_bucket = a.per_level // 5

    OUT.mkdir(parents=True, exist_ok=True)
    rows, short = [], {}
    for lv in LEVELS:
        d = OUT / f"level_{int(lv*100):02d}"
        d.mkdir(exist_ok=True)
        for q in buckets:
            got = 0
            while got < per_bucket and q:
                idx = int(q.pop())
                s_, g, weight, lab, sizes, ink = usable[idx]
                order = rng.permutation(np.arange(1, len(sizes) + 1))
                out, frac, ndrop = make_level(g, weight, lab, sizes, order, lv)
                # **狙いに届かない絵は段に入れない。** 一つの巨大な成分で墨が占められていると
                # 成分単位では刻めない —— 入れると段の意味が崩れる(規律9)
                if lv > 0 and frac < 0.7 * lv:
                    k = f"段 {lv:.2f} に届かない(成分が粗い)"
                    short[k] = short.get(k, 0) + 1
                    continue
                path = d / f"{s_}.png"
                Image.fromarray(out).save(path)
                rows.append({"level_target": lv, "tile": s_, "path": str(path),
                             "gt_ink_frac": round(ink, 5), "removed_frac": round(frac, 5),
                             "components_total": len(sizes), "components_removed": ndrop})
                got += 1
            if got < per_bucket:
                k = f"段 {lv:.2f} の五分位を埋めきれない(候補切れ)"
                short[k] = short.get(k, 0) + (per_bucket - got)
    for reason, n in sorted(short.items()):
        print(f"落とした: {reason} {n} 枚")
    dropped.update(short)

    with open(OUT / "ladder_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("══ 段ごとの**実測**(狙いではなく、実際に落ちた墨の割合)")
    print("   段(狙い)  枚数   落ちた割合 中央値   最小    最大    消した成分の中央値")
    summary = {}
    for lv in LEVELS:
        v = [r["removed_frac"] for r in rows if r["level_target"] == lv]
        c = [r["components_removed"] for r in rows if r["level_target"] == lv]
        summary[f"{lv:.2f}"] = {"n": len(v), "median": float(np.median(v)),
                                "min": float(min(v)), "max": float(max(v)),
                                "components_removed_median": float(np.median(c))}
        print(f"   {lv:>9.2f}  {len(v):4d}   {np.median(v):14.4f}  {min(v):.4f}  {max(v):.4f}"
              f"  {np.median(c):18.1f}")
    json.dump({"levels": list(LEVELS), "per_level": a.per_level, "seed": a.seed,
               "ink_threshold": INK, "core_threshold": CORE,
               "amount_measured_as": "濃さの合計 sum((255-g)/255)。画素数ではない", "operator": "連結成分を丸ごと消す(無作為の順)",
               "tiles_disjoint_across_levels": True, "dropped": dropped,
               "measured": summary},
              open(OUT / "ladder_design.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n控え: {OUT / 'ladder_manifest.csv'} / {OUT / 'ladder_design.json'}")

    if a.montage:
        # **原寸で並べる**(縮小した帯で論じない。規律6)。1行 = 1枚のタイル、列 = 段
        pick = [rows[i] for i in range(0, len(rows), max(1, len(rows) // 3))][:3]
        band = Image.new("L", (480 * len(LEVELS), 480 * len(pick)), 255)
        for r_i, r in enumerate(pick):
            s = r["tile"]
            g = np.asarray(Image.open(GT / f"{s}.jpg").convert("L"))
            mask = g < INK
            weight = np.where(mask, (255.0 - g) / 255.0, 0.0)
            lab, sizes = components(mask, weight)
            o = np.random.default_rng(a.seed).permutation(np.arange(1, len(sizes) + 1))
            for c_i, lv in enumerate(LEVELS):
                out, _, _ = make_level(g, weight, lab, sizes, o, lv)
                band.paste(Image.fromarray(out), (480 * c_i, 480 * r_i))
        mp = OUT / "ladder_band_native.png"
        band.save(mp)
        print(f"目視用(原寸、列=段 / 行=タイル): {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
