#!/usr/bin/env python3
"""梯子の聞き取りを組む(`config/judge_protocol_ladder_20261107.json` の通りに)。

**この器具は判定を作らない。** 作るのは、作者に見せる問いの JSON と、見せる絵と、
**どの問いがどの段だったかの対応表**である。対応表は原稿の名前を含むので `data/` の下に置く。

登録に書いてあって、ここで守るもの:
- **段 88 件 + 脚 `rough` 15 件 + 再掲 8 件 = 111 件。** 段のタイルと脚のタイルは重ねない
- **正規化**: 全部 JPEG quality 95 の経路に通す(段の出力は PNG、GT は JPEG で出てくる)
- **層化**: 脚のタイルも GT の墨の量(芯 `g < 128`)の五分位から 3 枚ずつ
- **順番は無作為、再掲は後半に置く。位置と直前の項目は画面が行に残す**(`interview_core.html`)
- **画面に段の名前を出さない。** 絵の名前は問いの ID だけ(`img/a0007.jpg`)

落としたものは理由別に数えて印字する(規律9)。
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "results/judge_damage_ladder_20261107"
ROUGH = ROOT / "results/trackc_judge_20261004/stroke_projected/rough"
GT = Path("/home/sh1/deepl/lineart-controlnet-sd15-refine/data/holdout_lineart_family_gt_line")
TILES = Path("/home/sh1/deepl/lineart/dataset/pairs_480/holdout_lineart_family.txt")
OUT = ROOT / "data/judge_ladder_20261107"

INTERVIEW_ID = "ladder_20261108_threshold"
JPEG_Q = 95          # 登録の `normalisation`
CORE = 128           # 層化に使う「芯」の閾値(梯子と同じ)
N_LEG = 15
N_QUINTILES = 5
N_REPEATS = 8
SEED = 20261108      # **聞き取りの組み方の seed。梯子の seed(20261107)とは別**


def jpeg95(src: Path, dst: Path):
    """**見せる絵は全部この経路を通る。** 腕ごとに形式が違うと、軸が圧縮を読む(Track N)。"""
    im = Image.open(src).convert("L")
    assert im.size == (480, 480), f"480px でない: {src} {im.size}"
    im.save(dst, "JPEG", quality=JPEG_Q, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    rng = np.random.default_rng(a.seed)

    man = LADDER / "ladder_manifest.csv"
    if not man.is_file():
        sys.exit(f"梯子の材料が無い: {man} —— tools/build_damage_ladder.py で作り直す")
    rungs = list(csv.DictReader(open(man, encoding="utf-8")))
    print(f"段: {len(rungs)} 件")
    used = {r["tile"] for r in rungs}

    # ---- 脚 `rough`: 段で使っていないタイルから、芯の五分位ごとに3枚 ----
    all_tiles = [Path(l.strip()).stem for l in open(TILES) if l.strip()]
    drop = Counter()
    cand = []
    for t in all_tiles:
        if t in used:
            drop["段で使った(脚に回さない)"] += 1
            continue
        if not (ROUGH / f"{t}_out.png").is_file():
            drop["脚の絵が無い"] += 1
            continue
        g = np.asarray(Image.open(GT / f"{t}.jpg").convert("L").resize((480, 480)))
        ink = float((g < CORE).mean())
        if ink == 0.0:
            drop["GT が白紙"] += 1
            continue
        cand.append((t, ink))
    cand.sort(key=lambda x: x[1])
    per_q = N_LEG // N_QUINTILES
    legs = []
    for b in np.array_split(np.array([t for t, _ in cand], dtype=object), N_QUINTILES):
        idx = rng.choice(len(b), size=per_q, replace=False)
        legs += [str(b[i]) for i in idx]
    ink = dict(cand)
    print(f"脚 `rough`: {len(legs)} 件(候補 {len(cand)} 枚から五分位ごとに {per_q} 枚)")
    for k, b in enumerate(np.array_split(np.array([t for t, _ in cand], dtype=object), N_QUINTILES)):
        got = [t for t in legs if t in set(map(str, b))]
        print(f"  五分位{k+1}: 候補 {len(b)} 枚(芯 {ink[str(b[0])]:.4f}-{ink[str(b[-1])]:.4f}) → {len(got)} 枚")
    print("落としたもの(理由別、規律9):")
    for k, v in drop.items():
        print(f"  {k}: {v}")
    assert not (set(legs) & used), "脚のタイルが段と重なっている"

    # ---- 順番(無作為)→ ID を振る。**ID は順番の通し番号で、段を漏らさない** ----
    base = [{"arm": f"level_{int(round(float(r['level_target'])*100)):02d}", "tile": r["tile"],
             "src": Path(r["path"]), "removed_frac": float(r["removed_frac"]),
             "gt_ink": float(r["gt_ink_frac"])} for r in rungs]
    for i, r in enumerate(base):
        if r["arm"] == "level_00":
            base[i]["arm"] = "gt"
    base += [{"arm": "rough", "tile": t, "src": ROUGH / f"{t}_out.png",
              "removed_frac": None, "gt_ink": ink[t]} for t in legs]
    order = rng.permutation(len(base))
    base = [base[i] for i in order]
    for i, r in enumerate(base):
        r["id"] = f"a{i:03d}"

    # ---- 再掲: 後半から8件。**健全性の確認で、一貫性の測定ではない**(登録 `_repeats`) ----
    half = len(base) // 2
    pick = sorted(rng.choice(range(half, len(base)), size=N_REPEATS, replace=False))
    reps = [{**base[k], "id": f"r{j}", "repeat_of": base[k]["id"]} for j, k in enumerate(pick)]
    items = [dict(r, repeat_of="") for r in base]
    pos = sorted(rng.choice(range(half, len(items)), size=N_REPEATS, replace=False))
    for p, rw in zip(reversed(pos), reversed(reps)):
        items.insert(p, rw)
    print(f"項目: {len(items)} 件(素 {len(base)} + 再掲 {N_REPEATS}、再掲は後半に置いた)")

    # ---- 絵を staging(全部 JPEG q95) ----
    img = out / "img"
    if img.exists():
        shutil.rmtree(img)
    img.mkdir(parents=True)
    for r in base:
        jpeg95(r["src"], img / f"{r['id']}.jpg")
    total = sum(p.stat().st_size for p in img.glob("*.jpg"))
    print(f"絵: {len(base)} 枚, {total/1e6:.2f} MB(JPEG q{JPEG_Q}) —— data URI にすると約 {total*4/3/1e6:.2f} MB")

    # ---- 問いの JSON(画面に出る文字。段の名前は出さない) ----
    spec = {
        "interview_id": INTERVIEW_ID,
        "page_title": "線画の判定(梯子 2026-11-07)",
        "question": "この絵は、線画として使えますか。",
        "unit": "いま見えている絵1枚(480画素の切り出し)。ほかの絵や元の絵とは比べません",
        "decides": "どこまで線が欠けると「使えない」に変わるかが決まります(採点器の上限を測れる場所を探しています)",
        "choices": [
            {"value": "usable", "label": "使える", "help": "線画として通る"},
            {"value": "not_usable", "label": "使えない", "help": "線画として通らない"},
        ],
        "initial_view": "native",
        "_initial_view": "**等倍(1:1)で始める。** 縮小すると細い線は薄く・途切れて見える —— それはこの収集が測っている軸そのものなので、画面の縮小が答えを作る。",
        "batch_size": 40,
        "items": [{"id": r["id"], "version": "jpeg95",
                   "images": [{"src": f"img/{(r['repeat_of'] or r['id'])}.jpg", "caption": "判定する絵"}]}
                  for r in items],
    }
    (out / "questions.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")

    with open(out / "items_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pos", "id", "arm", "tile", "removed_frac", "gt_ink", "repeat_of", "prev_id"])
        for i, r in enumerate(items):
            w.writerow([i, r["id"], r["arm"], r["tile"],
                        "" if r["removed_frac"] is None else f"{r['removed_frac']:.5f}",
                        f"{r['gt_ink']:.5f}", r["repeat_of"], items[i-1]["id"] if i else ""])
    json.dump({"interview_id": INTERVIEW_ID, "seed": a.seed, "jpeg_quality": JPEG_Q,
               "registration": "config/judge_protocol_ladder_20261107.json",
               "n_items": len(items), "n_base": len(base), "n_repeats": N_REPEATS,
               "legs_from_unused_tiles": len(cand), "dropped": dict(drop),
               "arms": sorted({r["arm"] for r in base}),
               "order": "無作為(この seed)。位置と直前の項目は画面が1行ごとに残す"},
              open(out / "design_interview.json", "w"), ensure_ascii=False, indent=1)
    print(f"問い: {out/'questions.json'}")
    print(f"対応表: {out/'items_manifest.csv'}(**原稿の名前を含むので追跡しない**)")


if __name__ == "__main__":
    main()
