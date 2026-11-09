#!/usr/bin/env python3
"""遮蔽の深さの第2段: 作者に出す 40層を選ぶ(登録 改訂3 §3)。

    ./venv/bin/python tools/occlusion/select_edge_stage2.py <out.csv>

主の母集団(採点で読んだ層)の near_16(R) ≥ 0.8 / ≤ 0.2 の帯から各 20層、1頁1層まで、seed 20261109。
出す順も同じ乱数で決める。帯は CSV に残すが、作者の画面には出さない。出力は原稿の名前を含むので追跡しない場所へ。
"""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_occlusion_edge import FLOOR, load_population, near  # noqa: E402

SEED, N = 20261109, 20


def main():
    out = Path(sys.argv[1])
    rows, excl, _, _ = load_population()
    read = []
    for r in rows:
        if (r["page_path"], r["layer_id"]) in excl or int(r["n_cov"]) < FLOOR:
            continue
        if not any(int(r[f"s{k}_n_cov"]) >= FLOOR for k in range(1, 6)):
            continue
        read.append((r, near(r, "r_", 16, "n_cov")))
    print(f"読んだ層(採点の主と同じはず 11,499): {len(read):,}")
    rng = random.Random(SEED)
    picked = []
    for band, test in (("hi", lambda x: x >= 0.8), ("lo", lambda x: x <= 0.2)):
        pool = sorted((r for r, n in read if test(n)), key=lambda r: (r["page_path"], int(r["layer_id"])))
        seen, got = set(), []
        for r in rng.sample(pool, len(pool)):
            if r["page_path"] in seen:
                continue
            seen.add(r["page_path"])
            got.append(r)
            if len(got) == N:
                break
        print(f"帯 {band}: 層 {len(pool):,} / 頁 {len({r['page_path'] for r in pool}):,} → 選んだ {len(got)}")
        picked += [(band, r) for r in got]
    order = rng.sample(range(len(picked)), len(picked))
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_no", "band", "page_path", "layer_id", "n_cov", "near16_r", "has_clip_occluder", "full_canvas_occluder"])
        for pos, i in enumerate(order, 1):
            band, r = picked[i]
            w.writerow([f"e{pos:02d}", band, r["page_path"], r["layer_id"], r["n_cov"],
                        f"{near(r, 'r_', 16, 'n_cov'):.4f}", r["has_clip_occluder"], r["full_canvas_occluder"]])
    print(f"書いた: {out}(seed {SEED}、{len(picked)} 行)")


if __name__ == "__main__":
    main()
