#!/usr/bin/env python3
"""届いた50組の画の壊れ方を測る(作者が11問で止めた。「抽出に失敗し、画像がバグっている」2026-11-15)。

**この器具は画の壊れ方だけを測る。層の役割は判定しない**(それは作者の問い)。

測るもの(層ごと):
- **頁の黒率**: 合成が真っ黒に落ちた形(作者が見たのはこれ)。白の紙に黒が 0.15 超で札を立てる
- **層の非白の画素**が、抽出側の数えた画素(`n_px_a_gt0`)のどれだけか。**低いほど描けていない**
- **256 の塊の占有**: 非白の画素が 256×256 の塊にだけ在って、塊の中も歯抜け = 取り込み画像を描けず、
  ラスタの塊だけが出た形。`完全な塊/占めた塊` が低く、占めた塊が bbox の面積に対して少ない層
- **bbox が 256 で整列しているか**: 整列している層は、画素の数も塊の数で決まっている見込みが高い
- **bbox の中の塊の占有率**: 占めた塊 / bbox の塊の数。低い = 塊が歯抜け(pos 4・8 がこの形)

**この器具は作者の目を再現できていない**(規律13 と同じ用心)。作者が「バグっている」と言った4行
(4・8・10・11)のうち、札で拾えるのは 4(描けていない)と 11(頁が黒い)だけで、**8 と 10 は拾えない**
—— 抽出側の数えた画素自体が部分的なので、割合は 1.00 に見える。**拾えた数と取りこぼしを印字する**

**印字するもの**: 50行の表 / 札ごとの行の数と `pos` / 作者が答えた11行の内訳 / 出力の CSV。
**作者の答えは書き換えない**(規律6)。壊れた画で取った答えは**版として残し、棒には使わない**。
"""
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent.parent
D = HERE / "data/role_predictor/image_material_check50_20261114"
LIST = HERE / "data/role_predictor/image_material_check50_20261114.csv"
ANS = HERE / "data/role_predictor/image_material_check50_answers_v1_broken_images_20261115.jsonl"
OUT = HERE / "data/role_predictor/image_material_check50_faults_20261115.csv"

BLACK_BAR = 0.15   # 頁の黒率がこれを超えたら「頁が黒い」
DRAWN_BAR = 0.60   # 層の非白 / 抽出側の画素 がこれ未満なら「描けていない見込み」


def main():
    rows = list(csv.DictReader(io.open(LIST, encoding="utf-8")))
    ans = {}
    if ANS.is_file():
        for line in io.open(ANS, encoding="utf-8"):
            r = json.loads(line)
            ans[r["item_id"]] = (r.get("value"), (r.get("comment") or "").strip())

    out, flags = [], {"頁が黒い": [], "描けていない見込み": [], "bbox が 256 整列": []}
    print("pos 画素(抽出側)  層の非白  割合  占めた塊 完全な塊率  頁の黒率 塊の占有 整列  作者の答え")
    for r in rows:
        pos = int(r["pos"])
        x0, y0, x1, y1 = (int(r["bbox_" + k]) for k in ("x0", "y0", "x1", "y1"))
        lg = np.asarray(Image.open(D / f"im50_{pos:02d}_layer.png").convert("L"))
        nw = lg < 250
        n_nw = int(nw.sum())
        hb, wb = nw.shape[0] // 256, nw.shape[1] // 256
        core = nw[:hb * 256, :wb * 256].reshape(hb, 256, wb, 256).sum(axis=(1, 3))
        occ = int((core > 0).sum())
        full = int((core == 65536).sum())
        pg = np.asarray(Image.open(D / f"im50_{pos:02d}_page.png").convert("L"))
        blk = float((pg <= 5).mean())
        n_px = int(r["n_px_a_gt0"])
        ratio = n_nw / max(1, n_px)
        aligned = all(v % 256 == 0 for v in (x0, y0, x1, y1))
        bb_blocks = max(1, ((x1 - x0 + 255) // 256) * ((y1 - y0 + 255) // 256))
        occ_ratio = occ / bb_blocks
        a = ans.get(f"im50_{pos:02d}", ("", ""))
        if blk > BLACK_BAR:
            flags["頁が黒い"].append(pos)
        if ratio < DRAWN_BAR:
            flags["描けていない見込み"].append(pos)
        if aligned:
            flags["bbox が 256 整列"].append(pos)
        print(f"{pos:>3} {n_px:>12} {n_nw:>9} {ratio:>5.2f} {occ:>8} {full/max(1,occ):>9.2f} {blk:>9.3f}  "
              f"{occ_ratio:>6.2f} {'はい' if aligned else 'いいえ':<4} {a[0] or '':<16}{a[1]}")
        out.append({"pos": pos, "page_path": r["page_path"], "layer_id": r["layer_id"],
                    "n_px_a_gt0": n_px, "layer_nonwhite": n_nw, "nonwhite_over_counted": round(ratio, 4),
                    "blocks_occupied": occ, "full_block_ratio": round(full / max(1, occ), 4),
                    "page_black_frac": round(blk, 4), "bbox_256_aligned": int(aligned),
                    "blocks_in_bbox": bb_blocks, "block_occupancy": round(occ_ratio, 4),
                    "answer_v1": a[0] or "", "comment_v1": a[1]})

    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print("\n札(数え上げ。規律9)")
    for k, v in flags.items():
        print(f"  {k}: {len(v)}行 {v}")
    authors_bug = [4, 8, 10, 11]
    caught = sorted(set(flags["頁が黒い"] + flags["描けていない見込み"]) & set(authors_bug))
    print("\n**作者の目との突き合わせ(規律13)** —— 作者が「バグっている」と言った行:", authors_bug)
    print(f"  札で拾えた {len(caught)}/{len(authors_bug)} {caught} / **取りこぼし** {sorted(set(authors_bug)-set(caught))}")
    print("  → **この札を作者の判定の代わりに使わない。** 壊れているかは画を見ないと言えない")
    import collections
    c = collections.Counter(v for v, _ in ans.values())
    print("\n作者が答えた11行(**壊れた画で取ったので棒には使わない。版として残す**)")
    for k, v in c.most_common():
        print(f"  {k}: {v}")
    print("  一言のあった行:", [i for i, (_, cm) in ans.items() if cm])
    print("\n出力", OUT)


main()
