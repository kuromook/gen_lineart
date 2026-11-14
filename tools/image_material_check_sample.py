# -*- coding: utf-8 -*-
"""**`image_material`(出力に画素あり)821層から、作者に見せる50枚を選ぶ。**

作者の「y」(2026-11-14)で走る。**選び方を先に書く(規律9: 選別は測定である)**:

  母集団 = `.clip` の `image_material` で **出力に画素あり**の層 821(`self_shown`=1 かつ
           `ancestors_shown`=1 かつ `empty_c`=0)。**ここだけが絵を落とす危険のある所**。
  層  = 画素の数で3つの帯に切る(中央値が 262,144 = 512×512 だったので、そこを境に取る)
        帯1 < 65,536 / 帯2 65,536〜262,144 / 帯3 > 262,144
  配り方 = 帯ごとに、母集団の割合に比例して配る(合計50)
  並べ方 = 種(seed=20261114)で混ぜる。**1作品単位から3層までに絞る**(1作品に偏らせない)
  落としたものは理由別に数えて印字する。

**棒(先に決める。規律10)** —— 作者の答えが「絵の一部」だった層の数で決める:
  0〜2 / 50(≤4%)  → 規則を据え置く。**0/50 は「6%未満」で厳密なゼロではない**(条例5 と同じ形)
  3〜9 / 50        → 理由「image_material(出力に画素あり)」だけを外し、候補に戻す
  10以上 / 50(≥20%)→ 理由を外し、さらに「見えない」「出力に画素が0」の側も50枚で確かめる
  **分からない**が多い(10以上)ときは **判定保留**。棒には使わない(正直な結論である)

**問い**(条例2: 何について答えるのかを画面に書く)
  「この層は、この頁の**絵の一部**ですか。それとも**資料**(下絵・参考・見ながら描くための画像)ですか」
  選択肢: 絵の一部 / 資料 / 分からない + コメント欄。
  画面には**層だけの絵**と**仕上げの頁(この層の場所に印)**を並べる。
  **項目の位置と直前の項目を記録する**(判定は文脈で動きうる)。
"""
import csv, collections, hashlib, io, random, sys

MERGED = "data/role_predictor/full_clip_pixels_layers_merged.csv"
OUT = "data/role_predictor/image_material_check50_20261114.csv"
SEED = 20261114
N = 50
PER_UNIT = 3
BANDS = ((0, 65536, "帯1 <65,536"), (65536, 262144, "帯2 65,536〜262,144"),
         (262144, 1 << 60, "帯3 >262,144"))


def sha(p):
    h = hashlib.sha256()
    with io.open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def unit(p):
    # F:\manuscripts\<年代札>\<作品>\... を作品単位の代わりに使う(census の split_unit は手元に無い)
    q = p.replace("/", "\\").split("\\")
    return "\\".join(q[2:5]) if len(q) > 5 else p


def band(n):
    for lo, hi, name in BANDS:
        if lo <= n < hi or (lo < n <= hi and lo == 262144):
            return name
    return BANDS[-1][2]


def main():
    print("表: %s sha256 %s" % (MERGED, sha(MERGED)))
    drop = collections.Counter()
    pop = []
    for r in csv.DictReader(io.open(MERGED, encoding="utf-8")):
        if r["has_image_material"] not in ("1", "True", "true"):
            drop["image_material でない"] += 1; continue
        if not (r["self_shown"] == "1" and r["ancestors_shown"] == "1"):
            drop["見えない(資料の見込みが高い側)"] += 1; continue
        if r["empty_c"] == "1":
            drop["出力に画素が0(範囲外で放置の形)"] += 1; continue
        if r["empty_c"] != "0":
            drop["測れない(描画できなかった)"] += 1; continue
        pop.append(r)
    print("\n== 落としたもの(規律9: 理由別) ==")
    for k, v in drop.most_common():
        print("   %-34s %6d" % (k, v))
    print("   ---- **母集団: %d 層**(出力に画素ありの image_material)" % len(pop))

    bc = collections.Counter(band(int(r["n_px_a_gt0"])) for r in pop)
    print("\n== 帯ごとの母集団と配り方(比例) ==")
    quota = {}
    for _, _, name in BANDS:
        quota[name] = int(round(N * bc[name] / len(pop)))
    # 丸めで 50 に合わせる(多い帯から足し引きする。**先に書いた規則**)
    while sum(quota.values()) != N:
        k = max(bc, key=lambda x: bc[x]) if sum(quota.values()) < N else max(quota, key=lambda x: quota[x])
        quota[k] += 1 if sum(quota.values()) < N else -1
    for _, _, name in BANDS:
        print("   %-24s 母集団 %4d(%.3f)→ **%d 枚**" % (name, bc[name], bc[name] / len(pop), quota[name]))

    rnd = random.Random(SEED)
    chosen, per = [], collections.Counter()
    skipped_unit = 0
    for _, _, name in BANDS:
        sub = [r for r in pop if band(int(r["n_px_a_gt0"])) == name]
        rnd.shuffle(sub)
        for r in sub:
            if len([c for c in chosen if band(int(c["n_px_a_gt0"])) == name]) >= quota[name]:
                break
            u = unit(r["page_path"])
            if per[u] >= PER_UNIT:
                skipped_unit += 1; continue
            per[u] += 1; chosen.append(r)
    print("\n   1作品単位3層までの制限で飛ばした層: %d" % skipped_unit)
    print("   **選んだ: %d 枚 / 作品単位 %d**" % (len(chosen), len(per)))

    rnd.shuffle(chosen)
    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos", "page_path", "layer_id", "band", "n_px_a_gt0",
                    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "pixel_formats"])
        for i, r in enumerate(chosen, 1):
            w.writerow([i, r["page_path"], r["layer_id"], band(int(r["n_px_a_gt0"])),
                        r["n_px_a_gt0"], r["bbox_x0"], r["bbox_y0"], r["bbox_x1"],
                        r["bbox_y1"], r["pixel_formats"]])
    print("   書いた: %s(%d 行 + 見出し)sha256 %s" % (OUT, len(chosen), sha(OUT)))
    print("   **`data/` の外に出さない。** 画像は census が抽出側に発注する")
    print("\n   帯の内訳(選んだ分): " + " / ".join(
        "%s %d" % (k, v) for k, v in collections.Counter(band(int(r["n_px_a_gt0"])) for r in chosen).items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
