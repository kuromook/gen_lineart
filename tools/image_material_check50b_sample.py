#!/usr/bin/env python3
"""(イ) 2つ目の50枚の二択を引く —— **見えている残りの340層**(登録 `image_material_50_image_check_20261114` の (イ))。

**この器具は画を1枚も見る前に書いた。選び方も棒も、ここで決めて登録する。**

母集団: `.clip` の取り込んだ画像の層(`has_image_material`)で、**層も祖先も表示**で、
第1弾の危険域に**入らなかった**層 ——
- **出力に画素が0**(`empty_c` = 1): 267
- **測れない**(`empty_c` が 0/1 でない): 73
計 **340層**。ここは画素の表が**古い切れ端**を読んでいた盲点で、**いま a が落としたままの所**である。

**引き方(先に決める)**
- **帯は使わない** —— 画素の数は古い切れ端の数なので、帯に使えない(census の指摘。E も同意)
- **2つの群に、母集団の割合で配る**: 出力に画素が0 267 → **39枚** / 測れない 73 → **11枚**
- 種 `20261115` で混ぜる(`sha256(seed|page|layer)` の順。再現できる)
- **1作品単位3層まで**(第1弾と同じ)
- **バイトが同じ写しの頁で、代表でない頁の層は除く**(census の提案を採る。第1弾では写しが2組入った)。
  **census が代表の印を持っているので、引いた50層のうち代表でない層を census が知らせ、
  その層は同じ群・同じ上限の中で、この並びの次の層に置き換える**(並びは決まっているので再現できる)

**棒(先に決める。第1弾と同じ形)** —— 「絵の一部」と答えた層の数で:
- **0〜2/50** → この340層は落とし続ける(**0/50 は「6%未満」で厳密なゼロではない**)
- **3〜9/50** → 理由『image_material(出力に画素が0)』と『(測れない)』を外して候補に戻す。
  **216,303 → 215,963 / 437,425 = 0.4937**(前後を両方、名前を付けて印字する)
- **10以上/50** → 外した上で、『見えない』4,356 の側も50枚で確かめる
- **「どれでもない」は「絵の一部」に数えない**(作者の 11-15 の言明で、第3の区分が在ると分かったため)。
  ただし**「どれでもない」と「分からない」の合計が10以上なら、この群は判定保留**
- **過去の数は撤回しない** —— 取り消すのは規則で、数は版として残る

**置き換え(2026-11-15)**: census が代表でないと知らせた3層(元の `pos` 41・42・44、どれも `測れない` の群)を
並びから外した。**同じ群・同じ上限の中で、並びの次の層が入る**。ほかの47層は動かない。

**印字するもの**: 落としたものを理由別に / 群ごとの母集団と枚数 / 上限で飛ばした層 / 作品単位の数 / 出力の sha256。
"""
import csv
import hashlib
import io
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "data/role_predictor/full_clip_pixels_layers_merged.csv"
OUT = HERE / "data/role_predictor/image_material_check50b_20261115.csv"
SEED = "20261115"
N = 50
PER_UNIT = 3
GROUPS = (("出力に画素が0", 39), ("測れない", 11))
# **登録どおりの置き換え**(2026-11-15)。census が「バイトが同じ写しの頁の、代表でない側」と知らせた層を
# 並びから外すと、同じ群・同じ上限の中で**並びの次の層**が入る。並びは種で決まっているので再現できる。
EXCLUDE = {
    ("F:\\manuscripts\\2024b\\hamlabi\\works\\page0013.clip", "84"),   # 元 pos 41
    ("F:\\manuscripts\\2024b\\hamlabi\\works\\page0001.clip", "67"),   # 元 pos 42
    ("F:\\manuscripts\\2024b\\hamlabi\\works\\page0013.clip", "82"),   # 元 pos 44
}


def unit(page_path):
    parts = page_path.replace("/", "\\").split("\\")
    return "\\".join(parts[:4]) if len(parts) > 4 else page_path


def main():
    drops = Counter()
    pool = defaultdict(list)
    for r in csv.DictReader(io.open(SRC, encoding="utf-8")):
        if r.get("has_image_material") not in ("1", "True", "true"):
            drops["画像素材でない"] += 1
            continue
        if not (r.get("self_shown") == "1" and r.get("ancestors_shown") == "1"):
            drops["見えない"] += 1
            continue
        e = r.get("empty_c")
        if e == "0":
            drops["出力に画素あり(第1弾で引いた群)"] += 1
            continue
        g = "出力に画素が0" if e == "1" else "測れない"
        pool[g].append(r)
    print("落としたもの(規律9)")
    for k, v in drops.most_common():
        print(f"   {k}: {v}")
    print("母集団", {k: len(v) for k, v in pool.items()}, "計", sum(len(v) for v in pool.values()))

    picked, skipped = [], Counter()
    for g, quota in GROUPS:
        rows = sorted(pool[g], key=lambda r: hashlib.sha256(
            f"{SEED}|{r['page_path']}|{r['layer_id']}".encode()).hexdigest())
        cnt = Counter()
        take = []
        for r in rows:
            if len(take) >= quota:
                break
            if (r["page_path"], r["layer_id"]) in EXCLUDE:
                skipped["代表でない頁の層(census)"] += 1
                continue
            u = unit(r["page_path"])
            if cnt[u] >= PER_UNIT:
                skipped[g] += 1
                continue
            cnt[u] += 1
            take.append((g, r))
        print(f"   {g}: 母集団 {len(rows)} → {len(take)}枚(上限で飛ばした層 {skipped[g]})")
        picked += take
    assert len(picked) == N, len(picked)

    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos", "group", "page_path", "layer_id", "empty_c", "pixel_formats",
                    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "render_status"])
        for i, (g, r) in enumerate(picked, 1):
            w.writerow([i, g, r["page_path"], r["layer_id"], r.get("empty_c", ""),
                        r.get("pixel_formats", ""), r["bbox_x0"], r["bbox_y0"],
                        r["bbox_x1"], r["bbox_y1"], r.get("render_status", "")])
    print("作品単位", len(set(unit(r['page_path']) for _, r in picked)),
          "/ 異なる頁", len(set(r['page_path'] for _, r in picked)))
    print("bbox が空欄の層", sum(1 for _, r in picked if not r["bbox_x0"]),
          "(**枠は描き直した層の画素から引く** —— 一覧の bbox は古い切れ端のもの)")
    print("出力", OUT, "sha256 先頭", hashlib.sha256(OUT.read_bytes()).hexdigest()[:16])


main()
