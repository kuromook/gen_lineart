#!/usr/bin/env python3
"""(ウ) 『資料』の名前で落とす 3,557層から50枚を引く —— **名前の規則のバリデーション**。

**この器具は画を1枚も見る前に書いた。引き方も棒も、11-16 の登録のまま**
(`hidden_reason_redecided_and_shiryou_name_clause_20261116.validation_hearing_registered_before_images`)。
**ユーザーの承認**: 「1 OK 2 ok」(2026-11-16)。作者の提案(逐語)
「3000のうち、これまでの決定論的除外で培ったものでどれくらいやれるか、バリデーション的に試す機会に
してもいいのかもしれない」。

母集団: `.clip`・**非表示**(層か祖先が非表示)・取り込んだ画像の層で、**名前か祖先が『資料』**の層。
census の鍵 `image_material_hidden_groups_key_20261116.csv` の `name_group` 1(`資料化下絵` 525)と
3(名前か祖先に資料 3,032)= **3,557層**。

**引き方(登録のまま)**
- **群に母集団の割合で配る: `資料化下絵` 7 / 名前か祖先に資料 43**
- 種 `20261117` で混ぜる(`sha256(seed|page|layer)` の順。再現できる)
- **1作品単位3層まで**((イ) と同じ `split_unit`)
- **年の札は層別しない**(母集団が 2017 に偏っているのは実態)が、**引いた50層の札の内訳を印字する**
- 代表でない頁の層は census が知らせ、**同じ群・同じ上限の中で並びの次の層に置き換える**((イ) と同じ)

**棒(登録のまま)** —— **「資料」でない答え**(絵の一部 + 下絵 + 3D背景 + 切り出し・コラージュ)の数で:
- **0〜2/50** → 名前の規則を据え置く(**0/50 は「6%未満」で厳密なゼロではない**)
- **3以上/50** → 名前の規則を撤回し、3,557 を候補に戻す(**215,162 → 215,961 相当**。前後を両方印字)
- **「どれでもない」+「分からない」が10以上なら判定保留**
- 群ごとの内訳は並べて印字するが**決め手にしない**(`資料化下絵` は7層しかない)

**この便で先に言っておく限界**: 85層(名前が `下絵 N` などで、**祖母に『資料』**を含む層)は
**母集団に入っていない**(census の一群の順で『下絵』が先に当たる)。**作者はこの形について裁定していない** ——
裁定が出て落とす側に移るなら、**この50枚はその85層を覆っていない**と書く。

**印字するもの**: 母集団と落としたもの / 群ごと / 上限で飛ばした層 / 作品単位 / 年の札 / 出力の sha256。
"""
import csv
import hashlib
import io
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "inbox/image_material_hidden_groups_key_20261116.csv"
OUT = HERE / "data/role_predictor/shiryou_name_check50_20261117.csv"
SEED = "20261117"
PER_UNIT = 3
GROUPS = (("名前が資料化下絵", 7), ("名前か祖先に資料", 43))


def main():
    rows = list(csv.DictReader(io.open(SRC, encoding="utf-8")))
    print(f"鍵の行 {len(rows)}(census の 4,356層)")
    drop = Counter()
    pool = defaultdict(list)
    for r in rows:
        lab = r["name_group_label"]
        if lab not in dict(GROUPS):
            drop[f"母集団外: {lab}"] += 1
            continue
        pool[lab].append(r)
    print("**落としたもの(規律9)**: " + " / ".join(f"{k} {v}" for k, v in sorted(drop.items())))
    print("   計 " + str(sum(drop.values())) + " → 母集団 " + str(sum(len(v) for v in pool.values())))
    for lab, n in GROUPS:
        print(f"   群 {lab}: 母集団 {len(pool[lab])} → 引く {n}")
    assert sum(len(v) for v in pool.values()) == 3557, sum(len(v) for v in pool.values())

    picked, skipped = [], Counter()
    per_unit = Counter()
    for lab, want in GROUPS:
        cand = sorted(pool[lab], key=lambda r: hashlib.sha256(
            f"{SEED}|{r['page_path']}|{r['layer_id']}".encode()).hexdigest())
        got = 0
        for r in cand:
            if got >= want:
                break
            u = r["split_unit"]
            if per_unit[u] >= PER_UNIT:
                skipped[f"{lab}: 上限({PER_UNIT}層/作品単位)"] += 1
                continue
            per_unit[u] += 1
            picked.append(r)
            got += 1
        assert got == want, (lab, got, want)
    print("**上限で飛ばした層**: " + (" / ".join(f"{k} {v}" for k, v in sorted(skipped.items())) or "0"))

    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos", "group", "page_path", "layer_id", "name_group", "work_folder", "year_label", "hidden_by"])
        for i, r in enumerate(picked, 1):
            w.writerow([i, r["name_group_label"], r["page_path"], r["layer_id"],
                        r["name_group"], r["work_folder"], r["year_label"], r["hidden_by"]])
    print(f"\n出力 {OUT}")
    print("   sha256 先頭", hashlib.sha256(OUT.read_bytes()).hexdigest()[:16])
    print("   50層 / 作品単位", len(set(r["split_unit"] for r in picked)),
          "/ 異なる頁", len(set(r["page_path"] for r in picked)))
    print("   群ごと", dict(Counter(r["name_group_label"] for r in picked)))
    print("   年の札", dict(Counter(r["year_label"] for r in picked)),
          "(母集団は", dict(Counter(r["year_label"] for v in pool.values() for r in v)), ")")
    print("   非表示の形", dict(Counter(r["hidden_by"] for r in picked)))
    print("\n**代表でない頁の層が在れば census が知らせ、同じ群・同じ上限の中で並びの次の層に置き換える**")


main()
