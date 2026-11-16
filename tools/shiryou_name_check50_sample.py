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

**置き換えは巡を重ねる(2026-11-17)**: 代表の印は census が持っているので、**入った層がまた代表でない**ことが起きる。
そのときも当てる決まりは同じ一手(『同じ群・同じ上限の中で、並びの次の層』)で、**巡を重ねるのは
その一手を不動点まで繰り返すことと同じ**。版は `v2`(第1巡)・`v3`(第2巡)…と残し、**前の版は撤回しない**。

**置き換え(2026-11-16。登録のまま)**: census が「代表でない頁の層」と知らせた4層
(元の `pos` 1・14・19・33。1 は `資料化下絵` の群、ほかは `名前か祖先に資料` の群。どれも束の大きさ2 で、
代表の頁の同じ番号の層は一覧に無い)を並びから外した。**同じ群・同じ上限の中で、並びの次の層が入る**
(並びは種で決まっているので再現できる)。**棒と母集団(3,557)は動かない** —— 配達の事情で、点数と独立(規律9-5)。
出力は `shiryou_name_check50_v2_20261117.csv`(前の一覧 `41b6ad4560930267` は残す)。

**印字するもの**: 母集団と落としたもの / 群ごと / 上限で飛ばした層 / 作品単位 / 年の札 / 出力の sha256 /
**置き換えで入った層と、動かなかった46層**。
"""
import csv
import hashlib
import io
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "inbox/image_material_hidden_groups_key_20261116.csv"
OUT = HERE / "data/role_predictor/shiryou_name_check50_v3_20261117.csv"
PREV = HERE / "data/role_predictor/shiryou_name_check50_v2_20261117.csv"
# **登録どおりの置き換え**(2026-11-16)。census が代表でないと知らせた4層を並びから外すと、
# 同じ群・同じ上限の中で**並びの次の層**が入る。上限は食わせない(外した層は数えない)。
EXCLUDE = {
    ("F:\\manuscripts\\2018\\comics4\\mars_v2\\mars_manuscript\\page0005.clip", "47"),   # 元 pos 1
    ("F:\\manuscripts\\2018\\comics4\\mars_v2\\mars_manuscript\\page0020.clip", "147"),  # 元 pos 14
    ("F:\\manuscripts\\2018\\comics4\\mars_v2\\mars_manuscript\\page0003.clip", "258"),  # 元 pos 19
    ("F:\\manuscripts\\2019b\\skima2\\task\\page0007.clip", "68"),                         # 元 pos 33
    # **第2巡(2026-11-17。census `reply_e_census_check50c_v2_one_more_nonrep_20261117.md`)**:
    # v2 の置き換えで新しく入った層が、また代表でない頁だった(v1 の4層とは別の束。代表は 2017 の札の側)。
    # 同じ決まりをもう一度当てる。**巡を重ねても、やっているのは『並びの次の層』の一手だけ**。
    ("F:\\manuscripts\\2018\\comics4\\mars_v2\\mars_manuscript\\page0008.clip", "74"),    # v2 pos 20
}
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
            if (r["page_path"], r["layer_id"]) in EXCLUDE:
                skipped[f"{lab}: 代表でない頁の層(census 11-16)"] += 1
                continue
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
    prev = {(r["page_path"], r["layer_id"]): r["pos"]
            for r in csv.DictReader(io.open(PREV, encoding="utf-8"))}
    now = {(r["page_path"], r["layer_id"]) for r in picked}
    kept = sorted(prev[k] for k in now & set(prev))
    came_in = [r for r in picked if (r["page_path"], r["layer_id"]) not in prev]
    went_out = sorted((v, k) for k, v in prev.items() if k not in now)
    print(f"\n**置き換え**: 動かなかった層 {len(kept)} / 外れた層 {len(went_out)} / 入った層 {len(came_in)}")
    for pos, (pg, lid) in went_out:
        print(f"   外: 元 pos {pos} {pg} 層 {lid}")
    for r in came_in:
        pos = [i for i, x in enumerate(picked, 1) if x is r][0]
        print(f"   入: pos {pos} [{r['name_group_label']}] {r['page_path']} 層 {r['layer_id']}"
              f" / 作品単位 {r['split_unit']} / 年 {r['year_label']} / 非表示 {r['hidden_by']}")
    # **実測は 5 入れ替わりで、census が知らせた4層より1つ多い。** 上限(3層/作品単位)の数えは
    # **群をまたいで共有**しているので、群1 で1層足すと、その作品単位の枠を食い、群2 で枠に収まって
    # いた層(元 pos 20 `2017\\ol`)が上限で外れる。**器具の決まりどおりの連鎖で、点数とは独立**(規律9-5)。
    print("   **前の版との差を印字した。** 上限(3層/作品単位)の数えは群をまたいで共有しているので、"
          "1層外すと別の群の行が連鎖で動きうる(v1 → v2 では census の4層 + 連鎖1層 = 5層)")
    print("\n**代表でない頁の層が在れば census が知らせ、同じ群・同じ上限の中で並びの次の層に置き換える**")


main()
