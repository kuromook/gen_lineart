#!/usr/bin/env python
"""**R4: `name_rule` が「なし」の層の膜/膜でないの数え上げと、30層の標本**(Track R の発注、2026-11-20)。

登録は R の木 `config/prereg_r4_film_rule_nashi_20261120.md`(commit 80f2bd0、標本を引く前)。census は引くだけで点数を見ない。

- 母集団: `.clip` の層のうち、全頁の画素のまとめた表に行があり `n_px_a_gt0 > 0` の層。名前は構造 DB の `layers`
- `name_rule` は役割の表 v3 と同じ関数(`stage_hearing.name_rule`、名前だけの関数)
- **膜** = `n_px_a_ge26 / n_px_a_gt0 < 0.5`(登録のまま)
- 門(理由別に数える。上から順に当て、最初に当たった理由で数える):
  1. 恒久除外の頁 2. E の a v7 `template_excluded` 3. `empty_c`=1
  4. R2 の学習の27作品フォルダー(`work_dir` が一致)
  5. 同じ作品: 27作品フォルダーと同じ `split_unit`(編の裁定と中身の共有で束ねた単位)に入る作品フォルダー
     ——「`2017\\copy` の答えで同じ作品とされた作品」は作者の答え「別の作品」(11-18)で 0。
     census の記録で同じ作品に束ねてあるものは同じ趣旨で外す(数を別の理由で印字)
- 標本: 膜の組 20 + 対照の組 10。1作品フォルダー(`work_dir`)から各組3層まで。`random.Random(0)`
  で各組の候補(並べた後)を混ぜ、上から上限を守って取る。混ぜた順も同じ乱数の続き

出力は `data/census_full/`(頁のパスを持つので公開しない・追跡しない)。
"""
import collections
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402
from stage_hearing import name_rule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
PIX = os.path.join(D, "full_clip_pixels_layers_merged.csv")
AV7 = os.path.join(D, "a_verdict_v7_20261116.csv")
OUT_POP = os.path.join(D, "r4_film_nashi_population_20261120.csv")
OUT_TAB = os.path.join(D, "r4_film_nashi_hearing_table_20261120.csv")
OUT_KEY = os.path.join(D, "r4_film_nashi_hearing_key_20261120.csv")
TRAIN27 = r"""2014\ako4-6DL, 2016\avbu, 2016\pc, 2016\suc, 2017\copy, 2017\running, 2017\toho, 2018\color_comics4,
2019\reitaisai2, 2019\skima2, 2019\succor, 2019b\akogareFANBOX, 2019b\marisa, 2020\ichigo, 2021\akoirai2,
2021\bakugi4, 2021\boxing, 2022\bakugi6, 2022b\4th, 2023\ap, 2023b\hamlabi, 2024\housei, 2024\kana,
2024b\fitness, 2024b\twobody, 2024c\mayer, 2024c\q2"""
N_FILM, N_CTRL, CAP = 20, 10, 3


def main() -> int:
    train = {"F:\\manuscripts\\" + s.strip() for s in TRAIN27.replace("\n", " ").split(",")}
    assert len(train) == 27
    c = db.connect()
    ex = exclusions.load()
    un = split_units.units(c)
    miss = sorted(t for t in train if t not in un)
    if miss:
        print("27作品フォルダーのうち DB に無い:", miss)
        return 1
    train_units = {un[t] for t in train}
    names = {(p, i): (n or "") for p, i, n in c.execute(
        "select l.page_path, l.layer_id, l.name from layers l join pages p using (page_path) where p.format='clip'").fetchall()}
    av7 = {(r["page_path"], int(r["layer_id"])): r["a_verdict_v7"]
           for r in csv.DictReader(open(AV7, encoding="utf-8"))}

    drop = collections.Counter()
    pop = []
    n_pix = 0
    for r in csv.DictReader(open(PIX, encoding="utf-8")):
        n_pix += 1
        k = (r["page_path"], int(r["layer_id"]))
        n = names.get(k)
        if n is None:
            drop["0 DB に層が無い"] += 1
            continue
        if name_rule(n) != "なし":
            drop["0 name_rule がなしでない(母集団の外)"] += 1
            continue
        gt0 = int(r["n_px_a_gt0"] or 0)
        if gt0 <= 0:
            drop["0 n_px_a_gt0 が 0・空(膜を定義できない)"] += 1
            continue
        p = k[0]
        w = wd(p)
        if p in ex:
            drop["1 恒久除外の頁"] += 1; continue
        if av7.get(k) == "template_excluded":
            drop["2 a v7 template_excluded"] += 1; continue
        if r["empty_c"] == "1":
            drop["3 empty_c=1"] += 1; continue
        if w in train:
            drop["4 R2 の学習の27作品フォルダー"] += 1; continue
        if un.get(w) in train_units:
            drop["5 27作品フォルダーと同じ split_unit(同じ作品に束ねた)"] += 1; continue
        film = int(r["n_px_a_ge26"] or 0) / gt0 < 0.5
        pop.append(dict(page_path=p, layer_id=k[1], work_dir=w, split_unit=un.get(w, ""),
                        year_label=p.split("\\")[2], layer_name=n,
                        self_shown=r["self_shown"], ancestors_shown=r["ancestors_shown"],
                        n_px_a_gt0=gt0, n_px_a_ge26=r["n_px_a_ge26"], film=int(film)))
    if av7 and len(av7) < 400000:
        print("a v7 の行が少ない"); return 1

    print(f"全頁の画素のまとめた表: {n_pix:,} 層")
    print("落とした(理由別、上から順に最初に当たった理由。0 は母集団の定義、1〜5 は門):")
    for k2, v in sorted(drop.items()):
        print(f"  {v:8,d}  {k2}")
    fc = collections.Counter(x["film"] for x in pop)
    print(f"\n門の後: name_rule なし {len(pop):,} 層 = 膜 {fc[1]:,} / 膜でない {fc[0]:,}")
    print("\n年代札ごと(膜 / 膜でない):")
    by_y = collections.defaultdict(collections.Counter)
    for x in pop:
        by_y[x["year_label"]][x["film"]] += 1
    for y in sorted(by_y):
        print(f"  {y:8s} {by_y[y][1]:7,d} / {by_y[y][0]:7,d}")
    by_w = collections.defaultdict(collections.Counter)
    for x in pop:
        by_w[x["work_dir"]][x["film"]] += 1
    nf = sum(1 for w in by_w if by_w[w][1])
    print(f"\n作品フォルダー: {len(by_w)}(膜の層を持つもの {nf})。作品フォルダーごとの表は出力の CSV で")

    with open(OUT_POP, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(pop[0])); wr.writeheader(); wr.writerows(pop)

    rng = random.Random(0)
    picks = []
    for grp, want in ((1, N_FILM), (0, N_CTRL)):
        cand = sorted((x for x in pop if x["film"] == grp), key=lambda x: (x["page_path"], x["layer_id"]))
        rng.shuffle(cand)
        per = collections.Counter()
        got = []
        for x in cand:
            if len(got) >= want:
                break
            if per[x["work_dir"]] >= CAP:
                continue
            per[x["work_dir"]] += 1
            got.append(x)
        print(f"\n組 {'膜' if grp else '対照(膜でない)'}: 候補 {len(cand):,} → 標本 {len(got)}(作品フォルダー {len(per)})")
        picks += [dict(x, group="film" if grp else "control") for x in got]
    rng.shuffle(picks)

    tab_cols = ["row", "page_path", "layer_id", "layer_name", "work_dir", "year_label",
                "self_shown", "ancestors_shown", "answer", "comment"]
    with open(OUT_TAB, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=tab_cols); wr.writeheader()
        for i, x in enumerate(picks, 1):
            wr.writerow({**{k3: x[k3] for k3 in tab_cols if k3 in x}, "row": i, "answer": "", "comment": ""})
    with open(OUT_KEY, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["row", "page_path", "layer_id", "group", "n_px_a_gt0", "n_px_a_ge26"])
        wr.writeheader()
        for i, x in enumerate(picks, 1):
            wr.writerow(dict(row=i, page_path=x["page_path"], layer_id=x["layer_id"], group=x["group"],
                             n_px_a_gt0=x["n_px_a_gt0"], n_px_a_ge26=x["n_px_a_ge26"]))
    print("\n標本の内訳: 年代札 × 組")
    for (y, g), v in sorted(collections.Counter((x["year_label"], x["group"]) for x in picks).items()):
        print(f"  {y:8s} {g:8s} {v}")
    hid = collections.Counter((x["group"], x["self_shown"] == "1" and x["ancestors_shown"] == "1") for x in picks)
    print("表示(自分も祖先も表示): " + " / ".join(f"{g} {'表示' if s else '非表示'} {v}" for (g, s), v in sorted(hid.items())))
    print(f"\n出力: {OUT_POP}\n      {OUT_TAB}\n      {OUT_KEY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
