#!/usr/bin/env python
"""**R2 の評価の便R2a: 試しの20層**(Track R の依頼 `note_census_order_r2a_stage_sample_20261128.md`)。

登録は R の木 `config/prereg_r2_name_ngram_3class_20261117.md` 追補1(`layer-picture` の `ba85544`、引く前)。
根拠: ユーザー 11-28「RはR2に通達を送り、あらためて聞き取りをたてる」(基盤の通達)。census は引くだけで点数を見ない。
**段の推測は付けない。** 画の発注はしない(ユーザーの承認の後に別の便)。

- 母集団: `.clip` の層のうち、全頁の画素のまとめた表に行がある層。名前は構造 DB の `layers`
- `name_rule` は役割の表 v3 と同じ関数(`stage_hearing.name_rule`)
- 門(上から順に当て、最初に当たった理由で数える。R4 の器具と同じ並び):
  1. 恒久除外の頁 2. **a v12 `template_excluded`(主)** 3. `empty_c`=1
  4. R2 の学習の27作品フォルダー(`work_dir` が一致)5. 27作品フォルダーと同じ `split_unit`
  —— 並べて、a v7 `template_excluded` で外れる数(門 2 を v7 に替えた場合)も印字する
- 除く: 便A の主81行・R1b の学習244行(R の器具 `r1_stage_predictor` から鍵だけ書き出した表)・R4 の30層。
  **門の後に残る重なりの数を印字**(0 のはず)
- 層: `name_rule` = 清書 8 / 下絵 6 / なし 6。「両方当たる」はどの層にも入れず数を印字。
  1作品フォルダーから各層 2層まで、`random.Random(0)` で層ごとに候補を混ぜて上から取る。
  取った後で、同じ乱数の続きで3層を混ぜる。非表示の層・資料のフォルダーの下の層も入れる

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
from r4_film_nashi_sample import TRAIN27  # noqa: E402
from stage_hearing import name_rule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
PIX = os.path.join(D, "full_clip_pixels_layers_merged.csv")
AV12 = os.path.join(D, "a_verdict_v12_20261126.csv")
AV7 = os.path.join(D, "a_verdict_v7_20261116.csv")
TRAIN325 = os.path.join(D, "r2_train325_keys_20261128.csv")
R4KEY = os.path.join(D, "r4_film_nashi_hearing_key_20261120.csv")
OUT_POP = os.path.join(D, "r2a_stage_population_20261128.csv")
OUT_TAB = os.path.join(D, "r2a_stage_hearing_table_20261128.csv")
OUT_KEY = os.path.join(D, "r2a_stage_hearing_key_20261128.csv")
WANT = (("清書", 8), ("下絵", 6), ("なし", 6))
CAP = 2


def main() -> int:
    print("[R2 の評価の便R2a: 試しの20層(Track R の依頼)]")
    train = {"F:\\manuscripts\\" + s.strip() for s in TRAIN27.replace("\n", " ").split(",")}
    assert len(train) == 27
    c = db.connect()
    ex = exclusions.load()
    un = split_units.units(c)
    train_units = {un[t] for t in train}
    names = {(p, i): (n or "") for p, i, n in c.execute(
        "select l.page_path, l.layer_id, l.name from layers l join pages p using (page_path) where p.format='clip'").fetchall()}
    av12 = {(r["page_path"], int(r["layer_id"])): r["a_verdict_v12"] for r in csv.DictReader(open(AV12, encoding="utf-8"))}
    av7 = {(r["page_path"], int(r["layer_id"])): r["a_verdict_v7"] for r in csv.DictReader(open(AV7, encoding="utf-8"))}
    print(f"    a v12 {len(av12):,} 行 / a v7 {len(av7):,} 行")
    known = collections.defaultdict(set)
    for r in csv.DictReader(open(TRAIN325, encoding="utf-8")):
        known[r["src"]].add((r["page_path"], int(r["layer_id"])))
    for r in csv.DictReader(open(R4KEY, encoding="utf-8")):
        known["r4_30"].add((r["page_path"], int(r["layer_id"])))
    print("    除く鍵: " + " / ".join(f"{k} {len(v)}" for k, v in sorted(known.items())))
    allknown = set().union(*known.values())

    drop = collections.Counter()
    drop_known = collections.Counter()
    v7_would = collections.Counter()
    pop = []
    n_pix = 0
    for r in csv.DictReader(open(PIX, encoding="utf-8")):
        n_pix += 1
        k = (r["page_path"], int(r["layer_id"]))
        n = names.get(k)
        if n is None:
            drop["0 DB に層が無い"] += 1
            continue
        p = k[0]
        w = wd(p)
        reason = None
        if p in ex:
            reason = "1 恒久除外の頁"
        elif av12.get(k) == "template_excluded":
            reason = "2 a v12 template_excluded"
        elif r["empty_c"] == "1":
            reason = "3 empty_c=1"
        elif w in train:
            reason = "4 R2 の学習の27作品フォルダー"
        elif un.get(w) in train_units:
            reason = "5 27作品フォルダーと同じ split_unit"
        # 並べて: 門 2 を a v7 に替えたら
        if p not in ex and av7.get(k) == "template_excluded":
            v7_would["a v7 template_excluded に当たる層(恒久除外の後)"] += 1
            if reason is None:
                v7_would["そのうち v12 の門の後に残っている層(v7 なら外れる)"] += 1
        if reason is not None:
            drop[reason] += 1
            if k in allknown:
                drop_known[reason] += 1
            continue
        nr = name_rule(n)
        pop.append(dict(page_path=p, layer_id=k[1], work_dir=w, split_unit=un.get(w, ""),
                        year_label=p.split("\\")[2], layer_name=n, name_rule=nr,
                        self_shown=r["self_shown"], ancestors_shown=r["ancestors_shown"]))

    print(f"    全頁の画素のまとめた表: {n_pix:,} 層")
    print("    落とした(理由別、上から順に最初に当たった理由。0 は母集団の定義、1〜5 は門):")
    for k2, v in sorted(drop.items()):
        print(f"      {v:8,d}  {k2}")
    print("    並べて(a v7 の門):")
    for k2, v in v7_would.items():
        print(f"      {v:8,d}  {k2}")
    print("    除く鍵が門で外れた数(理由別): " + (", ".join(f"{k} {v}" for k, v in sorted(drop_known.items())) or "0"))
    left = [x for x in pop if (x["page_path"], x["layer_id"]) in allknown]
    print(f"    **門の後に残った除く鍵の層: {len(left)}**(0 のはず)")
    pop = [x for x in pop if (x["page_path"], x["layer_id"]) not in allknown]

    by_nr = collections.defaultdict(list)
    for x in pop:
        by_nr[x["name_rule"]].append(x)
    print(f"\n    門の後: {len(pop):,} 層 / 作品フォルダー {len({x['work_dir'] for x in pop})}")
    for nr in ("清書", "下絵", "なし", "両方当たる"):
        xs = by_nr.get(nr, [])
        print(f"      name_rule {nr:6s} 層 {len(xs):8,d} / 作品フォルダー {len({x['work_dir'] for x in xs}):4d}"
              + ("(どの層にも入れない)" if nr == "両方当たる" else ""))
    print("    年代札ごと(清書 / 下絵 / なし):")
    by_y = collections.defaultdict(collections.Counter)
    for x in pop:
        by_y[x["year_label"]][x["name_rule"]] += 1
    for y in sorted(by_y):
        print(f"      {y:8s} {by_y[y]['清書']:7,d} / {by_y[y]['下絵']:7,d} / {by_y[y]['なし']:7,d}")

    with open(OUT_POP, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(pop[0])); wr.writeheader(); wr.writerows(pop)

    rng = random.Random(0)
    picks = []
    for nr, want in WANT:
        cand = sorted(by_nr[nr], key=lambda x: (x["page_path"], x["layer_id"]))
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
        print(f"\n    層 {nr}: 候補 {len(cand):,} → 標本 {len(got)}/{want}(作品フォルダー {len(per)})")
        picks += got
    rng.shuffle(picks)

    tab_cols = ["row", "page_path", "layer_id", "layer_name", "work_dir", "year_label",
                "self_shown", "ancestors_shown", "answer", "comment"]
    with open(OUT_TAB, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=tab_cols); wr.writeheader()
        for i, x in enumerate(picks, 1):
            wr.writerow({**{k3: x[k3] for k3 in tab_cols if k3 in x}, "row": i, "answer": "", "comment": ""})
    with open(OUT_KEY, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["row", "page_path", "layer_id", "name_rule"])
        wr.writeheader()
        for i, x in enumerate(picks, 1):
            wr.writerow(dict(row=i, page_path=x["page_path"], layer_id=x["layer_id"], name_rule=x["name_rule"]))
    print("\n    標本の内訳: 年代札(層は鍵の表にだけ書く。R は答えがそろうまで開かない)")
    for y, v in sorted(collections.Counter(x["year_label"] for x in picks).items()):
        print(f"      {y:8s} {v}")
    hid = sum(not (x["self_shown"] == "1" and x["ancestors_shown"] == "1") for x in picks)
    print(f"    非表示(自分か祖先が非表示): {hid} / {len(picks)}")
    print(f"    作品フォルダー: {len({x['work_dir'] for x in picks})}")
    print(f"\n    出力: {OUT_POP}\n          {OUT_TAB}\n          {OUT_KEY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
