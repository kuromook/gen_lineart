#!/usr/bin/env python
"""**便R2a のフォルダーの層の引き直し**(Track R の依頼 `note_census_order_r2a_replace_folders_20261128.md`)。

登録は R の木 `config/prereg_r2_name_ngram_3class_20261117.md` 追補3(`layer-picture` の `92d4a44`、答えの前・鍵は開いていない)。
**census は引くだけ。** 段の推測は付けない。画の発注はしない(ユーザーの承認の後に R から)。

- 門を1つ足す: 構造 DB `layers.folder` が真の層を外す。**前後の母集団の数を名前を付けて印字する**(規律9・曝露も棒の一部)
- 標本: 便R2a の20層のうちフォルダーでない層は残す。フォルダーの層は、**同じ `name_rule` の中で**、
  フォルダーでない層から `random.Random(0)` で引き直す(候補は (頁, 層) で並べてから混ぜ、上から取る)。
  既に引いた20層は除く。1作品フォルダーから各層 2層までの上限を、残す層と合わせて守る。
  層の順は R2a と同じ(清書 → 下絵 → なし)。取った後で、同じ乱数の続きで20行を混ぜる
- ついでに数える(記録。棒は動かさない): R4 の30層と、`name_rule_film` v1 の「あたり+下絵(膜)」の層の `folder` 真の数

入力: `r2a_stage_sample.py` の出力(母集団・鍵)。出力は `data/census_full/`(非公開・追跡しない)。
"""
import collections
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
POP = os.path.join(D, "r2a_stage_population_20261128.csv")
KEY0 = os.path.join(D, "r2a_stage_hearing_key_20261128.csv")
R4KEY = os.path.join(D, "r4_film_nashi_hearing_key_20261120.csv")
FILM = os.path.join(D, "layer_name_rule_film_v1_20261120.csv")
OUT_POP = os.path.join(D, "r2a_stage_population_v2_nofolder_20261128.csv")
OUT_TAB = os.path.join(D, "r2a_stage_hearing_table_v2_20261128.csv")
OUT_KEY = os.path.join(D, "r2a_stage_hearing_key_v2_20261128.csv")
ORDER = ("清書", "下絵", "なし")
CAP = 2


def main() -> int:
    print("[便R2a のフォルダーの層の引き直し(Track R の依頼、R2 追補3)]")
    c = db.connect()
    fol = {(p, i): bool(f) for p, i, f in c.execute("select page_path, layer_id, folder from layers").fetchall()}
    pop = list(csv.DictReader(open(POP, encoding="utf-8")))
    for x in pop:
        x["layer_id"] = int(x["layer_id"])
    keyed = {(x["page_path"], x["layer_id"]): x for x in pop}
    miss = sum((x["page_path"], x["layer_id"]) not in fol for x in pop)
    assert miss == 0, f"構造 DB に無い層 {miss}"
    nf = [x for x in pop if not fol[(x["page_path"], x["layer_id"])]]
    print(f"    母集団(追補1 の門の後){len(pop):,} 層 → 門 6 構造 DB folder 真 {len(pop) - len(nf):,} を外す → "
          f"**追補3 の母集団 {len(nf):,} 層**(作品フォルダー {len({x['work_dir'] for x in nf})})")
    for nr in ORDER + ("両方当たる",):
        a = sum(x["name_rule"] == nr for x in pop)
        b = sum(x["name_rule"] == nr for x in nf)
        print(f"      name_rule {nr:6s} {a:8,d} → {b:8,d}(フォルダー {a - b:,})")
    with open(OUT_POP, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(nf[0])); w.writeheader(); w.writerows(nf)

    k0 = list(csv.DictReader(open(KEY0, encoding="utf-8")))
    drawn = {(r["page_path"], int(r["layer_id"])) for r in k0}
    keep = [keyed[(r["page_path"], int(r["layer_id"]))] for r in k0 if not fol[(r["page_path"], int(r["layer_id"]))]]
    drop = [r for r in k0 if fol[(r["page_path"], int(r["layer_id"]))]]
    need = collections.Counter(r["name_rule"] for r in drop)
    print(f"\n    便R2a の20層: 残す(フォルダーでない){len(keep)} / 引き直す(フォルダー){len(drop)} —— "
          + " / ".join(f"{nr} {need[nr]}" for nr in ORDER))
    rng = random.Random(0)
    new = []
    for nr in ORDER:
        per = collections.Counter(x["work_dir"] for x in keep if x["name_rule"] == nr)
        cand = sorted((x for x in nf if x["name_rule"] == nr and (x["page_path"], x["layer_id"]) not in drawn),
                      key=lambda x: (x["page_path"], x["layer_id"]))
        rng.shuffle(cand)
        got, skip_cap = [], 0
        for x in cand:
            if len(got) >= need[nr]:
                break
            if per[x["work_dir"]] >= CAP:
                skip_cap += 1
                continue
            per[x["work_dir"]] += 1
            got.append(x)
        print(f"      層 {nr}: 候補 {len(cand):,}(既に引いた20層を除く)→ 引き直し {len(got)}/{need[nr]}"
              f"(上限で飛ばした {skip_cap})")
        assert len(got) == need[nr]
        new += got
    picks = keep + new
    src = {id(x): ("残す" if x in keep else "引き直し") for x in picks}
    rng.shuffle(picks)
    assert len(picks) == 20 and len({(x["page_path"], x["layer_id"]) for x in picks}) == 20
    assert not any(fol[(x["page_path"], x["layer_id"])] for x in picks)
    cap = collections.Counter((x["name_rule"], x["work_dir"]) for x in picks)
    print(f"    上限の点検: (層, 作品フォルダー) の最大 {max(cap.values())}(≤ {CAP})")

    tab_cols = ["row", "page_path", "layer_id", "layer_name", "work_dir", "year_label",
                "self_shown", "ancestors_shown", "answer", "comment"]
    with open(OUT_TAB, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=tab_cols); w.writeheader()
        for i, x in enumerate(picks, 1):
            w.writerow({**{k: x[k] for k in tab_cols if k in x}, "row": i, "answer": "", "comment": ""})
    with open(OUT_KEY, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["row", "page_path", "layer_id", "name_rule", "r2a_v1"])
        w.writeheader()
        for i, x in enumerate(picks, 1):
            w.writerow(dict(row=i, page_path=x["page_path"], layer_id=x["layer_id"], name_rule=x["name_rule"],
                            r2a_v1=src[id(x)]))
    hid = sum(not (x["self_shown"] == "1" and x["ancestors_shown"] == "1") for x in picks)
    print(f"    標本: 作品フォルダー {len({x['work_dir'] for x in picks})} / 非表示(自分か祖先){hid} / 20"
          f" / 新しい層の非表示 {sum(not (x['self_shown'] == '1' and x['ancestors_shown'] == '1') for x in new)}")
    print("    年代札(全20): " + " / ".join(f"{y} {v}" for y, v in sorted(collections.Counter(x['year_label'] for x in picks).items())))

    print("\n    ついでに数える(記録。棒は動かさない):")
    r4 = [(r["page_path"], int(r["layer_id"])) for r in csv.DictReader(open(R4KEY, encoding="utf-8"))]
    print(f"      R4 の {len(r4)} 層のうち folder 真 {sum(fol.get(k, False) for k in r4)}"
          f"(構造 DB に無い {sum(k not in fol for k in r4)})")
    fc = collections.Counter()
    for r in csv.DictReader(open(FILM, encoding="utf-8")):
        if r["name_rule_film"] == "あたり+下絵(膜)":
            k = (r["page_path"], int(r["layer_id"]))
            fc[(r["in_r4_population"], fol.get(k))] += 1
    for inn, lab in (("1", "確かめた母集団の中"), ("0", "外")):
        t = fc[(inn, True)] + fc[(inn, False)] + fc[(inn, None)]
        print(f"      name_rule_film あたり+下絵(膜) {lab} {t:,} 層のうち folder 真 {fc[(inn, True)]:,}"
              f"(構造 DB に無い {fc[(inn, None)]})")
    print(f"\n    出力: {OUT_POP}\n          {OUT_TAB}\n          {OUT_KEY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
