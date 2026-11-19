#!/usr/bin/env python
"""**対 v6 の線画・下絵の層すべてに `name_rule` を引く**(Track S の頼み、2026-11-20)。

- 母集団: QC 表 v6(`clip_pairs_qc_v6_20261114.csv`、1,726頁)の各行。
  manifest の行は `source_path` で引く(manifest は S の木の `data/clip_pairs_v6` を**読むだけ**)
- 層の名前は**構造 DB の `layers`**(断面は `db.py` の控え)から。manifest の
  `line_name` / `sketch_name` は統合名(`merged[5]: …`)のことがあるので使わない
- `name_rule` は役割の表 v3 と同じ関数(`stage_hearing.name_rule`)。**名前だけの関数**
  なので、表 v3 に無い層にもそのまま当てられる。新しい列ではない
- 落としたもの(manifest に行が無い頁・DB に層が無い層)は理由別に数えて印字する(規律9)

出力は `data/census_full/`(頁のパスと層名を持つので**公開しない・追跡しない**)。
"""
import collections
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
from stage_hearing import name_rule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QC = os.path.join(ROOT, "inbox", "clip_pairs_qc_v6_20261114.csv")
MANIFESTS = os.path.join(ROOT, "..", "lineart-cleanup-delta", "data", "clip_pairs_v6",
                         "clip_pairs", "**", "manifest.json")
OUT = os.path.join(ROOT, "data", "census_full", "s_v6_pairs_layer_name_rule_20261120.csv")
OUT_PAGES = os.path.join(ROOT, "data", "census_full", "s_v6_pairs_page_name_rule_20261120.csv")


def main() -> int:
    qc = list(csv.DictReader(open(QC, encoding="utf-8")))
    man = {}
    for f in glob.glob(MANIFESTS, recursive=True):
        for e in json.load(open(f, encoding="utf-8")):
            man.setdefault(e["source_path"], []).append(e)
    c = db.connect()
    names = {(p, i): n for p, i, n in c.execute(
        "select page_path, layer_id, name from layers").fetchall()}

    drop = collections.Counter()
    rows, pages = [], []
    for q in qc:
        p = q["source_path"]
        es = man.get(p)
        if not es:
            drop["頁: manifest に行が無い"] += 1
            continue
        if len(es) > 1:
            drop["頁: manifest に行が2つ以上(最初の行を使った)"] += 1
        e = es[0]
        per = {}
        for side, key in (("line", "line_ids"), ("sketch", "sketch_ids")):
            rules = []
            for lid in e.get(key) or []:
                n = names.get((p, int(lid)))
                if n is None:
                    drop[f"層: DB に無い({side})"] += 1
                    r = "DB に無い"
                else:
                    r = name_rule(n)
                rules.append(r)
                rows.append(dict(page_path=p, work_id=q["work_id"], pair_quality=q["pair_quality"],
                                 side=side, layer_id=lid, layer_name=n if n is not None else "",
                                 name_rule=r, selection=e.get(f"{side}_selection", "")))
            per[side] = "+".join(sorted(set(rules))) if rules else "層なし"
        pages.append(dict(page_path=p, work_id=q["work_id"], pair_quality=q["pair_quality"],
                          line_rules=per["line"], sketch_rules=per["sketch"],
                          line_selection=e.get("line_selection", ""),
                          sketch_selection=e.get("sketch_selection", "")))

    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT_PAGES, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pages[0])); w.writeheader(); w.writerows(pages)

    print(f"QC 表 v6: {len(qc)} 頁 / manifest の行(source_path): {len(man)}")
    print(f"層の行: {len(rows)} / 頁の行: {len(pages)}")
    print("落とした・注意(理由別):")
    for k, v in sorted(drop.items()):
        print(f"  {v:6d}  {k}")
    for qual in ("ok", None):
        sub = [x for x in pages if qual is None or x["pair_quality"] == qual]
        print(f"\n頁の組({'pair_quality = ok' if qual else '全部'}、{len(sub)} 頁): 線画の側 × 下絵の側")
        for (a, b), v in collections.Counter((x["line_rules"], x["sketch_rules"]) for x in sub).most_common():
            print(f"  {v:6d}  線画 {a} / 下絵 {b}")
    print("\n層の name_rule(側ごと):")
    for (s, r), v in sorted(collections.Counter((x["side"], x["name_rule"]) for x in rows).items()):
        print(f"  {v:6d}  {s} {r}")
    print(f"\n出力: {OUT}\n      {OUT_PAGES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
