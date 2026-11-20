"""Track S 段0: 対 v6 の母集団を数える(選別は測定 = 落とした数を理由別に印字する)。

正本は QC 表 v6(`clip_pairs_qc_v6.csv`、1,726行)。manifest は画像の在りかを引くためだけに使う。
(11-20: `unzip` が UTF-8 の印を読まず `005_2016_ng/新しいフォルダー_n/manifest.json` の `[]` を化けた名の
フォルダーに書き、v5 の20行が残っていた。zipfile で書き直し済み。器具は両者の行数の一致を印字する)

門(条例4・8)は上から順に当て、最初に当たった理由で数える:
  1. 恒久除外(census `author_exclusions.csv`)
  2. pair_quality != ok(理由は pair_quality の値そのもの)
  3. 名前の規則の厳しい門(ユーザー 2026-11-20「厳しいを選択」): 線画の側の層の name_rule が全部「清書」、
     下絵の側の層が全部「下絵」。表は census `s_v6_pairs_page_name_rule_20261120.csv`(理由は 線画側/下絵側 の組)
重複(`is_primary`)と自動の位置判定(`aligned`)は落とさずに列として印字する(使い方は登録で決める)。
世代・作品はプールしない: 作品フォルダー(`work_id`、年を含む)ごとに1行。
"""
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "data/clip_pairs_v6/clip_pairs/clip_pairs_qc_v6.csv"
PAIRS = ROOT / "data/clip_pairs_v6/clip_pairs"
EXCL = ROOT.parent / "lineart-census/data/census_full/author_exclusions.csv"
NAME_RULE = ROOT / "inbox/s_v6_pairs_page_name_rule_20261120.csv"
OUT = ROOT / "results/s0_population_v6.csv"


def main():
    qc = list(csv.DictReader(open(QC, encoding="utf-8")))
    excl = {r["page_path"] for r in csv.DictReader(open(EXCL, encoding="utf-8"))}
    nr = {r["page_path"]: r for r in csv.DictReader(open(NAME_RULE, encoding="utf-8"))}
    assert set(nr) == {r["source_path"] for r in qc}, "name_rule の表と QC 表の頁が一致しない"

    # manifest と QC 表の突き合わせ
    man_lines = set()
    for m in PAIRS.glob("**/manifest.json"):
        for r in json.load(open(m, encoding="utf-8")):
            man_lines.add(r["line"])
    qc_lines = {r["line"] for r in qc}
    print(f"QC 表 v6: {len(qc)} 行 / manifest の行: {len(man_lines)}")
    print(f"  manifest にあって QC 表に無い: {len(man_lines - qc_lines)}(使わない)")
    print(f"  QC 表にあって manifest に無い: {len(qc_lines - man_lines)}")
    missing_img = [r for r in qc if not any((PAIRS / r["work_id"]).glob(f"*/{r['line']}"))
                   or not any((PAIRS / r["work_id"]).glob(f"*/{r['sketch']}"))]
    print(f"  QC 表の行で画像が見つからない: {len(missing_img)}")

    dropped = collections.Counter()
    kept = []
    for r in qc:
        if r["source_path"] in excl:
            dropped["恒久除外(author_exclusions)"] += 1
        elif r["pair_quality"] != "ok":
            dropped[f"pair_quality={r['pair_quality']}"] += 1
        elif not (nr[r["source_path"]]["line_rules"] == "清書" and nr[r["source_path"]]["sketch_rules"] == "下絵"):
            x = nr[r["source_path"]]
            dropped[f"name_rule 線画={x['line_rules'] or '(空)'} / 下絵={x['sketch_rules'] or '(空)'}"] += 1
        else:
            kept.append(r)
    print("\n落とした頁(最初に当たった理由):")
    for k, v in sorted(dropped.items(), key=lambda x: -x[1]):
        print(f"  {k:45s} {v:5d}")
    print(f"  {'計':45s} {sum(dropped.values()):5d}")
    print(f"残り(恒久除外の外・ok・厳しい門): {len(kept)}")
    print(f"  うち is_primary=True(内容重複の代表): {sum(r['is_primary'] == 'True' for r in kept)}")
    print(f"  うち aligned=True(自動の位置判定): {sum(r['aligned'] == 'True' for r in kept)}")

    allw = collections.Counter(r["work_id"] for r in qc)
    rows = collections.defaultdict(lambda: collections.Counter())
    for r in kept:
        c = rows[r["work_id"]]
        c["ok"] += 1
        c["primary"] += r["is_primary"] == "True"
        c["aligned"] += r["aligned"] == "True"
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["work_id", "qc_rows", "kept", "kept_primary", "kept_aligned"])
        print(f"\n作品フォルダーごと({len(allw)} 作品。残りが 0 の作品も出す)")
        print(f"  {'work_id':32s} {'QC':>4s} {'残り':>4s} {'代表':>4s} {'自動整列':>6s}")
        for wid in sorted(allw):
            c = rows[wid]
            w.writerow([wid, allw[wid], c["ok"], c["primary"], c["aligned"]])
            print(f"  {wid:32s} {allw[wid]:4d} {c['ok']:4d} {c['primary']:4d} {c['aligned']:6d}")
    print(f"\n表: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
