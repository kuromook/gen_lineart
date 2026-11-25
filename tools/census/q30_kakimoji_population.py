#!/usr/bin/env python
"""**Q30: 描き文字の母集団の一覧**(Track Q の頼み `note_q_census_q30_kakimoji_population_20261125.md`、2026-11-25。
**新しい走行ではない** —— 構造 DB(10-26 の断面)と census の既存の表をつなぐだけ)。

対象(1行1層):
- (a) `kaki_name` = 層の名前に 描き文字・書き文字・かき文字 を含む(`starts_shita` = 名前が `下絵` で始まる、で分かる)
- (b) `effect_af3` = 層の名前に `effect`(大小を問わない)を含み、`effect_line_guess_af3` が 0 か 1(census の推測の列 11-25)
- 両方に当たる層は1行で `target` に両方を書く

空の印(Q は空の規則を作らない。census の印をそのまま):
- `empty_c` = **定義C**(`stage_hearing_round2_b.empty_c` の正本の規則)を全 .clip の画素の便(`full_clip_pixels_layers_merged.csv`、11-13〜14)で当てた値。
  1 = 空 / 0 = 空でない / 空欄 = 印を付けられない(訳は `empty_c_why`: 便に層が無い・.lip・描画できなかった)
- `empty_c_format5` = 画素の形式5 を含む層(**形式5 の空は 10-31 から調査中で分析対象外**。a では「空だが保留」)
- 隣に a の判定 v9(`a_verdict_v9` / `a_rule_v9`)を置く。**a の理由は最初の1つだけ**なので、名前で落ちた層は空でも理由が「名前」になる —— 空の印は `empty_c` で読む

ほかの列: 年代札(`pages.year`)・作品フォルダー(`author_process.work_dir_of`)・恒久除外の頁か・作者の答えが既に在るか
(census の描き文字の42層 / Q29 の20行 / Q9 の IC2 の9層 = `effect_lines_author_answers_20261117.csv`)・写しの組の早い方か(`page_file_copies`)。
落とした・印の無い行は理由別に印字する(規律9)。出力 `data/census_full/q30/q30_kakimoji_population_20261125.csv`(非公開)。
"""
import collections
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
from author_process import work_dir_of  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
OUT = os.path.join(D, "q30", "q30_kakimoji_population_20261125.csv")
KAKI = re.compile(r"(描き文字|書き文字|かき文字)")


def rd(name):
    return csv.DictReader(open(os.path.join(D, name), encoding="utf-8"))


def main() -> int:
    c = db.connect()
    lay = c.execute("select l.page_path, l.layer_id, coalesce(l.name,''), l.parent_id, v.v_layer_kind, p.year, p.work, p.format "
                    "from layers l join v_layer_kind v using (page_path, layer_id) join pages p using (page_path)").fetchall()
    byk = {(r[0], r[1]): r for r in lay}
    af3 = {(r["page_path"], int(r["layer_id"])): r["effect_line_guess_af3"] for r in rd("guess/effect_line_guess_af3_20261125.csv")}
    tgt = {}
    for r in lay:
        k = (r[0], r[1]); t = []
        if KAKI.search(r[2]):
            t.append("kaki_name")
        if "effect" in r[2].lower() and af3.get(k) in ("0", "1"):
            t.append("effect_af3")
        if t:
            tgt[k] = t
    eff_all = sum(1 for r in lay if "effect" in r[2].lower())
    print(f"(a) 名前に描き文字の語: {sum('kaki_name' in t for t in tgt.values()):,} 層 / (b) 名前に effect {eff_all:,} 層のうち AF3 が 0・1 "
          f"{sum('effect_af3' in t for t in tgt.values()):,}(AF3 の表に無い {sum(1 for r in lay if 'effect' in r[2].lower() and (r[0], r[1]) not in af3):,}・"
          f"表に在って判じていない {sum(1 for r in lay if 'effect' in r[2].lower() and af3.get((r[0], r[1])) == ''):,})/ 両方 {sum(len(t) == 2 for t in tgt.values())}")
    need = set(tgt)
    px = {}
    for r in rd("full_clip_pixels_layers_merged.csv"):
        k = (r["page_path"], int(r["layer_id"]))
        if k in need:
            px[k] = r
    av = {}
    for r in rd("a_verdict_v9_20261124.csv"):
        k = (r["page_path"], int(r["layer_id"]))
        if k in need:
            av[k] = r
    ex = exclusions.load()
    ans = collections.defaultdict(list)
    for r in rd("kakimoji_author_answers_20261125.csv"):
        ans[(r["page_path"], int(r["layer_id"]))].append("census描き文字42")
    for r in rd("q29/q29_author_layer_answers_20261125.csv"):
        ans[(r["page_path"], int(r["layer_id"]))].append("Q29")
    for r in rd("effect_lines_author_answers_20261117.csv"):
        ans[(r["page_path"], int(r["layer_id"]))].append("Q9_IC2")
    cp = {r["page_path"]: r for r in rd("page_file_copies_20261123.csv")}
    out = []
    for k in sorted(tgt):
        r = byk[k]
        anc, q = [], r[3]
        while q is not None and (k[0], q) in byk:
            a = byk[(k[0], q)]; anc.append(a[2]); q = a[3]
        p = px.get(k)
        if r[7] != "clip":
            ec, why = "", "lip(画素が無い)"
        elif p is None:
            ec, why = "", "画素の便に層が無い"
        elif p["render_status"] == "描画できなかった":
            ec, why = "", "描画できなかった"
        else:
            ec, why = p["empty_c"], ""
        a = av.get(k, {})
        out.append(dict(page_path=k[0], layer_id=k[1], layer_name=r[2], ancestors="/".join(reversed(anc)), v_layer_kind=r[4],
                        year_label=r[5], work_dir=work_dir_of(k[0], r[5], r[6]) or "", format=r[7], target=";".join(tgt[k]),
                        starts_shita=int(r[2].startswith("下絵")), effect_line_guess_af3=af3.get(k, ""),
                        empty_c=ec, empty_c_why=why, empty_c_format5=int(bool(p) and "5" in p["pixel_formats"].split(";")),
                        a_verdict_v9=a.get("a_verdict_v9", ""), a_rule_v9=a.get("a_rule_v9", ""),
                        permanently_excluded_page=int(k[0] in ex), author_answer_in=";".join(sorted(set(ans.get(k, [])))),
                        earliest_in_copy_group=cp.get(k[0], {}).get("earliest_in_group", ""),
                        copy_group_size=cp.get(k[0], {}).get("copy_group_size", "")))
    for t in ("kaki_name", "effect_af3"):
        s = [o for o in out if t in o["target"]]
        print(f"\n[{t}] 層 {len(s):,} / 頁 {len({o['page_path'] for o in s}):,}")
        print("  空の印 empty_c: " + " / ".join(f"{k or '印なし'} {v:,}" for k, v in collections.Counter(o["empty_c"] for o in s).most_common())
              + "(印なしの訳: " + (" / ".join(f"{k} {v:,}" for k, v in collections.Counter(o["empty_c_why"] for o in s if o["empty_c"] == "").items()) or "0") + ")")
        print(f"  空のうち形式5(調査中): {sum(o['empty_c'] == '1' and o['empty_c_format5'] for o in s):,}")
        print(f"  テキストの層 {sum(o['v_layer_kind'] == 'text' for o in s):,} / 恒久除外の頁 {sum(o['permanently_excluded_page'] for o in s):,} / "
              f"答えが在る {sum(bool(o['author_answer_in']) for o in s):,} / 下絵で始まる {sum(o['starts_shita'] for o in s):,} / "
              f"写しの遅い方 {sum(o['earliest_in_copy_group'] == '0' for o in s):,}")
        print("  種別: " + " / ".join(f"{k} {v:,}" for k, v in collections.Counter(o["v_layer_kind"] for o in s).most_common()))
        print("  形式: " + " / ".join(f"{k} {v:,}" for k, v in collections.Counter(o["format"] for o in s).most_common()))
        print("  年代札: " + " / ".join(f"{k} {v:,}" for k, v in sorted(collections.Counter(o["year_label"] for o in s).items())))
        rest = [o for o in s if o["empty_c"] != "1" and o["v_layer_kind"] != "text" and not o["permanently_excluded_page"]
                and not o["author_answer_in"]]
        print(f"  参考: 空(1)・テキスト・恒久除外・答えが在る を除いた残り {len(rest):,}(印なしは残りに入れた)")
    print(f"\n作者の答えの在る層: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(
        x for o in out for x in o["author_answer_in"].split(";") if x).items()) + f"(在る答えの層のうち対象に入らない "
          f"{sum(1 for k in ans if k not in tgt)})")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"出力: {OUT}({len(out):,}行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
