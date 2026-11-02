#!/usr/bin/env python
"""**作り直された材料(`_v2`)を、前の版と鍵ごとに突き合わせる**(2026-11-02)。

抽出側は読み手の2つの不具合(1ビットを空として読む / zlib の頭の取り違え)を直し、
**この集計の材料のうち4本を作り直して送ってきた。** **こちらの公開表はこの日から `_v2` を読む。**

**過去の結果は撤回しない**(規律10)—— **前の版の表は git の履歴に在り、前後の数をこの表に並べる。**
**作り直しは読み手の直しで、点数と独立だから棒ではない**(配達・器具の破損)。

**この器具が答える問い**: **動いた行は、全部その直しで説明が付くか。**
**説明は鍵で引く** —— 影響表(`zlib_affected_layers`、本体とマスク)と
1ビットの面の一覧(`zlib_1bit_sample_planes`、本体 6,587 / マスク 683)に、`page_path` + `layer_id` で。
**面も合わせる** —— 本体だけを読む材料で、マスクにしか当たっていない層が動いたら、説明が付いていない。

**既知セル**: **供給元が便に書いた「動いた行数」。** **この器具が作っていない入力**(T132)。

**届く範囲**(名前より狭く書く): **「動いた行が、直しの当たった鍵の中に在るか」まで。**
**「値が正しく直ったか」は届かない** —— 直した読み手の出力を、こちらは画素で確かめられない。
**1ビットのマスクの一覧は抜き取り**(各面の先頭3ブロック)なので、**マスクの1ビットで説明が付かない
行は「説明なし」ではなく「一覧が覆っていない」かもしれない** —— 行を分けて印字する。

    ./venv/bin/python tools/census/material_versions.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ZLIB = "inbox/zlib_affected_layers_20261101.csv"
BIT1 = "inbox/zlib_1bit_sample_planes_20261101.csv"

# (名前, 前の版(重ねる便があれば順に), 後の版, 読む面, 比べない列,
#  供給元が書いた動いた行数(列ごと。`*` は行の数), 便,
#  直し以外の宣言された理由(行 → 理由 or None))
MATERIALS = (
    ("依頼3の層の種別",
     ("inbox/f_request3_layer_kind_rendered_20261026.csv",),
     "inbox/f_request3_layer_kind_rendered_v2_20261102.csv", ("body",), (),
     {"drawn_ratio": 988, "v_layer_kind": 1276}, "deliver_b1_layer_colour_v2_20261102.md",
     "drawn_ratio 988 行 + `v_layer_kind` other → raster 1,276 行"),
    ("薄いアルファの帯",
     ("inbox/faint_alpha_905_20261027.csv",),
     "inbox/faint_alpha_905_v2_20261102.csv", ("body",), (),
     {"*": 1060}, "deliver_faint_alpha_905_v2_20261102.md", "取り違え 89 + 1ビット 971"),
    ("マスクの被覆",
     ("inbox/mask_cover_request3_20261026.csv",
      "inbox/mask_cover_request3_maskorigin55_20261030.csv"),
     "inbox/mask_cover_request3_v2_20261102.csv", ("mask",), ("rebuilt_20261102",),
     {"*": 113}, "deliver_mask_cover_request3_v2_20261102.md", "取り違え 99 + 1ビットのマスク 14"),
    ("墨の濃さ2段目(札 2019)",
     ("inbox/ink_darkness_stage2_2019_20261031.csv",),
     "inbox/ink_darkness_stage2_2019_v2_20261102.csv", ("body",), ("n_blocks_unreadable",),
     {"*": 2}, "deliver_census_ink_stage2_2019_v2_and_answers_20261102.md", "形式5 の `線画` 2層"),
)


def declared_other(name, old, new):
    """**直し以外に、便が理由を宣言して動いた行。** 宣言の在る材料だけ"""
    if name == "依頼3の層の種別":
        # 抽出側: 10-26 17:57 の DB の変更(`26bd8e1`、マスクを持つラスタを `raster` と数える)
        if (old["v_layer_kind"] == "other" and new["v_layer_kind"] == "raster"
                and new["layer_type"].strip() == "3"
                and all(old[c] == new[c] for c in old if c != "v_layer_kind")):
            return "DB の種別の変更(`layer_type = 3` を raster と数える。読み手の直しではない)"
    return None


def keyed(path):
    with open(os.path.join(ROOT, path), encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows


def key_of(r):
    return (r["page_path"], r["layer_id"].strip())


def load_hits():
    hits = collections.defaultdict(set)
    for path, cause in ((ZLIB, "取り違え"), (BIT1, "1ビット")):
        with open(os.path.join(ROOT, path), encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                hits[(cause, r["plane"])].add(key_of(r))
    return hits


def compare(name, olds, new_path, planes, ignore, hits):
    base = keyed(olds[0])
    by_key = {key_of(r): r for r in base}
    for patch in olds[1:]:               # **重ねる便は鍵で上書き**(build_census と同じ)
        for q in keyed(patch):
            if key_of(q) in by_key:
                by_key[key_of(q)].update(q)
    new = keyed(new_path)
    new_by = {key_of(r): r for r in new}
    order_same = [key_of(r) for r in base] == [key_of(r) for r in new]
    # **供給元は「頁の順」を約束した**(層の順は約束していない)—— 頁の初出の順で比べる
    pages_old = list(dict.fromkeys(r["page_path"] for r in base))
    pages_new = list(dict.fromkeys(r["page_path"] for r in new))
    page_order_same = pages_old == pages_new
    added = set(new_by) - set(by_key)
    removed = set(by_key) - set(new_by)
    moved, cols, causes = [], collections.Counter(), collections.Counter()
    for k, o in by_key.items():
        n = new_by.get(k)
        if n is None:
            continue
        diff = [c for c in o if c not in ignore and (o.get(c) or "") != (n.get(c) or "")]
        if not diff:
            continue
        moved.append(k)
        cols.update(diff)
        got = [f"{cause}({plane})" for (cause, plane), ks in sorted(hits.items())
               if k in ks and plane in planes]
        if got:
            causes[" + ".join(got)] += 1
            continue
        other = declared_other(name, o, n)
        if other:
            causes[other] += 1
            continue
        wrong_plane = [f"{cause}({plane})" for (cause, plane), ks in sorted(hits.items())
                       if k in ks]
        if wrong_plane:
            causes["**説明なし**(当たったのは読まない面: " + " + ".join(wrong_plane) + ")"] += 1
        else:
            causes["**説明なし**(どの一覧にも無い)"] += 1
    return dict(n_old=len(by_key), n_new=len(new), order_same=order_same,
                page_order_same=page_order_same,
                added=len(added), removed=len(removed), moved=len(moved),
                cols=cols, causes=causes)


def main() -> int:
    print("  [作り直された材料の前後(`_v2`)]")
    hits = load_hits()
    names = tuple(m[0] for m in MATERIALS)
    if names != bc.MATERIALS_REBUILT:
        raise SystemExit(f"**材料の名前が白名簿と違う**: {names} 対 {bc.MATERIALS_REBUILT}")
    rows, rc = [], 0
    for name, olds, new, planes, ignore, claimed, note_src, claimed_txt in MATERIALS:
        c = compare(name, olds, new, planes, ignore, hits)
        unexplained = sum(n for k, n in c["causes"].items() if k.startswith("**説明なし"))
        got = {col: (c["moved"] if col == "*" else c["cols"][col]) for col in claimed}
        agree = got == claimed
        ok = agree and not unexplained and not c["added"] and not c["removed"]
        rc |= 0 if ok else 1
        print(f"    {name}: 行 {c['n_old']:,} → {c['n_new']:,} / 増えた {c['added']} / "
              f"消えた {c['removed']} / 頁の順 {'同じ' if c['page_order_same'] else '**違う**'}"
              f"(層の順 {'同じ' if c['order_same'] else '違う'})/ **動いた行 {c['moved']:,}** / "
              "供給元の数 " + " ".join(f"{col if col != '*' else '行'} {n:,}→{got[col]:,}"
                                     for col, n in claimed.items())
              + f" **{'一致' if agree else '不一致'}**")
        for k, n in sorted(c["causes"].items(), key=lambda kv: -kv[1]):
            print(f"        {n:6,}  {k}")
        print("        動いた列: " + " / ".join(f"{col} {n:,}" for col, n in c["cols"].most_common()))
        for k, n in sorted(c["causes"].items(), key=lambda kv: -kv[1]):
            rows.append([name, " + ".join(os.path.basename(p) for p in olds),
                         os.path.basename(new), "/".join(planes),
                         c["n_old"], c["n_new"], c["added"], c["removed"],
                         "yes" if c["page_order_same"] else "no", c["moved"],
                         " / ".join(f"{col if col != '*' else '行'} {n}" for col, n in claimed.items()),
                         k, n, unexplained,
                         " / ".join(f"{col} {m}" for col, m in c["cols"].most_common()),
                         "census track(前後の版を鍵で突き合わせた)", "2026-11-02",
                         note_src, f"供給元が書いた内訳: {claimed_txt}"])
    rows += [[""] * 18 + [n] for n in NOTE]
    n = bc.write_csv(os.path.join(ROOT, "census", "material_versions.csv"),
                     ["material_rebuilt", "before", "after", "planes_read", "n_rows_before",
                      "n_rows_after", "n_rows_added", "n_rows_removed", "same_page_order",
                      "n_rows_moved", "moved_claimed", "cause", "n_rows_cause",
                      "n_rows_unexplained", "columns_moved", "by", "as_of", "source",
                      "note"], rows)
    print(f"    material_versions.csv {n}行 —— "
          + ("**動いた行は全部、直しか宣言された理由で説明が付き、供給元の数と一致した**"
             if not rc else "**一致しない材料が在る。表を引く前に読む**"))
    return rc


NOTE = [
    "**2026-11-02 から、公開表はこの4本の材料の `_v2` を読む** —— "
    "**前の版の表は撤回しない**(規律10。前の数は git の履歴に在る)。**作り直しは読み手の直しで棒ではない**",
    "**届く範囲は「動いた行が、直しの当たった鍵の中に在るか」まで** —— "
    "**値が正しく直ったかは届かない**(直した読み手の出力を、こちらは画素で確かめられない)",
    "**面を合わせて説明する** —— 本体だけを読む材料(層の種別・薄いアルファ・墨の濃さ)は本体の当たりだけで、"
    "マスクの被覆はマスクの当たりだけで説明する。**読まない面に当たった鍵で説明が付く行は「説明なし」に数える**",
    "**1ビットの面の一覧は抜き取りである**(各面の先頭3ブロック。抽出側が明記)—— "
    "**マスクの1ビットはこの一覧に無いことが在りうる**",
    "**依頼3の層の種別の 1,276 行は読み手の直しではない** —— **10-26 17:57 の構造 DB の変更**"
    "(`layer_type = 3` を raster と数える)で、**前の版はその変更の前に作られていた**(抽出側の説明。"
    "器具は**種別だけが other → raster に動き、`layer_type` が 3 で、他の列が全部同じ行**を数える)",
    "**既知セルは供給元が便に書いた「動いた行数」**(この器具が作っていない入力。T132)",
]

if __name__ == "__main__":
    raise SystemExit(main())
