#!/usr/bin/env python
"""**ComicStudio(`.cpg`)v3 の層名の棚卸し** —— Track LS の依頼(`note_census_ls_cpg_v3_name_inventory_20261128.md`)。
ユーザーの承認は LS の便の逐語(2026-11-28「1,2,3 すすめて」)。

**census がするのは数えることだけ。** 何を落とすかは LS と作者が決める。拡張子を種別の代わりに使わない(規律13)。

**決めたこと(数える前に)**:
- 母集団: 抽出側の v3 の層の表(`comicstudio_v3_layers_PRIVATE_20261128.csv`、216,622層。型紙の塊と `.cru` は抽出側で除いてある)。
  **`.clip` の棚卸しとも、第1段 v2 とも1つの表にしない**(母集団が違う)
- 恒久除外の名簿を門にする: 頁の道の完全一致と、作品フォルダーの一致の両方で当て、外した層の数を印字する
- 行 = (`ext`, 層名)。**名前は文字列の完全一致**(`.clip` の棚卸しと同じ。畳まない)。名前が無い層は `""` の行
- **`.ctx`(テキスト)は名前を出さず1行に畳む**(奥付に作者の個人情報がある)
- 列: 層 / 頁 / 作品単位(`split_units.py` の単位。単位の無い作品フォルダーはフォルダー名を単位にして、その層の数を印字)/
  作品フォルダー / 版 4 と版 5 の層(畳まない)/ 非表示(`visible` 0)/ 祖先の最上位の名前の上位3件 /
  祖先の上から2段の上位3件 / `.cra` は 読めた・空・読めて描いた画素 0 の層の数
- 並びは作品単位の多い順、次に層の多い順
- **出力は非公開**(`data/` 配下)。公開表には何も足さない

    ./venv/bin/python tools/census/cpg_v3_name_inventory.py
"""
import collections
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = "data/census_full/cs_v3"
LAYERS = f"{D}/comicstudio_v3_layers_PRIVATE_20261128.csv"
PAGES = f"{D}/comicstudio_v3_pages_20261128.csv"
LAYERS_SHA = "f13b379d"
OUT = f"{D}/cpg_v3_name_inventory_PRIVATE_20261128.csv"
CTX_ROW = "(テキストの層。名前を出さない)"


def top(counter, k=3):
    return " | ".join(f"{a}:{n}" for a, n in counter.most_common(k))


def main() -> int:
    print("[ComicStudio v3 の層名の棚卸し(Track LS の依頼)]")
    sha = hashlib.sha256(open(os.path.join(ROOT, LAYERS), "rb").read()).hexdigest()
    print(f"    層の表 sha256 {sha[:16]}…(便の値の頭 {LAYERS_SHA}: {sha.startswith(LAYERS_SHA)})")
    if not sha.startswith(LAYERS_SHA):
        raise SystemExit("**層の表の sha が便の値と違う**")
    fmt = {r["cpg_path"]: r["format_version"] for r in csv.DictReader(open(os.path.join(ROOT, PAGES), encoding="utf-8"))}
    ex = exclusions.load()
    ex_wd = {wd(p) for p in ex}
    un = split_units.units(db.connect())
    agg = collections.defaultdict(lambda: dict(n=0, pages=set(), units=set(), wds=set(), v4=0, v5=0, vother=0, hid=0,
                                               top1=collections.Counter(), top2=collections.Counter(),
                                               cra_read=0, cra_empty=0, cra_other=0, cra_px0=0))
    n_all = n_ex_page = n_ex_wd = n_nounit = 0
    nounit_wds = set()
    with open(os.path.join(ROOT, LAYERS), encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            n_all += 1
            p = r["cpg_path"]
            if p in ex:
                n_ex_page += 1
                continue
            w = wd(p)
            if w in ex_wd:
                n_ex_wd += 1
                continue
            u = un.get(w)
            if u is None:
                n_nounit += 1
                nounit_wds.add(w)
                u = "(単位なし)" + w
            ext = r["ext"]
            name = CTX_ROW if ext == ".ctx" else r["name"]
            a = agg[(ext, name)]
            a["n"] += 1
            a["pages"].add(p)
            a["units"].add(u)
            a["wds"].add(w)
            v = fmt.get(p)
            a["v4" if v == "4" else "v5" if v == "5" else "vother"] += 1
            if r["visible"] == "0":
                a["hid"] += 1
            anc = r["ancestors"].split("/") if r["ancestors"] else []
            a["top1"][anc[0] if anc else "(祖先なし)"] += 1
            a["top2"]["/".join(anc[:2]) if anc else "(祖先なし)"] += 1
            if ext == ".cra":
                s = r["cra_status"]
                if s == "読めた":
                    a["cra_read"] += 1
                    if r["n_px"] == "0":
                        a["cra_px0"] += 1
                elif s.startswith("空"):
                    a["cra_empty"] += 1
                else:
                    a["cra_other"] += 1
    n = sum(a["n"] for a in agg.values())
    print(f"    層 {n_all:,} → 恒久除外(頁の道の一致){n_ex_page} / (作品フォルダーの一致){n_ex_wd} を外して {n:,}"
          f"(全部の層が行に入った: {n + n_ex_page + n_ex_wd == n_all})")
    print(f"    作品単位の無い作品フォルダー {len(nounit_wds):,}(層 {n_nounit:,}。フォルダー名を単位として数えた)")
    vo = sum(a["vother"] for a in agg.values())
    print(f"    版: 4 {sum(a['v4'] for a in agg.values()):,} / 5 {sum(a['v5'] for a in agg.values()):,} / 頁の表に無い・他 {vo}")
    out = sorted(agg.items(), key=lambda kv: (-len(kv[1]["units"]), -kv[1]["n"], kv[0]))
    by_ext = collections.Counter()
    for (ext, _), a in agg.items():
        by_ext[ext] += a["n"]
    print(f"    行(拡張子 × 層名){len(out):,} / 拡張子ごとの層 {dict(by_ext.most_common())}")
    cra = [a for (e, _), a in agg.items() if e == ".cra"]
    print(f"    .cra 読めた {sum(a['cra_read'] for a in cra):,}(描いた画素 0 {sum(a['cra_px0'] for a in cra)})"
          f" / 空 {sum(a['cra_empty'] for a in cra):,} / 他 {sum(a['cra_other'] for a in cra)}")
    print(f"    非表示 {sum(a['hid'] for a in agg.values())}")
    for th in (3, 5, 10, 20):
        sel = [kv for kv in out if len(kv[1]["units"]) >= th]
        print(f"      作品単位 {th} 以上にまたがる行 {len(sel):,}(層 {sum(kv[1]['n'] for kv in sel):,})")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ext", "name", "n_layers", "n_pages", "n_split_units", "n_work_dirs", "n_layers_v4", "n_layers_v5",
                    "n_visible0", "ancestor_top1", "ancestor_top2", "n_cra_read", "n_cra_read_px0", "n_cra_empty", "n_cra_other"])
        for (ext, name), a in out:
            c = ext == ".cra"
            w.writerow([ext, name, a["n"], len(a["pages"]), len(a["units"]), len(a["wds"]), a["v4"], a["v5"], a["hid"],
                        top(a["top1"]), top(a["top2"]),
                        a["cra_read"] if c else "", a["cra_px0"] if c else "", a["cra_empty"] if c else "",
                        a["cra_other"] if c else ""])
    print(f"    → {OUT}({len(out):,}行、非公開)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
