#!/usr/bin/env python
"""**全層の層名の棚卸し** —— Track E の依頼(`note_census_template_first_hold_hearing_20261110.md` §2(a))。

**ユーザーの指示(2026-11-10、逐語。E の便から)**: 「まず単純なところから始められないか　PICOトンボや枠などのテンプレートは
全く同じレイヤでしかない　レイヤ名もぶれていない　これらは決定論的に除外できるはずだ」

**census がするのは数えることだけ。** 何を落とすかの表を決めるのは E(と作者)。

**決めたこと(数える前に)**:
- 母集団: 構造 DB の全層(`.clip` と `.lip`)。恒久除外の頁の層は外し、数を印字する
- **名前は文字列の完全一致で束ねる**(大文字小文字・全角半角・空白を畳まない。「ぶれ」は束ねると見えなくなる)。
  名前が無い層は `""` の行に入れる
- 行 = (層名, 種別 `v_layer_kind`)。同じ名前が種別をまたぐなら別の行になる(種別の列で見分ける)
- 列: 層の数 / **作品単位の数**(`split_units.py` の単位)/ 作品フォルダーの数 / 頁の数 / `.clip` の頁 / `.lip` の頁
- 並びは作品単位の数の多い順
- **出力は非公開**(層名は公開の表に書かない)。公開表には何も足さない

    ./venv/bin/python tools/census/layer_name_inventory.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = "data/census_full/layer_name_inventory_20261110.csv"


def main() -> int:
    print("[全層の層名の棚卸し(Track E の依頼)]")
    con = db.connect()
    ex = exclusions.load()
    un = split_units.units(con)
    rows = con.execute("""select l.page_path, coalesce(l.name, ''), k.v_layer_kind, p.format
                            from layers l join v_layer_kind k using (page_path, layer_id) join pages p using (page_path)""").fetchall()
    agg = collections.defaultdict(lambda: dict(n=0, units=set(), wds=set(), pages=set(), clip=set(), lip=set()))
    n_ex = n_nounit = 0
    for p, name, kind, fmt in rows:
        if p in ex:
            n_ex += 1
            continue
        w = wd(p)
        u = un.get(w)
        if u is None:
            n_nounit += 1
            u = "(単位なし)" + w
        a = agg[(name, kind)]
        a["n"] += 1
        a["units"].add(u)
        a["wds"].add(w)
        a["pages"].add(p)
        (a["clip"] if fmt == "clip" else a["lip"]).add(p)
    n = sum(a["n"] for a in agg.values())
    print(f"    層 {len(rows):,} → 恒久除外の頁の層 {n_ex} を外して {n:,}(全部の層が行に入った: {n + n_ex == len(rows)})")
    print(f"    作品単位の無い作品フォルダーの層 {n_nounit:,}(フォルダー名を単位として数えた)")
    out = sorted(agg.items(), key=lambda kv: (-len(kv[1]["units"]), -kv[1]["n"], kv[0]))
    print(f"    行(層名 × 種別) {len(out):,} / 異なる層名 {len({k[0] for k in agg}):,}")
    for th in (3, 5, 10, 20):
        sel = [kv for kv in out if len(kv[1]["units"]) >= th]
        print(f"      作品単位 {th} 以上にまたがる行 {len(sel):,}(層 {sum(kv[1]['n'] for kv in sel):,})")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer_name", "v_layer_kind", "n_layers", "n_split_units", "n_work_dirs", "n_pages", "n_clip_pages", "n_lip_pages"])
        for (name, kind), a in out:
            w.writerow([name, kind, a["n"], len(a["units"]), len(a["wds"]), len(a["pages"]), len(a["clip"]), len(a["lip"])])
    print(f"    → `{OUT}`(非公開。層名を含む)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
