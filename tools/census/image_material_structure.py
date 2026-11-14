#!/usr/bin/env python
"""**画像素材の層(ラスタライズされていない取り込み画像)の構造の表** —— Track E の頼み(2026-11-14 の3通)。構造 DB と全頁の画素の表だけ。新しい走行はしない。

**数を見る前に決めたこと**:
- 母集団: 構造 DB の `has_image_material` = 真の層(.clip と .lip)から、恒久除外の頁の層を外したもの
- 列: 祖先の名前(`layer_role_table.page_order` と同じ歩き方)/ 層自身と祖先の表示(`visibility & 1`。ビット列を等値比較しない)/
  層のオフセット(`offset_x` / `offset_y`)/ 生のブロックの鍵(`ResizableImageInfo` / `MaterialContentType` / `LinkLayerInfo` の有無)
- .clip の層には a の判定 v2 の理由(`a_verdict_v2_20261114.csv`)を並べる
- 印字: `has_image_material` と種別 `image_material` の集合の差 / 見えている層の祖先の上位 / .lip の層の表示の内訳

    ./venv/bin/python tools/census/image_material_structure.py
"""
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402
from layer_role_table import page_order  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = "data/census_full/a_verdict_v2_20261114.csv"
OUT = "data/census_full/image_material_structure_20261114.csv"


def main() -> int:
    print("[画像素材の層の構造]")
    con = db.connect()
    ex = exclusions.load()
    rows = con.execute("""select l.page_path, l.layer_id, coalesce(l.name, ''), p.format, k.v_layer_kind, l.visibility & 1,
                                 l.offset_x, l.offset_y, cast(l.raw as varchar)
                            from layers l join pages p using (page_path) join v_layer_kind k using (page_path, layer_id)
                           where l.has_image_material""").fetchall()
    n0 = len(rows)
    rows = [r for r in rows if r[0] not in ex]
    print(f"    has_image_material の層 {n0:,} → 恒久除外の頁の層 {n0 - len(rows)} を外して {len(rows):,}"
          f"(.clip {sum(r[3] == 'clip' for r in rows):,} / .lip {sum(r[3] == 'lip' for r in rows):,})")
    kind = collections.Counter((r[3], r[4]) for r in rows)
    print("    種別(v_layer_kind)の内訳: " + " / ".join(f"{f} {k} {n:,}" for (f, k), n in kind.most_common()))
    allk = dict(con.execute("""select p.format, count(*) from v_layer_kind k join pages p using (page_path)
                                where k.v_layer_kind = 'image_material' group by 1""").fetchall())
    print(f"    種別 image_material の層(恒久除外の頁を含む): " + " / ".join(f"{f} {n:,}" for f, n in sorted(allk.items()))
          + " —— 種別は has_image_material から作る。text・filter など先に当たる種別に入った層は種別 image_material に入らない")
    po = page_order(con, sorted({r[0] for r in rows}))
    v2 = {(r["page_path"], r["layer_id"]): r["a_rule_v2"] for r in csv.DictReader(open(os.path.join(ROOT, V2), encoding="utf-8"))}
    out = []
    for r in rows:
        d = json.loads(r[8]) if r[8] else {}
        o = po.get((r[0], r[1]))
        out.append(dict(page_path=r[0], layer_id=r[1], format=r[3], name=r[2], v_layer_kind=r[4],
                        ancestors=o[2] if o else "", self_shown_db=int(r[5]), ancestors_shown_db=o[3] if o else "",
                        offset_x=r[6], offset_y=r[7], has_resizable_image_info=int("ResizableImageInfo" in d),
                        material_content_type=d.get("MaterialContentType", ""), has_link_layer=int("LinkLayerInfo" in d),
                        a_rule_v2=v2.get((r[0], str(r[1])), "")))
    print(f"    生のブロックの鍵: ResizableImageInfo あり {sum(x['has_resizable_image_info'] for x in out):,}/{len(out):,}"
          f" / MaterialContentType あり {sum(1 for x in out if x['material_content_type'] != ''):,}"
          f" / LinkLayerInfo あり {sum(x['has_link_layer'] for x in out):,}")
    seen = [x for x in out if x["format"] == "clip" and "見えない" not in x["a_rule_v2"]]
    print(f"    .clip で見えている層 {len(seen):,}(a の理由が 見えない 以外)。祖先の上位 12:")
    for (a, n) in collections.Counter(x["ancestors"] for x in seen).most_common(12):
        print(f"        {a or '(頁の直下)'} {n:,}")
    risky = [x for x in seen if "出力に画素あり" in x["a_rule_v2"]]
    print(f"    うち出力に画素ありの {len(risky)}層の祖先の上位 8: "
          + " / ".join(f"{a or '(頁の直下)'} {n}" for a, n in collections.Counter(x["ancestors"] for x in risky).most_common(8)))
    lip = [x for x in out if x["format"] == "lip"]
    print("    .lip の層の表示: " + " / ".join(f"層 {a} 祖先 {b}: {n}" for (a, b), n in sorted(collections.Counter(
        (x["self_shown_db"], x["ancestors_shown_db"]) for x in lip).items())))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"    → `{OUT}`(非公開、{len(out):,}層)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
