#!/usr/bin/env python
"""**`is_base_mask` —— ベースマスク(それ自体に絵を持たない層)の列**(作者の裁定、2026-11-23。名前で引く)。

作者の言明(逐語、2026-11-23、E 経由): 「ベースマスクであるかどうかのカラムはcensusのDBに必要だと思える　文脈によって消し、文脈によって使う、しかもベースマスクはそれ自体が絵をもたない」
作者の答え(census の問い、2026-11-23): どの頁でもベースマスクと言える名前 = 「rendering_base の3つ, bg base, base」/ 心理トーンのベースマスクの名前 = 「bg base と同じ名前」

- 述語: 層名が**ちょうど** `NAMES` のどれか、かつフォルダーでない(フォルダーは中身の合成なので入れない。数える)
- 近い名前(`bg base 2` / `rendering_base0001` など)は作者が名を挙げていないので入れない(数える)
- `a_is_clipping_base` とは別の列(構造の札ではなく層の種類の札)。重なりを印字する
- 作者の過去のラベル(役割の表 v3 の聞き取り)と食い違う層を印字する(裁定は消さない。両方残す)
- 母集団 = 構造 DB の全層 437,561(a の母集団 437,425 の外の 136層も行に持つ。列 `in_a_population`)

出力 `data/census_full/is_base_mask_20261123.csv`(全層。非公開)。
"""
import collections
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
OUT = os.path.join(D, "is_base_mask_20261123.csv")
AV7 = os.path.join(D, "a_verdict_v7_20261116.csv")
CB = os.path.join(D, "a_clipping_base_v2_20261123.csv")
ROLE = os.path.join(D, "layer_role_table_20261112_v3.csv")
NAMES = ["rendering_base_back0001", "rendering_base_middle0001", "rendering_base_front0001", "bg base", "base"]
NEAR = re.compile(r"^(bg base|rendering_base|base)")


def main() -> int:
    c = db.connect()
    rows = c.execute("select l.page_path, l.layer_id, coalesce(l.name,''), l.folder, v.v_layer_kind, p.format from layers l "
                     "join v_layer_kind v using (page_path, layer_id) join pages p using (page_path)").fetchall()
    av = {(r["page_path"], int(r["layer_id"])): r["a_verdict_v7"] for r in csv.DictReader(open(AV7, encoding="utf-8"))}
    cb = {(r["page_path"], int(r["layer_id"])): r["a_is_clipping_base"] for r in csv.DictReader(open(CB, encoding="utf-8"))}
    lab = {(r["page_path"], int(r["layer_id"])): r["author_stage"] for r in csv.DictReader(open(ROLE, encoding="utf-8")) if r["author_stage"]}
    out, drop, near = [], collections.Counter(), collections.Counter()
    for p, i, n, fo, k, f in rows:
        hit = n in NAMES
        if hit and (fo or k in ("folder", "frame_folder")):
            drop[f"名前は当たるがフォルダー({n})"] += 1; hit = False
        if not hit and n not in NAMES and NEAR.match(n):
            near[n] += 1
        out.append(dict(page_path=p, layer_id=i, is_base_mask=int(hit), in_a_population=int((p, i) in av)))
    print(f"構造 DB の層 {len(rows):,} / **is_base_mask = 1: {sum(o['is_base_mask'] for o in out):,}**")
    by = collections.Counter((n, k, f) for (p, i, n, fo, k, f), o in zip(rows, out) if o["is_base_mask"])
    for n in NAMES:
        print(f"  {n}: " + " / ".join(f"{k}・{f} {v:,}" for (nn, k, f), v in sorted(by.items()) if nn == n))
    print("入れなかった: " + " / ".join(f"{k} {v}" for k, v in drop.items()))
    print("近い名前(入れない): " + " / ".join(f"{k} {v}" for k, v in near.most_common(12)) + (f" ほか {len(near) - 12}" if len(near) > 12 else ""))
    hitk = [(r[0], r[1]) for r, o in zip(rows, out) if o["is_base_mask"]]
    print(f"a の母集団の外: {sum(k not in av for k in hitk)}")
    print("a v7 の判定: " + " / ".join(f"{k} {v:,}" for k, v in collections.Counter(av.get(k, '外') for k in hitk).most_common()))
    print(f"a_is_clipping_base との重なり: 両方 {sum(cb.get(k) == '1' for k in hitk):,} / ベースマスクだが土台でない {sum(cb.get(k) == '0' for k in hitk):,}"
          f" / 土台だがベースマスクでない {sum(1 for k, v in cb.items() if v == '1') - sum(cb.get(k) == '1' for k in hitk):,}")
    conf = collections.Counter(lab[k] for k in hitk if k in lab)
    print("作者の過去のラベル(聞き取り 265層)のある層: " + " / ".join(f"{k} {v}" for k, v in conf.items()) + "(段ではない 以外は裁定と食い違いうる。両方残す)")
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
    print(f"出力: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
