#!/usr/bin/env python
"""**`下地` と `輪郭線` の名前の総なめ**(Track E の通達 `note_census_e_shitaji_ruled_base_mask_name_sweep_20261126.md`。
作者の裁定 11-26「下地Nはクリップスタジオペイントの３D素材をLT変換したときに作られるベースマスクだ」。**新しい走行ではない** —— 構造 DB の引き直し)。

- `下地`: 構造 DB の全層から、層名に `下地` を含む層(NFKC で畳んだ名前でも探し、**名前は畳まずに**そのまま分けて出す)。E の頼みは「`下地` で始まる」—— 含むが始まらない層が在れば別に印字する
- `輪郭線`: 層名が `輪郭線` で始まる層(同じく NFKC でも)。**E は a を動かさない。R・Q へ回す数**。`sib_shitaji` = 同じ親の下に名前が `下地` で始まる兄弟が在る
- 名前 × 種別 × 形式ごとに: 層 / 頁 / 単位 / 年の札 / a v9 の判定 / 土台(`a_is_clipping_base`)・乗る層の数。フォルダーは別の行(種別で分かれる)
- E の 161(白さの棚卸しから)と合わない分を名前を付けて印字する
出力(非公開。層名を含む): `data/census_full/shitaji_rinkaku_layers_20261126.csv`(層ごと)。E の `data/census/` へ写す。
"""
import collections
import csv
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
OUT = os.path.join(D, "shitaji_rinkaku_layers_20261126.csv")
E161 = {"下地 1": 108, "下地 2": 46, "下地 3": 6, "下地 4": 1}
nf = lambda s: unicodedata.normalize("NFKC", s).replace(" ", "")  # noqa: E731


def main() -> int:
    c = db.connect()
    rows = c.execute("select l.page_path, l.layer_id, coalesce(l.name,''), l.parent_id, v.v_layer_kind, p.format, p.year "
                     "from layers l join v_layer_kind v using (page_path, layer_id) join pages p using (page_path)").fetchall()
    sel = [r for r in rows if "下地" in nf(r[2]) or nf(r[2]).startswith("輪郭線")]
    shitaji_parent = {(r[0], r[3]) for r in rows if nf(r[2]).startswith("下地")}
    un = split_units.units(c)
    keys = {(r[0], str(r[1])) for r in sel}
    av, cb = {}, {}
    for r in csv.DictReader(open(os.path.join(D, "a_verdict_v9_20261124.csv"), encoding="utf-8")):
        if (r["page_path"], r["layer_id"]) in keys:
            av[(r["page_path"], r["layer_id"])] = r["a_verdict_v9"]
    for r in csv.DictReader(open(os.path.join(D, "a_clipping_base_v3_v8_20261123.csv"), encoding="utf-8")):
        if (r["page_path"], r["layer_id"]) in keys:
            cb[(r["page_path"], r["layer_id"])] = r
    out = []
    for p, lid, nm, par, kind, fmt, year in sel:
        k = (p, str(lid))
        fam = "下地" if "下地" in nf(nm) else "輪郭線"
        out.append(dict(family=fam, starts=int(nf(nm).startswith(fam)), page_path=p, layer_id=lid, layer_name=nm, v_layer_kind=kind,
                        format=fmt, year_label=year, split_unit=un[wd(p)].split("\\", 2)[-1],
                        a_verdict_v9=av.get(k, "(a の外)"), a_is_clipping_base=cb.get(k, {}).get("a_is_clipping_base", ""),
                        n_clipped_layers=cb.get(k, {}).get("n_clipped_layers", ""),
                        sib_shitaji=int((p, par) in shitaji_parent) if fam == "輪郭線" else ""))
    for fam in ("下地", "輪郭線"):
        s = [o for o in out if o["family"] == fam]
        print(f"\n[{fam}] 層 {len(s):,}(始まらないが含む {sum(1 for o in s if not o['starts'])})")
        g = collections.defaultdict(list)
        for o in s:
            g[(o["layer_name"], o["v_layer_kind"], o["format"])].append(o)
        for (nm, kind, fmt), xs in sorted(g.items(), key=lambda kv: -len(kv[1])):
            av_c = collections.Counter(x["a_verdict_v9"] for x in xs)
            line = (f"  {nm!r} {kind} {fmt}: 層 {len(xs)} / 頁 {len({x['page_path'] for x in xs})} / 単位 {len({x['split_unit'] for x in xs})} / "
                    f"札 {','.join(sorted({x['year_label'] for x in xs}))} / a " + " ".join(f"{k} {v}" for k, v in av_c.most_common())
                    + f" / 土台 {sum(x['a_is_clipping_base'] == '1' for x in xs)}・乗る {sum(int(x['n_clipped_layers'] or 0) for x in xs)}")
            if fam == "輪郭線":
                line += f" / 兄弟に下地 {sum(x['sib_shitaji'] for x in xs)}"
            if fam == "下地" and kind != "folder":
                line += f" / E の 161 では {E161.get(nm, 0)}"
            print(line)
    sh = [o for o in out if o["family"] == "下地" and o["v_layer_kind"] != "folder"]
    print(f"\nE の 161 と比べて: 総なめの下地(フォルダーを除く) {len(sh)} / 差 {len(sh) - sum(E161.values())}")
    for nm in sorted({o["layer_name"] for o in sh} | set(E161)):
        a, b = sum(1 for o in sh if o["layer_name"] == nm), E161.get(nm, 0)
        if a != b:
            print(f"  {nm!r}: 総なめ {a} / E {b}" + "(" + " / ".join(f"{k} {v}" for k, v in collections.Counter(
                o["format"] for o in sh if o["layer_name"] == nm).items()) + ")")
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"出力: {OUT}({len(out):,}行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
