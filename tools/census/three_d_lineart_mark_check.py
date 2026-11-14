#!/usr/bin/env python
"""**3D 出力の線画の印 —— 名前の側と形式の側の的中率と取りこぼし**(Track R の頼み、規律13)。

作者(2026-11-13): 3D 出力の線画は人が描いた線画とも他とも混ざらない種別にしたい。経路は CSP の LT 変換と Blender のレンダリング。

**数を見る前に決めたこと**:
- 正(作者が3D と答えた層): R の一覧 `r_3d_lineart_rows_author_20261113.csv` の10行(異なる層 9)
- 負(作者が答えて、3D とは書いていない層): 便A の答え(97)・便B の答え(作者が答えた行)・12頁265層の段のラベル。
  正の層は負から外す。**負は「作者が3D と言わなかった層」で、「3D ではないと言った層」ではない**(コメントに書かなかっただけかもしれない)
- 印は2つ、どちらも前から在るもの:
  - 名前の側 = 構造 DB の `v_layer_name_marks.name_3d`(`3d|rendering|レンダ|モデル`。供給元が先に定めた印。ここで作った規則ではない)
  - 形式の側 = 全頁の画素の `pixel_formats` に `5` を含む(10-31 の census の記録: 形式5 の層はほぼ Blender の層)
  - 両方(AND)と、どちらか(OR)も印字する
- 層ごとに、正の全行を名前・形式つきで印字する(非公開の出力にだけ。公開の表には出さない)

    ./venv/bin/python tools/census/three_d_lineart_mark_check.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R3D = "data/census_full/r_3d_lineart_rows_author_20261113.csv"
PIX = "data/census_full/full_clip_pixels_layers_merged.csv"
A = "data/census_full/hearing_round2_a_answers.csv"
B = "data/census_full/hearing_round2_b_answers.csv"
V3 = "data/census_full/layer_role_table_20261112_v3.csv"
OUT = "data/census_full/three_d_lineart_mark_check_20261114.csv"


def rd(p):
    return list(csv.DictReader(open(os.path.join(ROOT, p), encoding="utf-8")))


def main() -> int:
    print("[3D 出力の線画の印: 名前の側と形式の側]")
    pos = {(r["page_path"], r["layer_id"]) for r in rd(R3D)}
    neg = set()
    neg |= {(r["page_path"], r["layer_id"]) for r in rd(A)}
    neg |= {(r["page_path"], r["layer_id"]) for r in rd(B) if r["answer_source"] == "author"}
    neg |= {(r["page_path"], r["layer_id"]) for r in rd(V3) if r["author_stage"]}
    neg -= pos
    pix = {(r["page_path"], r["layer_id"]): r for r in rd(PIX)}
    con = db.connect()
    marks = {(p, str(i)): (n, bool(m)) for p, i, n, m in con.execute(
        "select l.page_path, l.layer_id, l.name, v.name_3d from layers l join v_layer_name_marks v using (page_path, layer_id)").fetchall()}
    rows = []
    for k in sorted(pos | neg):
        name, n3 = marks[k]
        pf = pix[k]["pixel_formats"] if k in pix else None
        f5 = None if pf is None else ("5" in pf.split(";"))
        rows.append(dict(page_path=k[0], layer_id=k[1], name=name, author_3d=int(k in pos), name_3d=int(n3),
                         pixel_formats="(.clip の画素なし)" if pf is None else pf, format5="" if f5 is None else int(f5)))
    lost = [r for r in rows if r["format5"] == ""]
    print(f"    母集団: 作者が答えた層 {len(rows)}(正 = 3D {len(pos)} / 負 = 3D と書いていない {len(neg)})。"
          f"形式が無い層(.lip など){len(lost)}(正 {sum(r['author_3d'] for r in lost)})は形式の側の母数から外す")
    rules = {"名前(name_3d)": lambda r: r["name_3d"] == 1,
             "形式5": lambda r: r["format5"] == 1,
             "名前 AND 形式5": lambda r: r["name_3d"] == 1 and r["format5"] == 1,
             "名前 OR 形式5": lambda r: r["name_3d"] == 1 or r["format5"] == 1}
    use = [r for r in rows if r["format5"] != ""]
    for lab, f in rules.items():
        tp = sum(1 for r in use if r["author_3d"] and f(r))
        fn = sum(1 for r in use if r["author_3d"] and not f(r))
        fp = sum(1 for r in use if not r["author_3d"] and f(r))
        npos = tp + fn
        print(f"    {lab}: 的中 {tp}/{npos}(取りこぼし {fn})/ 負に当たった {fp}/{len(use) - npos}"
              + (f" / 当たった中の正の率 {tp}/{tp + fp}" if tp + fp else ""))
    print("    正の層(非公開):")
    for r in rows:
        if r["author_3d"]:
            print(f"        layer_id {r['layer_id']}: 名前 {r['name']} / name_3d {r['name_3d']} / 形式 {r['pixel_formats']}")
    fpn = sorted({(r["name"], r["pixel_formats"]) for r in use if not r["author_3d"] and r["format5"] == 1})
    print(f"    形式5 に当たった負の層の (名前, 形式) {len(fpn)}種(上位 12): " + " / ".join(f"{n}[{p}]" for n, p in fpn[:12]))
    # **見た後の観察**(数を見た後で思いついた名前の形。規則ではない。使うなら R が先に登録してから、別の層で確かめる)
    pre = lambda r: r["name"].lower().startswith(("lineart", "rendering"))  # noqa: E731
    print(f"    見た後の観察(規則ではない): 名前が lineart / rendering で始まる層 —— 正 {sum(1 for r in use if r['author_3d'] and pre(r))}/{len(pos)}"
          f" / 負 {sum(1 for r in use if not r['author_3d'] and pre(r))}/{len(use) - len(pos)}")
    names = {(p, str(i)): (n or "") for p, i, n in con.execute("select page_path, layer_id, name from layers").fetchall()}
    allc = {True: 0, False: 0}
    for k, r in pix.items():
        if names[k].lower().startswith(("lineart", "rendering")):
            allc["5" in r["pixel_formats"].split(";")] += 1
    print(f"        全 .clip(312,762層)でその名前の層: 形式5 あり {allc[True]} / なし {allc[False]}(作者の答えは付いていない)")
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"    → `{OUT}`(非公開、{len(rows)}層)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
