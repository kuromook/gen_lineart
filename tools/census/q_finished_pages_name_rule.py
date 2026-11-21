#!/usr/bin/env python
"""**コマ枠フォルダーの無い頁ごとの、名前の決まり v3 の段の層の数**(Track Q の問い、2026-11-22。新しい列ではない)。

- 頁 = 枠らしさの一覧 `q_frame_likeness_order_20261120.csv` の `main` の頁(4,531)
- 層 = 構造 DB の `layers` の、その頁の**フォルダーでもテキストでもない層**(フォルダーは中身の合成なので別に数える)
- 段 = `stage_hearing.name_rule`(役割の表 v3 と同じ、名前だけの関数): 清書 / 下絵(あたりを含む)/ 両方当たる / なし
- **「線画」= `name_rule` が 清書 の層**。両方当たる は線画に入れず別の列(名前に線画系と下絵系の両方)
- 画素のある層だけの数も並べる(11-14 の画素の表で `n_px_a_gt0` > 0。表に行が無い層は「画素不明」)
- 落としたもの(フォルダー・テキスト)は数えて印字する(規律9)

出力 `data/census_full/q_finished_pages_name_rule_20261122.csv`(非公開)。
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
from stage_hearing import name_rule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
ORDER = os.path.join(ROOT, "outbox", "q_frame_likeness_order_20261120.csv")
PIX = os.path.join(D, "full_clip_pixels_layers_merged.csv")
OUT = os.path.join(D, "q_finished_pages_name_rule_20261122.csv")
CHECK = ["2019\\ben\\ben\\page0019.clip", "2019\\plecia\\plecia\\work.clip", "2019b\\marisa\\marisa_work\\page0005.clip",
         "2019b\\tanu\\tanu\\page0031.clip", "2019b\\yakuza3\\yakuza3\\page0017.clip", "2020\\CD\\コミック.clip",
         "2020\\marisa_color\\marisa.clip", "2020\\yakuza4\\yakuza4\\page0025.clip",
         "2020b\\akokate\\akokatei\\page0024.clip", "2021\\sengoku\\tetsu\\page0022.clip"]
STAGES = ["清書", "下絵", "両方当たる", "なし"]


def main() -> int:
    meta = {}
    for r in csv.DictReader(open(ORDER, encoding="utf-8")):
        if r["set"] == "main":
            meta[r["page_path"]] = (r["year_label"], r["work_dir"])
    print(f"枠らしさの一覧の main の頁: {len(meta):,}")
    pix = {}
    for r in csv.DictReader(open(PIX, encoding="utf-8")):
        if r["page_path"] in meta:
            pix[(r["page_path"], int(r["layer_id"]))] = int(r["n_px_a_gt0"] or 0)
    c = db.connect()
    q = ",".join("?" * len(meta))
    cnt = {p: collections.Counter() for p in meta}
    drop = collections.Counter()
    for p, i, n, fo, k in c.execute(
            f"select l.page_path, l.layer_id, coalesce(l.name,''), l.folder, v.v_layer_kind from layers l "
            f"join v_layer_kind v using (page_path, layer_id) where l.page_path in ({q})", list(meta)).fetchall():
        if fo or k in ("folder", "frame_folder"):
            drop["フォルダー"] += 1; cnt[p]["n_folder"] += 1; continue
        if k == "text":
            drop["テキスト"] += 1; continue
        s = name_rule(n)
        cnt[p][s] += 1
        px = pix.get((p, i))
        cnt[p][s + "_px" if px else s + "_px不明" if px is None else s + "_空"] += 1
    print("数えなかった層: " + " / ".join(f"{k} {v:,}" for k, v in drop.items()))
    rows = []
    for p, (y, w) in meta.items():
        x = cnt[p]
        rows.append(dict(page_path=p, year_label=y, work_dir=w,
                         **{f"n_{s}": x[s] for s in STAGES},
                         **{f"n_{s}_with_px": x[s + "_px"] for s in STAGES},
                         n_px_unknown=sum(x[s + "_px不明"] for s in STAGES), n_folder=x["n_folder"]))
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print("\n札ごと: 頁 / 線画(清書)の層が1つ以上 / うち画素のある清書の層が1つ以上 / 両方当たるだけ在る頁")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["year_label"]].append(r)
    tot = collections.Counter()
    for y in sorted(by):
        rs = by[y]
        a = sum(r["n_清書"] > 0 for r in rs); b = sum(r["n_清書_with_px"] > 0 for r in rs)
        o = sum(r["n_清書"] == 0 and r["n_両方当たる"] > 0 for r in rs)
        tot.update(dict(n=len(rs), a=a, b=b, o=o))
        print(f"  {y:8s} {len(rs):5d} / {a:5d} / {b:5d} / {o:4d}")
    print(f"  {'計':8s} {tot['n']:5d} / {tot['a']:5d} / {tot['b']:5d} / {tot['o']:4d}")
    print("\n確かめ(Q が目視した10頁。IC2 は「10頁とも線画の層 0」):")
    idx = {r["page_path"]: r for r in rows}
    for s in CHECK:
        p = "F:\\manuscripts\\" + s
        r = idx.get(p)
        if r is None:
            print(f"  {s}: main の頁に無い"); continue
        print(f"  {s}: 清書 {r['n_清書']}(画素あり {r['n_清書_with_px']})/ 下絵 {r['n_下絵']} / 両方 {r['n_両方当たる']} / なし {r['n_なし']}")
    print(f"\n出力: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
