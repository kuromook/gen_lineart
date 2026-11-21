#!/usr/bin/env python
"""**対 v6 の下絵の側に入らなかった「下絵」名の層**(Track S の問い、2026-11-21、H1 の作者の答えから)。

S の問い: (1) v6 の下絵の側はどの層をどの規則で選んだか、外れた理由を層ごとに / (2) 層が 2〜4 しか無い頁は何か /
(3) それが S の母集団 492頁(QC ok かつ 線画の側 = 清書・下絵の側 = 下絵)にどれだけあるか。

**v6 の下絵の選び方(抽出側の器具 `scripts/extract/extract_clip_dataset.py` を census が読んだ。v6 は v5・v4 のまま)**:
- 候補 = フォルダーでも 3D の描画でもない層のうち、統計の門(`SKETCH`)に入る層か、名前が下絵系
  (`下絵`/`下描き`/`下書き` を含む、または `sketch`/`rough`/`pencil`)で緩めた門(`NAMED_SKETCH`)に入る層
- **選ぶのは1層だけ**: 描いた割合(`drawn_ratio`)が最大の候補(頁ごとの上書き `SKETCH_LAYER_OVERRIDES` を除く)
- **同じ名前の下絵を重ねる `--sketch-merge` は v6 では切ってある**(発注「下絵の選び方は v5 のまま」。複数の層に分かれた下絵は Q の区分)
- **下絵の側は非表示を見ない**(非表示で外すのは線画の側だけ)

**census が層ごとに言えること / 言えないこと**: 統計の門の値(純黒の割合・灰の平均ほか)は census の表に無い。
したがって外れた理由は次の順で、**最初に当たったもの**で数える(規律9):
1. フォルダー(器具は候補にしない)
2. 画素なし(11-14 の画素の表で `n_px_a_gt0` = 0 か、行が無い・描画できなかった)
3. **選ばれた層より描いた画素が多い** → 1層を選ぶ規則では外れないので、**門で外れたか 3D・上書き(推定)**
4. **選ばれた層以下** → **1層だけ選ぶ規則で外れた**と矛盾しない(門で外れた可能性も残る。census では分けられない)

非表示(自分か祖先)は理由ではなく**印**として別の列に書く。名前の判定は器具の関数をそのまま写した(`has_sketch_name`)。
`name_rule`(役割の表 v3 の関数)も並べる。

出力 `data/census_full/s_h1_sketch_not_in_pairs_layers_20261121.csv` / `..._pages_20261121.csv`(非公開)。
"""
import collections
import csv
import ntpath
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
from stage_hearing import name_rule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "census_full")
PAGES = os.path.join(D, "s_v6_pairs_page_name_rule_20261120.csv")
LAYERS = os.path.join(D, "s_v6_pairs_layer_name_rule_20261120.csv")
PIX = os.path.join(D, "full_clip_pixels_layers_merged.csv")
OUT_L = os.path.join(D, "s_h1_sketch_not_in_pairs_layers_20261121.csv")
OUT_P = os.path.join(D, "s_h1_sketch_not_in_pairs_pages_20261121.csv")
H1 = {  # S の H1 の頁(work_id, ファイル名)
    "h03": ("017_2017_stoptime", "page0007.clip"), "h04": ("021_2018_comics4", "page0006.clip"),
    "h06": ("020_2018_china2", "page0004.clip"), "h10": ("001_2015_gakusai3", "page0035.clip"),
    "h11": ("016_2017_running", "page0032.clip"), "h12": ("002_2016_aljanne2", "page0002.clip"),
    "h14": ("018_2018_ako6", "ako6-56.clip"), "h15": ("003_2016_avbu", "page0028.clip"),
    "h18": ("069_2024b_fitness", "page0030.clip"), "h19": ("006_2016_pc", "page0013.clip"),
    "h20": ("017_2017_stoptime", "page0004.clip"),
}
FEW = [("063_2023b_hamlabi", "page0007.clip"), ("056_2022b_hyoui_color", "page0026.clip"),
       ("056_2022b_hyoui_color", "page0044.clip"), ("048_2022_akocult", "page0017.clip"),
       ("032_2019b_ako8", "page0045.clip")]


def has_sketch_name(name):  # 抽出側の器具の関数をそのまま
    n = name.strip().casefold()
    return ("下絵" in name or "下描き" in name or "下書き" in name or n in {"sketch", "rough", "pencil"})


def main() -> int:
    pages = [r for r in csv.DictReader(open(PAGES, encoding="utf-8"))
             if r["pair_quality"] == "ok" and r["line_rules"] == "清書" and r["sketch_rules"] == "下絵"]
    pset = {r["page_path"]: r["work_id"] for r in pages}
    print(f"S の母集団(QC ok・線画の側 清書・下絵の側 下絵): {len(pset)} 頁")
    used = collections.defaultdict(set)
    for r in csv.DictReader(open(LAYERS, encoding="utf-8")):
        if r["side"] == "sketch" and r["page_path"] in pset:
            used[r["page_path"]].add(int(r["layer_id"]))
    c = db.connect()
    q = ",".join("?" * len(pset))
    lay = collections.defaultdict(list)
    for p, i, n, fo, k in c.execute(
            f"select l.page_path, l.layer_id, l.name, l.folder, v.v_layer_kind from layers l "
            f"join v_layer_kind v using (page_path, layer_id) where l.page_path in ({q})", list(pset)).fetchall():
        lay[p].append((i, n or "", fo, k))
    pix = {}
    for r in csv.DictReader(open(PIX, encoding="utf-8")):
        if r["page_path"] in pset:
            pix[(r["page_path"], int(r["layer_id"]))] = r
    out, prow = [], []
    reason_n = collections.Counter()
    reason_pages = collections.defaultdict(set)
    for p, w in pset.items():
        u = used[p]
        upx = max((int(pix[(p, i)]["n_px_a_gt0"] or 0) for i in u if (p, i) in pix), default=0)
        n_unused = 0
        for i, n, fo, k in lay[p]:
            if i in u or not has_sketch_name(n):
                continue
            r = pix.get((p, i))
            gt0 = int(r["n_px_a_gt0"] or 0) if r and r["n_px_a_gt0"] not in ("", None) else 0
            if fo or k in ("folder", "frame_folder"):
                why = "1 フォルダー(候補にしない)"
            elif r is None or r["render_status"] != "描いた" or gt0 == 0:
                why = "2 画素なし(描かれていない・描画できなかった・行なし)"
            elif gt0 > upx:
                why = "3 選ばれた層より画素が多い(門か 3D・上書きで外れた、推定)"
            else:
                why = "4 選ばれた層以下(1層だけ選ぶ規則と矛盾しない)"
            hidden = "" if r is None else int(not (r["self_shown"] == "1" and r["ancestors_shown"] == "1"))
            n_unused += why[0] in "34"
            reason_n[why] += 1
            reason_pages[why].add(p)
            out.append(dict(page_path=p, work_id=w, layer_id=i, layer_name=n, name_rule=name_rule(n),
                            v_layer_kind=k, n_px_a_gt0=gt0, chosen_n_px_a_gt0=upx, hidden_self_or_ancestor=hidden,
                            reason=why))
        nonfolder = sum(1 for i, n, fo, k in lay[p] if not fo and k not in ("folder", "frame_folder"))
        prow.append(dict(page_path=p, work_id=w, n_layers=len(lay[p]), n_layers_not_folder=nonfolder,
                         sketch_ids_used=" ".join(map(str, sorted(u))), n_sketch_named_unused_with_pixels=n_unused))
    print("\n[1] 下絵名の層で対の下絵の側に入らなかった層(理由は上から最初に当たったもの):")
    for k in sorted(reason_n):
        print(f"  {reason_n[k]:6,d} 層 / {len(reason_pages[k]):4d} 頁  {k}")
    hid = collections.Counter((x["reason"][0], x["hidden_self_or_ancestor"]) for x in out)
    print("  うち非表示(自分か祖先)の印: " + " / ".join(f"理由{a} 非表示={b} {v}" for (a, b), v in sorted(hid.items())))
    nr = collections.Counter(x["name_rule"] for x in out if x["reason"][0] in "34")
    print("  画素がある層(理由3・4)の name_rule: " + " / ".join(f"{k} {v}" for k, v in nr.most_common()))
    np_ = sum(1 for x in prow if x["n_sketch_named_unused_with_pixels"] > 0)
    print(f"  画素のある下絵名の層が対に入らなかった頁: {np_} / {len(prow)}")
    print("\n[2] 層の数(DB の全層 / フォルダーでない層)")
    dist = collections.Counter(min(x["n_layers"], 10) for x in prow)
    print("  全層の数の分布(10 は10以上): " + " / ".join(f"{k}:{v}" for k, v in sorted(dist.items())))
    few = [x for x in prow if x["n_layers"] <= 4]
    print(f"  全層 4 以下の頁: {len(few)} / {len(prow)}(作品フォルダー {len({x['work_id'] for x in few})})")
    print("\n[H1 の頁](S の selection は work_id と頁のファイル名だけ。同じ組が2頁あれば両方出す)")
    byname = collections.defaultdict(list)
    for x in prow:
        byname[(x["work_id"], ntpath.basename(x["page_path"]))].append(x)
    for h, k in list(H1.items()) + [("層の少ない頁", k) for k in FEW]:
        xs = byname.get(k)
        if not xs:
            print(f"  {h} {k[1]}: S の母集団に無い"); continue
        for x in xs:
            rs = collections.Counter(y["reason"][:1] for y in out if y["page_path"] == x["page_path"])
            kinds = collections.Counter(kk for i, n, fo, kk in lay[x["page_path"]])
            print(f"  {h} {k[1]}{' (同じ組が ' + str(len(xs)) + ' 頁)' if len(xs) > 1 else ''}: 全層 {x['n_layers']}"
                  f"(フォルダーでない {x['n_layers_not_folder']})/ 使った下絵 id {x['sketch_ids_used']} / "
                  f"外れた下絵名の層 理由別 {dict(sorted(rs.items()))} / 種別 {dict(kinds.most_common(4))}")
    for path, rows in ((OUT_L, out), (OUT_P, prow)):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"\n出力: {OUT_L}\n      {OUT_P}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
