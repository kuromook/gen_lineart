#!/usr/bin/env python
"""**線画の層のアルファが2値(0 か 255 だけ)か** —— Track N の照会(2026-11-02)への答え。

**照会**: 第2段の走行で、線画のタイルの「灰 128〜220 の画素 ÷ 描かれた画素」が
**8作だけ作品の中央値が 0** だった。**その8作の線画の層に、アンチエイリアスの無いペン
(2値の線)で描かれた頁が含まれるか。** 引けないなら、書き出しの段で灰が落ちた可能性を
切り分けたい —— **「同じ頁の原稿の層の画素に灰の中間値があるか」で足りる。**

**この器具が答えるのは後者である**: **層の画素のアルファが 0 か 255 だけの層(2値の層)が、
どの作品にどれだけ在るか。** 材料は**墨の濃さの1段目・2段目**(抽出側が層の本体を描いて数えた。
`n_px_alpha_gt0` と `n_px_alpha_255` が等しければ、描かれた画素は全部アルファ 255)。

**届く範囲**(名前より狭く書く。T132):

- **届く**: 層のアルファが2値か / その層の描かれた画素の量 / 頁ごとに2値の層を持つか
- **届かない**:
  - **ペンの設定そのもの**(アンチエイリアスの有無)—— **ラスタの層にブラシの参照は残らない**
    (抽出側、2026-11-02 のパターンブラシの答え)。**だから「2値のペンで描いた」とは書かない。
    「アルファが2値の層」と書く**
  - **書き出したタイルの灰** —— N のタイルは合成して白点を揃えた後の値で、
    **この表は合成前の層の値である。** 灰はアルファの縁だけでなく色そのものからも来る
  - **層が線画か** —— **層別 L1 は名前で選んだ線画の層**(T106)
  - **N の作品名とこちらの作品フォルダーの対応** —— **名前で合わせた**(3つめの区切り)。
    **名前の一致は同定ではない**(条例9)
- **外すもの**(規律9): **形式5 の層**(「調査中」。アルファが全面 255 として読まれるので、
  **2値に見えるのは読みの形である**)/ 描かれた画素が 0 の層

    ./venv/bin/python tools/census/line_binary_alpha.py
"""
import collections
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import ink_darkness  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK_IDS = "data/census_full/ink_darkness_work_ids.csv"
OUT_FULL = "data/census_full/line_binary_alpha_layers_20261102.csv"
SOURCE = "note_n_census_aliased_lines_20261102.md"
# **N が名を挙げた8作**(年の札, 作品フォルダー名)。**名前で合わせる**(条例9)
N_WORKS = {("2017", "copy"), ("2017", "pink"), ("2017", "ako5"), ("2018", "ako6"),
           ("2019", "ben"), ("2019", "fringe3"), ("2019", "succor"), ("2019", "fringe")}


def load_rows():
    rows = list(csv.DictReader(open(os.path.join(ROOT, ink_darkness.STAGE1), encoding="utf-8")))
    for p in sorted(glob.glob(os.path.join(ROOT, ink_darkness.STAGE2_GLOB))):
        rep = ink_darkness.STAGE2_REPLACED.get(os.path.basename(p))
        if rep:
            p = os.path.join(os.path.dirname(p), rep)
        rows += list(csv.DictReader(open(p, encoding="utf-8")))
    return rows


def main() -> int:
    print("  [線画の層のアルファが2値か(Track N の照会)]")
    ids = {r["work_dir"]: r["work_id"] for r in
           csv.DictReader(open(os.path.join(ROOT, WORK_IDS), encoding="utf-8"))}
    drops = collections.Counter()
    agg = collections.defaultdict(collections.Counter)
    pages = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    full = []
    seen = set()
    for r in load_rows():
        if r["stratum"] != "L1":
            continue
        key = (r["page_path"], r["layer_id"])
        if key in seen:                     # **1段目と2段目の両方に在る層は1回だけ**
            drops["1段目と2段目の両方に在る(2回目を数えない)"] += 1
            continue
        seen.add(key)
        if r["pixel_formats"] == "5":
            drops["形式5(調査中。アルファが全面 255 と読まれる)"] += 1
            continue
        gt0, a255 = int(r["n_px_alpha_gt0"] or 0), int(r["n_px_alpha_255"] or 0)
        if gt0 == 0:
            drops["描かれた画素が 0"] += 1
            continue
        wd = r["work_dir"]
        binary = gt0 == a255
        a = agg[(r["year_label"], wd)]
        a["layers"] += 1
        a["binary_layers"] += int(binary)
        a["px"] += gt0
        a["px_binary"] += gt0 if binary else 0
        a["px_partial"] += gt0 - a255
        pg = pages[(r["year_label"], wd)][r["page_path"]]
        pg[0] += 1
        pg[1] += int(binary)
        full.append([r["page_path"], r["layer_id"], r["year_label"], wd,
                     ids.get(wd, ""), "yes" if binary else "no", gt0, gt0 - a255])
    for why, n in sorted(drops.items()):
        print(f"    数えなかった層 {n:,} —— {why}")

    out = []
    for (y, wd), a in sorted(agg.items(), key=lambda kv: (kv[0][0], ids.get(kv[0][1], ""))):
        ps = pages[(y, wd)]
        name = wd.split("\\")[3] if len(wd.split("\\")) > 3 else ""
        in_n = "yes" if (y, name) in N_WORKS else "no"
        out.append([ids.get(wd, "**名札なし**"), y, in_n, a["layers"], a["binary_layers"],
                    f"{a['binary_layers'] / a['layers']:.3f}",
                    f"{a['px_binary'] / a['px']:.4f}", f"{a['px_partial'] / a['px']:.4f}",
                    len(ps), sum(1 for n, b in ps.values() if b > 0),
                    sum(1 for n, b in ps.values() if b == n), ""])
    found = {(r[1], (wd.split("\\")[3] if len(wd.split("\\")) > 3 else ""))
             for r, wd in zip(out, [k[1] for k in sorted(agg, key=lambda k: (k[0], ids.get(k[1], "")))])}
    missing = sorted(N_WORKS - found)
    print(f"    作品フォルダー {len(out)} / **N の8作のうち名前で見つかった {8 - len(missing)}**"
          + (f"(見つからない: {missing})" if missing else ""))
    print(f"    {'作品':>5} {'札':>5} {'N':>3} {'層':>6} {'2値の層':>7} {'比':>6} "
          f"{'画素の比':>8} {'縁の比':>7} {'頁':>4} {'2値を持つ頁':>6} {'全部2値の頁':>6}")
    for r in out:
        if r[2] == "yes" or float(r[5]) >= 0.05:
            print(f"    {r[0]:>5} {r[1]:>5} {r[2]:>3} {r[3]:>6} {r[4]:>7} {r[5]:>6} "
                  f"{r[6]:>8} {r[7]:>7} {r[8]:>4} {r[9]:>6} {r[10]:>6}")
    n_yes = [r for r in out if r[2] == "yes"]
    n_no = [r for r in out if r[2] == "no"]
    print(f"    **2値の層の比が 0.05 以上の作品**: N の8作 {sum(float(r[5]) >= 0.05 for r in n_yes)}"
          f"/{len(n_yes)} / それ以外 {sum(float(r[5]) >= 0.05 for r in n_no)}/{len(n_no)}")
    print(f"    **全部の線画の層が2値の頁**: {sum(r[10] for r in out)} 頁(全作品)")

    rows = out + [[""] * 11 + [n] for n in NOTE]
    n_pub = bc.write_csv(os.path.join(ROOT, "census", "line_binary_alpha.csv"),
                         ["work_id", "year_label", "in_n_query", "n_layers",
                          "n_layers_binary_alpha", "share_layers_binary_alpha",
                          "share_px_on_binary_layers", "share_px_partial_alpha",
                          "n_pages", "n_pages_with_binary_layer", "n_pages_all_binary",
                          "note"], rows)
    with open(os.path.join(ROOT, OUT_FULL), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page_path", "layer_id", "year_label", "work_dir", "work_id",
                    "binary_alpha", "n_px_alpha_gt0", "n_px_partial_alpha"])
        w.writerows(full)
    print(f"    line_binary_alpha.csv {n_pub}行 / 層ごとの全量 {len(full):,}行(`{OUT_FULL}`)")
    return 0


NOTE = [
    "**Track N の照会への答え**(2026-11-02): **線画の層(層別 L1)のうち、描かれた画素が全部アルファ 255 の層"
    "(「アルファが2値の層」)を作品ごとに数えた。** **材料は墨の濃さの1段目・2段目**(抽出側が層の本体を描いて数えた。"
    "札 2019 は 11-02 の `_v2`)。**作品は `ink_darkness_stage*.csv` と同じ名札 `W<n>`**",
    "**「2値のペンで描いた」とは書かない** —— **ラスタの層にペンの設定は残らない**(抽出側 11-02)。"
    "**分かるのは層の画素のアルファが 0 か 255 だけ、ということまで**",
    "**合成前の層の値である** —— **N のタイルは合成して白点を揃えた後の値。** "
    "**灰はアルファの縁だけでなく色そのものからも来る**(この表は色を見ていない)。"
    "**だからこの表は「書き出しの段で灰が落ちたか」の片側(原稿の層に縁が在るか)だけに答える**",
    "**`share_px_partial_alpha` はアルファが 1〜254 の画素 ÷ 描かれた画素**(層の縁。合成すると灰になる画素の上界)",
    "**形式5 の層は外した**(「調査中」。**アルファが全面 255 と読まれるので、2値に見えるのは読みの形**)。"
    "**描かれた画素が 0 の層も外した。** **外した数は器具が印字する**(規律9)",
    "**1ビットの層の穴は効かない** —— **1段目(10-30)は1ビットの直しの前の読み手で、1ビットの層を空として読んだ。** "
    "**影響の一覧(本体 6,587面)と鍵で当てると、層別 L1 に在る1ビットの層は 2層だけで、どちらも N の8作の外**"
    "(2026-11-02 に引いた)。**2値の線の層が「空」として外れた形ではない**",
    "**層別 L1 は名前で選んだ線画の層である**(T106)。**N の作品名との対応は名前で合わせた** —— "
    "**名前の一致は同定ではない**(条例9)",
    "**読み(2026-11-02)**: **N の8作は8つとも、線画の層の 1割〜6割がアルファ2値の層である**"
    "(0.106〜0.597)。**同じ形は8作の外にも在る**(2値の層の比 0.05 以上は、8作の外の 59作品フォルダーのうち 20)。"
    "**全部の線画の層が2値の頁は全作品で 3頁だけ** —— **だから「頁ごと2値のペンで描かれた」では "
    "N の 40〜74% を説明できない。** **頁の中で2値の層と縁の在る層が混ざっている。** "
    "**タイルごとにどちらの層が効いているかは、この表では届かない**(N の側で、タイルの範囲の層を引けば届く)",
    "**同じ数の2行が在る**(札 2019 と 2019b の同じ名前の作品フォルダー)—— **双子を畳んでいない。足さない**",
]

if __name__ == "__main__":
    raise SystemExit(main())
