#!/usr/bin/env python
"""**読み手の2つ目の不具合(zlib の頭の取り違え)が、この集計のどの材料に当たるか。**

**一声は抽出側から来た**(2026-11-01、`deliver_occ_full_v2_mono1bit_20261101_addendum.md` §3
と `note_reader_zlib_fix_schema_20261101.md`):

> **`.clip` の画素ブロックの zlib は 74バイト目から始まる。読み手は 46バイト目から
> `78 01` を探していた。手前の 70バイト目に在る圧縮後の長さが 376(= 0x0178)などだと、
> そこが `78 01` になって頭と取り違える。取り違えたブロックは、読み手が黙って空として読む。**
> **census へ: この直しで墨の濃さ2段目の値が動く可能性があります。**

**抽出側の見立ては「0.2% の層で 1〜数ブロックなので、全体の値はほとんど動かない。
これは推定です」だった。** **この器具は、その推定を<ins>この集計の母集団で</ins>数に替える**
(条例5: 供給元から取った数は、ここで誰かが測るまで測った者の名を帯びる)。

**器具の破損の直しは棒の外**だが(`CLAUDE.md` §3)、
**前後の数を両方、名前を付けて印字する**のが要求である。
**この表は「前」の側** —— **どの材料の何層が、直す前の読み手で画素を失っていたか。**

**この器具が届く範囲**(名前より狭く書く。T132):

- **届く**: 影響表の (頁, 層, 面) が、この集計の材料のどの行に当たるか ——
  **結合は `page_path` + `layer_id`。名前では結合しない**
- **届かない**: **値がどれだけ動くか。** 失ったブロックの画素が**墨だったのか白だったのか
  分からない**(ブロックは読めなかったので、中身は直した読み手でしか分からない)。
  **上界しか書けない** —— 1ブロックは 256×256 = 65,536画素
- **届かない**: **マスクの取り違えが本体の測りに効くか。** **抽出側へ照会中** ——
  2段目の列(`n_px_ink` ほか)が層の本体だけを見ているのか、
  マスクを当てた後を見ているのかを census は知らない。**だから両方を別の列で数える**

    ./venv/bin/python tools/census/reader_defect.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AFFECTED = "inbox/zlib_affected_layers_20261101.csv"
SCAN_PAGES = "inbox/zlib_scan_pages_20261101.csv"
SOURCE = "deliver_zlib_affected_20261101.md"
BLOCK_PX = 256 * 256          # 1ブロックの画素。**上界の計算にだけ使う**

# **この集計が画素の読み手を通った材料として宣言するもの。**
# **宣言していない材料は、この表に出ない** —— 門を名簿で持つ(`delivery.py` と同じ形)。
# (名前, 便, 公開表に効くか, 述語, 絞り)
# **名前の白名簿は `build_census.READER_DEFECT_MATERIALS`** —— 走行の最初に突き合わせる
MATERIALS = (
    ("墨の濃さ2段目(主結論の母集団)",
     "inbox/ink_darkness_stage2_*_20261031.csv", "yes",
     "**層別 L1 かつ `pixel_formats = 2` の 40,323層** —— "
     "**主結論 0.9177 を引いた母集団そのもの**", ("L1", "2")),
    ("墨の濃さ2段目(全部)",
     "inbox/ink_darkness_stage2_*_20261031.csv", "yes",
     "6札の線画・用紙の層 48,465層・2,443頁(形式5 の 1,919 と 2,399 を含む)", None),
    ("墨の濃さ1段目",
     "inbox/ink_darkness_stage1_20261030.csv", "yes",
     "6フォルダーの 4,063層(L1 3,773 / C1 288 / C2 2)", None),
    ("依頼3の層の種別",
     "inbox/f_request3_layer_kind_rendered_20261026.csv", "yes",
     "872頁のフォルダー以外の全層 56,576層 —— "
     "**`layers_by_year_kind_type.csv` の `drawn_ratio` 由来の列がこれに乗る**", None),
    ("薄いアルファの帯",
     "inbox/faint_alpha_905_20261027.csv", "yes",
     "905頁の 62,105層(`faint_alpha_gap.csv`)", None),
    ("マスクの被覆",
     "inbox/mask_cover_request3_20261026.csv", "yes",
     "766頁の 16,273層(`mask_cover_by_kind.csv` / `mask_cover_bound.csv`)", None),
    ("マスクの被覆(原点を直した55頁)",
     "inbox/mask_cover_request3_maskorigin55_20261030.csv", "yes",
     "55頁の 1,501層", None),
    ("本体の InitColor",
     "inbox/body_initcolor_delivered_20261026.csv", "yes",
     "トーン・グラデ・ベタの 2,095層(`body_initcolor.csv`)", None),
    ("グラデーションの版の対",
     "inbox/grad_v1_v2_layer_map_20261026.csv", "yes",
     "65頁の 258層(`grad_v1_v2_by_layer.csv`)", None),
    ("調査中の候補",
     "inbox/under_investigation_candidates_20261029_v2.csv", "no",
     "294頁の 583層 —— **分析対象外の層の一覧**(作者の決定 2026-10-29)", None),
    ("1ビットの影響一覧",
     "inbox/mono_1bit_affected_layers_20261031.csv", "no",
     "2,777頁の 6,506層 —— **先の不具合の一覧。この表の材料ではない**", None),
    ("`_v2` でまだ空の28層",
     "inbox/occ_v2_still_unread_mono_20261101.csv", "no",
     "27頁の 28層 —— **抽出側が「全部この取り違えだった」と書いた層。"
     "こちらで引き直す**", None),
    ("構造の全層",
     "inbox/census_r8_layer_structure_20261027.csv", "no",
     "9,649頁の 437,561層 —— **画素を1つも持たない構造の表。"
     "この表の列はこの不具合では動かない**", None),
    ("`.lip` の層",
     "inbox/census_n4_lip_layers_20261027.csv", "no",
     "1,986頁の 124,663層 —— **`.lip` には画素の読み手が無い**ので、"
     "この不具合は当たらない(0 であることが読みの裏付けになる)", None),
)

NOTE = [
    "**この表は「直す前」の側である。** **器具の破損の直しは棒の外**だが"
    "(`CLAUDE.md` §3)、**前後の数を両方、名前を付けて印字する**のが要求 —— "
    "**直した読み手で作り直した材料が届いたら、同じ列で「後」を並べる**",
    "**失った画素の数は書けない。書けるのは上界である** —— "
    "**ブロックは読めなかったので、中身が墨だったか白だったかは、"
    "直した読み手でしか分からない。** 1ブロックは 256×256 = 65,536画素",
    "**本体とマスクを別の列で数えた。** **2段目の列が層の本体だけを見ているのか、"
    "マスクを当てた後を見ているのかを census は知らない**(抽出側へ照会中)—— "
    "**分からないものを1つの列に足さない**",
    "**抽出側の見立ては「0.2% の層で 1〜数ブロックなので、全体の値はほとんど動かない。"
    "これは推定です」だった。** **この表はその推定を、この集計の母集団で数に替えたもの**"
    "(条例5)—— **一致しても、言明だけでは一致ではない**",
    "**宣言していない材料はこの表に出ない。** 名簿は "
    "`tools/census/reader_defect.py` の `MATERIALS` —— "
    "**歩行器に門が無いと、黙って入る**(2026-11-01 に納品の木で踏んだ形)",
]


def affected():
    """影響表を読む。**面(本体 / マスク)で分けて持つ。**"""
    path = os.path.join(ROOT, AFFECTED)
    body, mask = {}, {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["page_path"], r["layer_id"])
            (body if r["plane"] == "body" else mask)[key] = int(r["n_wrong"])
    pages = {p for p, _ in body} | {p for p, _ in mask}
    print(f"    影響表: 本体 {len(body)} 層 / マスク {len(mask)} 層"
          f" / 頁 {len(pages)}(抽出側の全数走査、2026-11-01)")
    return body, mask, pages


def supplier_totals():
    """**供給元の頁の表から、こちらで合計を引き直す**(条例5)。"""
    path = os.path.join(ROOT, SCAN_PAGES)
    n_pages = n_wrong = n_hit = 0
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n_pages += 1
            n_wrong += int(r["n_wrong"])
            n_hit += 1 if int(r["n_wrong"]) else 0
    print(f"    供給元の頁の表を引き直した: {n_pages} 頁 / "
          f"取り違えたブロック {n_wrong} / 当たった頁 {n_hit}")
    return n_pages, n_wrong, n_hit


def rows_of(glob_or_path, pick):
    """材料を読む。**層別と形式で絞る指定が在れば、そこで絞る**(母集団を書く)。"""
    import glob as _g
    paths = sorted(_g.glob(os.path.join(ROOT, glob_or_path)))
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if pick and (r.get("stratum"), r.get("pixel_formats")) != pick:
                    continue
                out.append((r.get("page_path", ""), r.get("layer_id", "")))
    return out, paths


def main() -> int:
    print("  [読み手の2つ目の不具合(zlib の頭の取り違え)の当たり]")
    # **名簿が2か所に在るなら、片方が古くなる。** 走行の最初に突き合わせる
    declared, here = set(bc.READER_DEFECT_MATERIALS), {m[0] for m in MATERIALS}
    if declared != here:
        print(f"    **名簿が食い違う** —— 点検側だけ {sorted(declared - here)} /"
              f" 器具側だけ {sorted(here - declared)}")
        return 1
    body, mask, aff_pages = affected()
    supplier_totals()
    rows = []
    print(f"    {'材料':34} {'層':>8} {'頁':>6} {'本体':>5} {'マスク':>6}"
          f" {'ブロック':>8} {'影響頁':>6}")
    for name, path, feeds, predicate, pick in MATERIALS:
        keys, files = rows_of(path, pick)
        if not files:
            print(f"    **材料が無い**: {path}")
            return 1
        pages = {p for p, _ in keys}
        nb = sum(1 for k in keys if k in body)
        nm = sum(1 for k in keys if k in mask)
        bb = sum(body[k] for k in keys if k in body)
        bm = sum(mask[k] for k in keys if k in mask)
        npg = len(pages & aff_pages)
        print(f"    {name[:34]:34} {len(keys):>8} {len(pages):>6} {nb:>5} {nm:>6}"
              f" {bb + bm:>8} {npg:>6}")
        rows.append([name, feeds, len(keys), len(pages), nb, nm, bb, bm, npg,
                     (bb + bm) * BLOCK_PX, predicate,
                     "census track(抽出側の影響表をこの集計の材料に当てた)",
                     "2026-11-01", SOURCE, ""])
    # **既知セル**: 抽出側が「28層は全部この取り違えだった」と書いた行を引き直す
    v2 = [r for r in rows if r[0].startswith("`_v2`")][0]
    ok = v2[4] == v2[2]
    print(f"    **既知セル**: `_v2` でまだ空の 28層のうち、本体に取り違えが在るのは"
          f" {v2[4]} 層 —— {'**全部。抽出側の申告と一致**' if ok else '**食い違い**'}")
    rows += [["", "", "", "", "", "", "", "", "", "", "", "", "", "", n]
             for n in NOTE]
    header = ["material", "feeds_public_table", "n_layers", "n_pages",
              "n_layers_body_wrong", "n_layers_mask_wrong",
              "n_blocks_wrong_body", "n_blocks_wrong_mask", "n_pages_in_defect",
              "n_px_bound", "predicate", "by", "as_of", "source", "note"]
    n = bc.write_csv(os.path.join(ROOT, "census", "reader_defect.csv"), header, rows)
    print(f"    reader_defect.csv {n}行")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
