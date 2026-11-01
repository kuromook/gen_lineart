#!/usr/bin/env python
"""**マスクで形を持つ層の枚数。** Track M の原案の材料を数える。

**発注は Track M**(2026-11-01、`note_m_order_mask_counts_for_candidates_20261101.md`)。
**作者の承認が付いている**(逐語):

> **見ればカラーかはわかるのでこれから見て選ぶ、
> ただマスクの枚数はあってもよさそうなので発注はしよう**

**なぜ要るのか**(Track M の申告): 原案は作者の言葉で
**「塗りの入っていないマスク層だけで面を見つける」**。
その材料は**層自身の画素ではなくレイヤーマスクの側に在る**ので、
**`has_mask` の枚数が分からないと「また材料が無い頁を選ぶ」**ことになる。

数える母集団は **候補8点 + Track M の既知セル6頁 = 14頁**。
**母集団 9,653頁への展開は頼まれていない**(Track M: 「要りません」)。

**可視の規則は Track M と合わせる**: **`visibility & 1` が立ち、
かつ<ins>祖先のフォルダーも全部立つ</ins>**。
**`visibility == 1` だけで引くと `3`(表示・マスクあり)を落とす** ——
**マスクを持つ層はまさに `3`** なので、この取り違えはこの発注を無意味にする(条例3)。

**この器具が届く範囲**(名前より狭く書く。T132):

- **届く**: 可視の規則の取り違え / 結合の取り違え(`page_path` + `layer_id` で結合し、
  **名前では結合しない**)/ **Track M の既知セル6頁との食い違い** ——
  **既知セルは Track M が別に数えた値**で、**この器具が作っていない入力**である
- **届かない**: **マスクに形が在るか。** **`has_mask` は構造の欄で、ラスタではない** ——
  **白紙のマスクも True になる**(抽出側へ照会中: Q5「マスクの既定値」)。
  **そして `layer_colour` に行の無い層は `formats` が「空」ではなく「不明」である**
  (条例6: 「一覧に無い」は「存在しない」ではない)。別の列に分けて数える
- **届かない**: **フォルダーのマスクが子孫にどう効くか** —— **census は未測定**。
  **数えるのは葉と別に数えたフォルダーの枚数だけ**で、効き方は抽出側の領分

**T87 についての訂正**(この走行で見つかった。下の `HAS_MASK_NOTE`):
**条例3 は「`has_mask` は全層 True で何も運ばない」と書いていたが、
それは 2,963層の範囲の言明だった** —— 構造 DB の 437,561層では
**True 135,164 / False 302,397 に分かれる。**

    ./venv/bin/python tools/census/mask_material.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import db as census_db  # noqa: E402
import keys as census_keys  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 候補8点。**`colour_pilot.py` の全量側の出力から読む**(手で書き写さない)
CANDIDATES = "data/census_full/colour_pilot_candidates_20261101.csv"
SOURCE = "note_m_order_mask_counts_for_candidates_20261101.md"

# **既知セル**: Track M が自分で数えた6頁の値。**この器具が作っていない入力**(T132)。
# 列は (頁, 可視の葉, 葉の has_mask, t2, t3, 他, 画素なし, 全面, フォルダーの has_mask)
KNOWN_CELLS = (
    (r"F:\manuscripts\2017\illustration\sunshine2\page0019.clip", 117, 65, 65, 0, 0, 65, 0, 3),
    (r"F:\manuscripts\2017\illustration\sunshine2\page0002.clip", 52, 14, 9, 5, 0, 9, 5, 0),
    (r"F:\manuscripts\2018\illust201710\page0003.clip", 18, 0, 0, 0, 0, 0, 0, 0),
    (r"F:\manuscripts\2018\illust201710\page0005.clip", 89, 2, 0, 2, 0, 0, 0, 1),
    (r"F:\manuscripts\2018\skima11\skima12H.clip", 121, 1, 0, 1, 0, 0, 0, 0),
    (r"F:\manuscripts\2018\skima11\skima12chara.clip", 114, 5, 1, 2, 2, 1, 0, 0),
)
KC_COLS = ("n_leaf_visible", "n_leaf_has_mask", "n_hm_type2", "n_hm_type3",
           "n_hm_other", "n_hm_no_pixels", "n_hm_full_alpha", "n_folder_has_mask")

# **可視は祖先まで見る。** 根(`layer_type = 256`、名前は空、1頁に1つ)から降りる。
# **数え方は SQL に畳まず Python で数える** ——
# **畳むと、どの行をどの理由で落としたかを印字できない**(規律9)
ROWS_SQL = """
with recursive vis(page_path, layer_id, folder) as (
    select page_path, layer_id, folder from layers
      where page_path = ? and parent_id is null and (visibility & 1) = 1
  union all
    select l.page_path, l.layer_id, l.folder from layers l
      join vis v on l.page_path = v.page_path and l.parent_id = v.layer_id
      where v.folder and (l.visibility & 1) = 1
)
select v.layer_id, v.folder, l.layer_type, l.has_mask, k.v_layer_kind,
       c.page_path is not null as has_colour_row, c.formats, c.drawn_ratio
  from vis v
  join layers l using (page_path, layer_id)
  join v_layer_kind k using (page_path, layer_id)
  left join layer_colour c using (page_path, layer_id)
"""

NOTE = [
    "**`has_mask` は構造の欄で、ラスタではない。** **白紙のマスクも True になり得る** —— "
    "**マスクの既定値(全面通すか全面遮るか)は抽出側へ照会中**"
    "(Track M の Q5、2026-11-01)。**だからこの表の枚数は材料の上界である**",
    "**`n_hm_no_pixels` と `n_hm_pixels_unknown` は別の量である** —— "
    "前者は**色の表に行が在って `formats` が空**(画素のブロックが無い)、"
    "後者は**色の表に行が無い**(測られていない)。"
    "**`layer_colour` は 72,839層・872頁だけ**なので、"
    "**「一覧に無い」を「存在しない」と読まない**(条例6)",
    "**可視は `visibility & 1` かつ祖先のフォルダーも全部立つ。** "
    "**`visibility == 1` だけで引くと `3`(表示・マスクあり)を落とす** —— "
    "**マスクを持つ層はまさに `3`** なので、この取り違えはこの表を無意味にする(条例3)。"
    "**根は `layer_type = 256`(名前は空、1頁に1つ、9,649頁)**",
    "**`layer_type` の分けは Track M の求め** —— `2` は塗りつぶし/グラデーションで"
    "**画素を持たず**、`3` は**画素を持つがアルファが全面**。"
    "**どちらも形はマスクの側だが、納品の状態が違う**(Track M の実測、2026-11-01)",
    "**フォルダーの `has_mask` は葉と別に数えた。** "
    "**フォルダーのマスクが子孫にどう効くかは census は未測定** —— "
    "**抽出側の領分**(Track M の Q と同じ問い)。"
    "**効くなら、葉の枚数だけでは材料を数え落とす**",
    "**既知セル6頁は Track M が別に数えた値**(この器具が作っていない入力。T132)。"
    "**届く範囲は可視の規則と結合の取り違えまで** —— "
    "**両方が同じ構造 DB から出ているので、壊れた DB でも一致する**",
]
# **T87 の訂正。** 条例3 の文は 2,963層の範囲の言明だった
HAS_MASK_NOTE = (
    "**T87(`has_mask` は全層 True で何も運ばない)の射程を測り直した** —— "
    "**全 437,561層では True {t} / False {f}** に分かれる。"
    "**T87 の 2,963層は Track L の納品の範囲**で、"
    "**条例3 がそれを「全層」と書いていた**(2026-11-01、この表を作って気づいた)。"
    "**点検の名前は、その射程より広く書かれていた**(T132 の規則そのもの)")


def candidates():
    """候補8点を全量側の出力から読む。**手で書き写さない。**"""
    path = os.path.join(ROOT, CANDIDATES)
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["to_author"] == "yes"]
    print(f"    候補: {os.path.basename(path)} の `to_author = yes` {len(rows)} 行")
    return rows


def count_page(con, path):
    """1頁を数える。**落としたものを理由別に数える**(規律9)。"""
    rows = con.execute(ROWS_SQL, [path]).fetchall()
    c = collections.Counter()
    kinds = collections.Counter()
    drops = collections.Counter()
    for (_lid, folder, ltype, hm, kind, has_row, formats, drawn) in rows:
        if folder:
            if hm:
                c["n_folder_has_mask"] += 1
            drops["フォルダー(葉ではない)"] += 1
            continue
        if kind == "paper":
            drops["用紙(`layer_type = 1584`)"] += 1
            continue
        c["n_leaf_visible"] += 1
        if not hm:
            continue
        c["n_leaf_has_mask"] += 1
        kinds[kind] += 1
        c["n_hm_type2" if ltype == 2 else
          "n_hm_type3" if ltype == 3 else "n_hm_other"] += 1
        if not has_row:
            c["n_hm_pixels_unknown"] += 1      # **「空」ではなく「不明」**
        elif formats is None or formats == "":
            c["n_hm_no_pixels"] += 1
        if drawn is not None and drawn == 1.0:
            c["n_hm_full_alpha"] += 1
    return c, kinds, drops


def main() -> int:
    print("  [マスクで形を持つ層の枚数(Track M の発注)]")
    cands = candidates()
    con = census_db.connect()
    try:
        in_db = {r[0]: (r[1], r[2]) for r in
                 con.execute("select page_path, year, format from pages").fetchall()}
        hm_t, hm_f = con.execute(
            "select count(*) filter (where has_mask), "
            "count(*) filter (where not has_mask) from layers").fetchone()
        print(f"    **`has_mask` の射程**: 全 {hm_t + hm_f} 層で"
              f" True {hm_t} / False {hm_f}(**T87 は 2,963層の範囲の言明だった**)")

        order = [(r["page_path"], "候補(作者が見て選ぶ)") for r in cands]
        order += [(p, "既知セル(Track M)") for p, *_ in KNOWN_CELLS]
        missing = [p for p, _ in order if p not in in_db]
        if missing:
            print(f"    **母集団に無い頁 {len(missing)} 件。表を書かずに止める。**")
            return 1

        reg = census_keys.page_registry()
        kc = {p: dict(zip(KC_COLS, v)) for p, *v in [list(k) for k in KNOWN_CELLS]}
        rows, drops_all = [], collections.Counter()
        n_checked = n_disagree = 0
        print(f"    {'頁':>5} {'葉':>5} {'hm':>4} {'t2':>4} {'t3':>4} {'他':>4}"
              f" {'画素なし':>8} {'不明':>5} {'全面':>5} {'フォ':>5}  既知セル")
        for path, role in order:
            c, kinds, drops = count_page(con, path)
            drops_all.update(drops)
            key = reg.key_for(path, where=SOURCE)
            year, fmt = in_db[path]
            verdict = "—"
            if path in kc:
                n_checked += 1
                diff = [k for k in KC_COLS if c[k] != kc[path][k]]
                if diff:
                    n_disagree += 1
                    verdict = "**不一致: " + " ".join(diff) + "**"
                else:
                    verdict = "一致"
            print(f"    {key:>5} {c['n_leaf_visible']:>5} {c['n_leaf_has_mask']:>4}"
                  f" {c['n_hm_type2']:>4} {c['n_hm_type3']:>4} {c['n_hm_other']:>4}"
                  f" {c['n_hm_no_pixels']:>8} {c['n_hm_pixels_unknown']:>5}"
                  f" {c['n_hm_full_alpha']:>5} {c['n_folder_has_mask']:>5}  {verdict}")
            rows.append([key, year, fmt, role,
                         c["n_leaf_visible"], c["n_leaf_has_mask"],
                         c["n_hm_type2"], c["n_hm_type3"], c["n_hm_other"],
                         c["n_hm_no_pixels"], c["n_hm_pixels_unknown"],
                         c["n_hm_full_alpha"], c["n_folder_has_mask"],
                         ";".join(f"{k}:{n}" for k, n in sorted(kinds.items())),
                         verdict,
                         "Track M の発注 / census track が構造 DB から数えた",
                         "2026-11-01", SOURCE, ""])
        reg.report("頁の鍵")
        n_keys = reg.close()
        for why, n in sorted(drops_all.items()):
            print(f"      葉として数えなかった層 {n} —— {why}")
        print(f"    **既知セル {n_checked} 頁 / 不一致 {n_disagree} 頁**"
              "(**この器具が作っていない入力**。T132)")

        note = list(NOTE) + [HAS_MASK_NOTE.format(t=f"{hm_t:,}", f=f"{hm_f:,}")]
        rows += [["", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                  "", "", "", n] for n in note]
        header = ["page_key", "year_label", "format", "role",
                  "n_leaf_visible", "n_leaf_has_mask",
                  "n_hm_type2", "n_hm_type3", "n_hm_other",
                  "n_hm_no_pixels", "n_hm_pixels_unknown", "n_hm_full_alpha",
                  "n_folder_has_mask", "hm_kinds", "known_cell",
                  "by", "as_of", "source", "note"]
        n_pub = bc.write_csv(os.path.join(ROOT, "census", "mask_material.csv"),
                             header, rows)

        # **全量側: 鍵と頁のパスの対応**(公開側には1つも出さない)
        full = os.path.join(ROOT, "data/census_full", "mask_material_subjects.csv")
        with open(full, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["page_key", "page_path", "role"])
            for path, role in order:
                w.writerow([reg.map[path], path, role])
        print(f"    mask_material.csv {n_pub}行 / page_keys.csv {n_keys}行"
              f" / mask_material_subjects.csv {len(order)}行")
        return 1 if n_disagree else 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
