"""集計の材料と、その出所。

**供給元から取った数は、ここで誰かが測るまで、測った者の名を帯びる**(T132)。
だからこの表は「どの数を誰が測ったか」を列に持つ。`measured_by` を空にしてはいけない。
"""

# 供給元が報告した数。**ここでは測っていない。** 再測するには DB 側の材料が要る
# (列 `needs` にそれを書く。足りないものは基盤経由で抽出側へ依頼する)
SUPPLIER_REPORTED = [
    dict(
        key="db_files",
        predicate="構造 DB の `pages` 行 = 原稿ファイル(.clip 7,667 + .lip 1,986)",
        n_pages=9653, n_layers=None,
        measured_by="抽出側(um350)", as_of="2026-10-26",
        needs="`pages` の年ごとの件数(いまは総数しか無い)",
        caveat="ファイル数。**異なり数とは別の量**(T96 の形)",
    ),
    dict(
        key="db_distinct",
        predicate="sha256 で重複を畳んだ異なり数(.clip 6,194 + .lip 1,224)",
        n_pages=7418, n_layers=None,
        measured_by="抽出側(um350)", as_of="2026-10-26",
        needs="`file_hashes`(全 9,653 の sha256)。これが来ればここで再測できる",
        caveat="**ファイル 9,653 と異なり 7,418 は同じ母集団の別の量。**並べて書く",
    ),
    dict(
        key="db_layers",
        predicate="構造 DB の `layers` 行(**`.lip` を入れる前**の数)",
        n_pages=None, n_layers=312898,
        measured_by="抽出側(um350)", as_of="2026-10-24 以前",
        needs="`.lip` 1,986 を入れた後の層数。**まだ更新が来ていない**",
        caveat="**`.lip` 込みの数ではない。** 更新前の値であることを外さない",
    ),
]

# ここで読める CSV。行は自分で数える(ただし **供給元の配列から数えているので、
# 供給元の集計と一致しても独立な確認ではない** —— 転記の一致である。T132)
LOCAL_CSV = [
    dict(
        key="req3_layer_kind",
        path="inbox/f_request3_layer_kind_rendered_20261026.csv",
        predicate="依頼3の872頁の、**フォルダー以外の全層**(種別×描画×drawn_ratio)",
        supplier="抽出側(um350)", as_of="2026-10-26",
        caveat="**フォルダー層は入っていない。** 層数を added33 の `n_layers` と比べない",
    ),
    dict(
        key="added33_structure",
        path="inbox/pages_added33_structure_20261026.csv",
        predicate="後から入った33頁の構造(層数・フォルダー数・画布・dpi)",
        supplier="抽出側(um350)", as_of="2026-10-26",
        caveat="**`canvas_unit` 列が無い。** 行ごとに mm か px かを決められない(T126)",
    ),
]

# **どの材料にも1件も入っていない母集団。** 「0件だった」ではなく「見ていない」
ABSENT = [
    dict(
        key="lip_files",
        predicate="`.lip` ファイル 1,986(異なり 1,224)",
        n_pages=1986,
        measured_by="抽出側(um350)",
        as_of="2026-10-26",
        needs="`.lip` を含む層の表。**いまは1件も手元に無い**",
        caveat="**ここで読める2本の CSV には `.lip` が1頁も入っていない** —— "
               "`ext_lip = 0` は「`.lip` を数えて 0 だった」ではなく"
               "**「材料に `.lip` が無い」**。"
               "この集計の層の数は**すべて `.clip` のもの**(T126 の形: "
               "母集団の境界が読み手の対応形式から来ていて、誰も登録しなかった)",
    ),
]

# **値そのものが近日動く列。** 動いたらこの集計の数は古くなる(基盤 2026-10-26)
VOLATILE = [
    dict(
        column="drawn_ratio",
        scope="`v_layer_kind` = `mask_fill` の層",
        what="抽出側が読み手を直すと、いま 0 と読まれている値が 1 付近へ動く",
        why="`mask_fill`(トーン・べた塗り)の本体は `InitColor 255` = 全面で、0 は「空」ではない",
        n_affected_in_req3=11455,
        note="動かす前に一声もらう約束。**この集計の mask_fill 行は、その連絡で無効になる**",
    ),
]
