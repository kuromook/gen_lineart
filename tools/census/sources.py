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
        satisfied_when=("source", "pages_by_year_all"),
        caveat="ファイル数。**異なり数とは別の量**(T96 の形)",
    ),
    dict(
        key="db_distinct",
        predicate="sha256 で重複を畳んだ異なり数(.clip 6,194 + .lip 1,224)",
        n_pages=7418, n_layers=None,
        measured_by="census track(2026-10-26 に全 9,653 行から数え直した)",
        as_of="2026-10-26",
        needs="",   # **満たされた。** `request9` の sha256 で測れた
        satisfied_when=("source", "file_sha256"),
        caveat="**ファイル 9,653 と異なり 7,418 は同じ母集団の別の量。**並べて書く。"
               "**ハッシュは抽出側が計算した列**なので、先方の読みが違えば同じく違う",
    ),
    dict(
        key="clip_files",
        predicate="`.clip` ファイルのみ 7,667(異なり 6,194)。"
                  "**重複の3つの量はこの母集団の数**",
        n_pages=7667, n_layers=None,
        measured_by="抽出側(um350)/ 重複は census track が数え直した",
        as_of="2026-10-26",
        needs="",
        satisfied_when=None,
        caveat="**7,667 − 6,194 を「複製の数」と名乗らない** —— "
               "余り 1,473 / 双子を持つファイル 2,458 / 塊 985 は**3つの別の量**",
    ),
    dict(
        key="db_layers",
        predicate="構造 DB の `layers` 行、**`.clip` だけ・`.lip` を入れる前**の数",
        n_pages=None, n_layers=312898,
        measured_by="抽出側(um350)", as_of="2026-10-24 以前",
        needs="**`.clip` だけの層数を、437,561 と同じ走行で取り直したもの。** "
              "いまの 312,898 はそれより前の走行の数で、"
              "**同じ走行の数でないと `.clip` と `.lip` を比べられない**",
        satisfied_when=("source", "clip_only_layers"),
        caveat="**全数の問いについては `db_all_layers`(437,561)が後継。** "
               "この行は `.clip` だけの数として残すが、**取り直していない。** "
               "**437,561 − 312,898 を『`.lip` の層数』と名乗らない** —— "
               "別の走行の数の引き算で、`A − B` は量ではなく手続きである",
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
        key="body_initcolor",
        path="inbox/body_initcolor_delivered_20261026.csv",
        predicate="トーン・グラデ・ベタの**本体の InitColor**(1層1行、2,095層)",
        supplier="抽出側(um350)", as_of="2026-10-26",
        satisfies="body_initcolor",
        caveat="**1層1行なのでここで測れる。** `set` 列に作品名が入るので"
               "**段 A では種類に畳む**(畳んだ世代の数を印字する)",
    ),
    dict(
        key="file_sha256",
        path="inbox/request9_file_sha256_all_20261026.csv",
        predicate="**全 9,653 ファイルの `page_path + sha256`**(1ファイル1行)",
        supplier="抽出側(um350)", as_of="2026-10-26",
        caveat="**これで異なり数と重複の3つの量がここで測れる。** ただし"
               "**ハッシュそのものは抽出側が計算した** —— 先方の読みが違えば同じく違う",
    ),
    dict(
        key="req3_body_blocks",
        path="inbox/faint_alpha_905_20261027.csv",
        predicate="905頁の `body_init` / `body_blocks` / `body_stored` と"
                  "**アルファ≥1 の比**(62,105層。依頼3の 56,576層を全部含む)",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**依頼3の3列がこれで埋まった** —— 下限の層が何層かを計数できる。"
               "**併せて `drawn_ratio` の閾 26 が落としている薄い画素が見える**",
    ),
    dict(
        key="grad_v1_v2_map",
        path="inbox/grad_v1_v2_layer_map_20261026.csv",
        predicate="グラデ v1/v2 の**層ごとの対応**(258層)",
        supplier="抽出側(um350)", as_of="2026-10-26",
        caveat="**層を鍵にした突き合わせはこれが無いとできない** —— "
               "PNG の名前は頁の中の順番を encode している",
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
        satisfied_when=("source", "lip_layers"),
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


def by_key(key: str) -> dict:
    """材料を**鍵で引く。** 添字で引くと、材料を1つ挿した日に別の CSV を読む
    (2026-10-26 に実際そうなった: `LOCAL_CSV[1]` が added33 から InitColor に変わった)。
    """
    for s in LOCAL_CSV:
        if s["key"] == key:
            return s
    raise KeyError(f"材料が宣言されていない: {key}")


# **`needs` が古くなっていないかの点検。**
# 2026-10-26 に `db_layers` の `needs` が「`.lip` 後の層数はまだ来ていない」と
# 書いたままで、**`db_all_layers`(437,561)がその更新だった** ——
# **同じ表の2行が食い違っていた。** 手で見つけたので、**次を器具が見つける形にする。**
#
# `satisfied_when = ("source", <鍵>)` は「その材料が `LOCAL_CSV` に在れば満たされた」。
# **満たされているのに `needs` が空でない行**は指摘。
# **満たされていないのに `needs` が空の行**も指摘(頼み忘れ)。


def audit_needs() -> int:
    """**この点検が届く範囲**(名前より狭く書く):

    - **届く**: 宣言した材料が `LOCAL_CSV` に入ったのに `needs` が残っている /
      入っていないのに `needs` が空(頼み忘れ)
    - **届かない**: **`satisfied_when` を宣言していない行**(`clip_files` のように
      条件なしの行は、何も点検されない)。そして**材料が届いても中身が
      その問いに答えていない場合** —— 鍵が在ることと、答えが在ることは別
    """
    rows = SUPPLIER_REPORTED + ABSENT
    have = {s["key"] for s in LOCAL_CSV}
    bad = []
    print("  [`needs` 欄の点検] 満たされた材料が来たのに `needs` が残っていないか")
    for s in rows:
        cond = s.get("satisfied_when")
        needs = (s.get("needs") or "").strip()
        if cond is None:
            state = "条件なし"
            ok = True
        else:
            kind, key = cond
            satisfied = kind == "source" and key in have
            state = f"{key} は{'在る' if satisfied else '未着'}"
            ok = (not needs) if satisfied else bool(needs)
            if not ok:
                bad.append((s["key"], state, needs))
        print(f"    {s['key']:16s} {state:28s} needs={'空' if not needs else '有り'}"
              f"  {'' if ok else '**食い違い**'}")
    if bad:
        for key, state, needs in bad:
            print(f"    **{key}: {state} なのに needs が"
                  f"{'残っている' if needs else '空'}**")
        return 1
    print("    食い違いなし")
    return 0
