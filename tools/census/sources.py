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
        needs="",   # **満たされた**(便 n1 が年 × 形式の頁数を持ってきた)
        satisfied_when=("source", "pages_by_year_all"),
        caveat="ファイル数。**異なり数とは別の量**(T96 の形)。"
               "**年 × 形式の行は `population_by_format.csv`** —— "
               "**`.lip` 1,986頁もこの 9,653 に入っている**"
               "(作者の決定 2026-10-27)",
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
        predicate="構造 DB の `layers` 行のうち **`.clip` の分**。"
                  "**437,561 と同じ走行の数**(便 n2、2026-10-27)",
        n_pages=None, n_layers=312898,
        measured_by="抽出側(um350)/ census track が依頼8 の表から数え直した",
        as_of="2026-10-27",
        needs="",   # **満たされた。** 便 n2 が同じ走行の形式ごとの数を持ってきた
        satisfied_when=("source", "clip_only_layers"),
        caveat="**数は前の走行と同じ 312,898 だが、同じ数であることは同じ走行であることでは"
               "ない** —— 便 n2 で**同じ走行の数として取り直した**(抽出側の言明: "
               "この DB を作り直したとき `.clip` の層の数は変わっていない)。"
               "**これで `437,561 − 312,898` を使わずに `.lip` の 124,663 が言える** —— "
               "`A − B` は量ではなく手続きなので、**同じ走行の形式ごとの数として来るまで"
               "待った。** 形式ごとの行は `population_by_format.csv`",
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
        key="layer_names_all",
        path="inbox/census_r7_layer_names_20261027.csv",
        predicate="**全数の層名**(16,241 の異なる名前 / 437,561層)。"
                  "`is_folder`・年の範囲・種類の内訳・親の名前の異なり数つき",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**これで名前側が全数に届いた。** だが"
               "**素性側(画素の分布)は8頁しか届かない** —— "
               "**両面が一致したと言える範囲は8頁で、名前側の数は全数である。**"
               "**同じ行に並べない。** そして**フォルダーの行を含む**"
               "(`黒消し` は層では0件、フォルダーで 609)",
    ),
    dict(
        key="layer_structure_all",
        path="inbox/census_r8_layer_structure_20261027.csv",
        predicate="**全数の層の構造**(437,561層の `parent_id`/`order_in_parent`/"
                  "`ancestors`/`composite`/`clip`/`visibility`/`opacity`/`z_global`)",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**`z_global` の向きは供給元が v2 で訂正した**(v1 は逆。CSV は正しかった)"
               " —— **こちらでは兄弟の `order_in_parent` との一致でしか確かめていない。**"
               "**画素は入っていない**ので、**何を隠しているかは測れない**"
               "(手前に在ることは重なっていることではない)",
    ),
    dict(
        key="pages_by_year_all",
        path="inbox/census_n1_pages_by_year_20261027.csv",
        predicate="**全数の年 × 形式の頁数**(9,653頁 = `.clip` 7,667 + `.lip` 1,986。22行)",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**頁の数はこの便だけが持っている** —— 依頼8 の表は**層の在る頁**しか"
               "持たないので、**読めなかった 4頁(`.clip`)がそこには出てこない。**"
               "`n_pages` と `n_pages_with_layers` は**別の列**にしてある(T138)",
    ),
    dict(
        key="clip_only_layers",
        path="inbox/census_n2_layers_by_format_20261027.csv",
        predicate="**同じ走行の形式ごとの層数**(`.clip` 312,898 / `.lip` 124,663。"
                  "どちらもフォルダーを含む)",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**行はここから出していない** —— 年の次元が無いので、"
               "`population_by_format.csv` の行は依頼8 の表から数えて、"
               "**この便とは突き合わせだけをする。** "
               "**同じ DB の同じ走行なので、一致は一致であって独立な確認ではない**(T132)",
    ),
    dict(
        key="lip_layers",
        path="inbox/census_n4_lip_layers_20261027.csv",
        predicate="**`.lip` の層の表**(124,663層 / 1,986頁。"
                  "`v_layer_kind`・`layer_type`・`rendered`)",
        supplier="抽出側(um350)", as_of="2026-10-27",
        caveat="**`rendered` は全行が空** —— 抽出側の言明は"
               "**「描けなかったのではなく、描いていない」。**"
               "**走行のたびに全行を数えて確かめている**(言明と中身の突き合わせ)。"
               "**この便の層の属性は依頼8 の表にも在る**ので、"
               "**行はこの便から出さず、突き合わせに使う**(T132)",
    ),
    dict(
        key="added33_structure",
        path="inbox/pages_added33_structure_20261026.csv",
        predicate="後から入った33頁の構造(層数・フォルダー数・画布・dpi)",
        supplier="抽出側(um350)", as_of="2026-10-26",
        caveat="**`canvas_unit` 列が無い。** 行ごとに mm か px かを決められない(T126)",
    ),
]

# **届いているが、まだこの集計に入っていない便。**
#
# **これを書く理由**: 「数が無い」には2つ在る —— **材料が無い**のと
# **材料は在るが誰も入れていない**。`needs` の欄は前者を指すので、
# **後者をそこに混ぜると「頼んである」と読まれる。** 別の列に出す。
ARRIVED_NOT_INTEGRATED = [
    dict(path="inbox/census_n5_added33_canvas_20261027.csv",
         supplies="canvas_px",
         what="33頁の `canvas_unit`(mm 31頁 / px 2頁)と画素の画布",
         why_not_yet="**`canvas_declared.csv` の単位の列を埋める作業がまだ**"),
    dict(path="inbox/mask_cover_request3_20261026.csv",
         supplies=None,
         what="依頼3のマスクの被覆(16,273行・766頁)",
         why_not_yet="**§3-4 の納品の覆いに入れる作業がまだ**。"
                     "`mask_cover` は下限ではないが、**`LayerMaskOffset` を足していない**ので"
                     "ずれた層では被覆が動く(幅と層数は未測定)"),
]


def audit_arrived() -> int:
    """**宣言した便が実在するか。** 無い便を「届いている」と書いたら指摘。"""
    import os
    bad = []
    print("  [届いているが未統合の便]")
    for a in ARRIVED_NOT_INTEGRATED:
        here = os.path.exists(a["path"])
        if not here:
            bad.append(a["path"])
        print(f"    {os.path.basename(a['path']):46s} "
              f"{'在る' if here else '**不在**'}  {a['what']}")
    for path in bad:
        print(f"    **{path} を「届いている」と書いたが、ファイルが無い**")
    return 1 if bad else 0


# **どの材料にも1件も入っていない母集団。** 「0件だった」ではなく「見ていない」
ABSENT = [
    dict(
        key="lip_files",
        predicate="`.lip` ファイル 1,986(異なり 1,224)/ 124,663層。"
                  "**母集団に入っている**(作者の決定 2026-10-27: 「入れる」)。"
                  "**ただし画素は1頁も無い**",
        n_pages=1986,
        measured_by="抽出側(um350)",
        as_of="2026-10-27",
        needs="**`.lip` の画素。** 便 n4 の `rendered` は**全行が空**で、"
              "抽出側の言明は**「描けなかったのではなく、描いていない」。** "
              "**描く発注を出すかどうかは、まだ誰も決めていない**",
        satisfied_when=("source", "lip_rendered"),
        caveat="**「母集団に入っている」と「全部の表に入っている」は別である** —— "
               "**名前の側(依頼7)と構造の側(依頼8)には前から入っていた**"
               "(437,561 = `.clip` 312,898 + `.lip` 124,663)。"
               "**素性の側(画素の分布)には 0 頁。** "
               "`ext_lip = 0` の行は「`.lip` を数えて 0 だった」ではなく"
               "**「その材料に `.lip` が無い」**(T126)。"
               "**形式ごとの行は `population_by_format.csv`**",
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
            # **「未着」と「届いているが未統合」を言い分ける** ——
            # 同じ語にすると、**入れ忘れが依頼待ちに見える**
            pending = any(a.get("supplies") == key for a in ARRIVED_NOT_INTEGRATED)
            state = (f"{key} は在る" if satisfied else
                     f"{key} は**届いているが未統合**" if pending else
                     f"{key} は未着")
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
